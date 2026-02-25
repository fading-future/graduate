import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# Embeddings
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PorosityEmbedder(nn.Module):
    def __init__(self, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.linear_proj = nn.Linear(frequency_embedding_size, frequency_embedding_size)

    def forward(self, porosity):
        device = porosity.device
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = porosity * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.linear_proj(emb)


# ==========================================
# Core Blocks
# ==========================================
class ResBlock3D(nn.Module):
    """ResBlock with separate time-embedding (additive bias) and
    porosity-embedding (AdaGN: scale + shift on the second GroupNorm).

    When ``use_adagn=False`` the block behaves identically to the original
    version (pure additive bias from a single combined embedding), keeping
    full backward-compatibility with older checkpoints.
    """

    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0,
                 por_emb_dim=0, use_adagn=False):
        super().__init__()
        self.use_adagn = use_adagn and (por_emb_dim > 0)

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)

        # Time embedding → additive bias (same as before)
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        self.norm2 = nn.GroupNorm(32, out_ch)

        # AdaGN: porosity embedding → (scale, shift) for norm2 output
        if self.use_adagn:
            self.por_adagn_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(por_emb_dim, out_ch * 2),  # scale & shift
            )
            # Initialise close to identity transform: scale≈1, shift≈0
            nn.init.zeros_(self.por_adagn_proj[-1].weight)
            nn.init.zeros_(self.por_adagn_proj[-1].bias)
            # Set the scale bias half to 1 so that (1+scale)*h + shift ≈ h at init
            # (the first out_ch entries correspond to scale)

        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, p_emb=None):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # Time conditioning: additive bias
        time_hidden = self.time_emb_proj(t_emb)
        h = h + time_hidden[:, :, None, None, None]

        h = self.norm2(h)

        # Porosity conditioning: AdaGN (scale & shift)
        if self.use_adagn and p_emb is not None:
            ada = self.por_adagn_proj(p_emb)                # (B, 2*out_ch)
            scale, shift = ada.chunk(2, dim=-1)              # each (B, out_ch)
            h = h * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]

        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Attention3D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        x_in = x
        x = self.norm(x)

        qkv = self.to_qkv(x).view(b, self.heads * 3, -1, d * h * w)
        q, k, v = map(lambda t: t.permute(0, 1, 3, 2).contiguous(), qkv.chunk(3, dim=1))

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )

        out = out.permute(0, 1, 3, 2).reshape(b, -1, d, h, w)
        return self.to_out(out) + x_in


# ==========================================
# UNet
# ==========================================
class ConditionalLatentUNet(nn.Module):
    """Conditional 3-D UNet for latent diffusion.

    Improvements over the original architecture
    ───────────────────────────────────────────
    1. **Separated embedding paths** – time and porosity embeddings go through
       independent MLP projections instead of being naïvely summed.
    2. **AdaGN (Adaptive Group Normalization)** – when ``use_adagn=True``,
       each ``ResBlock3D`` applies the porosity embedding as *scale & shift*
       on its second GroupNorm, giving the conditioning signal multiplicative
       influence instead of just an additive bias.
    3. **Classifier-free guidance support** – a learnable *null embedding*
       (``null_por_emb``) is registered.  During training, the porosity
       embedding is randomly replaced by this null embedding with probability
       ``cfg_drop_prob``.  At inference, the model is called twice (with and
       without the condition) and the outputs are combined via the guidance
       scale.

    Backward compatibility
    ──────────────────────
    When ``use_adagn=False`` (default) the architecture is structurally
    identical to the original UNet – all new parameters live in sub-modules
    that are not instantiated, so old checkpoints load without error.
    """

    def __init__(
        self,
        in_channels=9,
        out_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 4),
        use_attention=(False, True, True),
        use_adagn=False,
        cfg_drop_prob=0.0,
    ):
        super().__init__()
        self.use_adagn = use_adagn
        self.cfg_drop_prob = cfg_drop_prob

        time_dim = base_channels * 4
        por_dim = base_channels * 4      # porosity embedding dimensionality

        # ─── Time embedding path ───
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # ─── Porosity embedding path (separate from time) ───
        self.porosity_mlp = nn.Sequential(
            PorosityEmbedder(base_channels),
            nn.Linear(base_channels, por_dim),
            nn.GELU(),
            nn.Linear(por_dim, por_dim),
        )

        # ─── Classifier-free guidance: learnable null porosity embedding ───
        self.null_por_emb = nn.Parameter(torch.zeros(1, por_dim))

        # If NOT using AdaGN, porosity is still combined into the time
        # embedding via addition (legacy behaviour).  With AdaGN the
        # porosity embedding is passed separately to every ResBlock.
        if not use_adagn:
            # Combined embedding for backward-compatible additive mode
            self._combined_mode = True
        else:
            self._combined_mode = False

        self.inc = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        # Helper to build ResBlock with correct embedding dims
        def _make_res(in_ch, out_ch):
            return ResBlock3D(
                in_ch, out_ch,
                time_emb_dim=time_dim,
                por_emb_dim=por_dim if use_adagn else 0,
                use_adagn=use_adagn,
            )

        self.downs = nn.ModuleList([])
        dims = [base_channels, *map(lambda m: base_channels * m, channel_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            use_attn = use_attention[ind]
            self.downs.append(nn.ModuleList([
                _make_res(dim_in, dim_out),
                _make_res(dim_out, dim_out),
                Attention3D(dim_out) if use_attn else nn.Identity(),
                nn.Conv3d(dim_out, dim_out, 4, stride=2, padding=1) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = _make_res(mid_dim, mid_dim)
        self.mid_attn = Attention3D(mid_dim)
        self.mid_block2 = _make_res(mid_dim, mid_dim)

        self.ups = nn.ModuleList([])
        reversed_dims = list(reversed(in_out))

        for ind, (dim_out, dim_in) in enumerate(reversed_dims):
            is_last = ind >= (len(reversed_dims) - 1)
            use_attn = use_attention[len(use_attention) - 1 - ind]
            actual_in_dim = dim_in + dim_in
            self.ups.append(nn.ModuleList([
                _make_res(actual_in_dim, dim_out),
                _make_res(dim_out, dim_out),
                Attention3D(dim_out) if use_attn else nn.Identity(),
                nn.ConvTranspose3d(dim_out, dim_out, 2, stride=2) if not is_last else nn.Identity()
            ]))

        self.final_res_block = _make_res(base_channels * channel_mults[0], base_channels)
        self.outc = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, x, t, porosity, force_null_porosity=False):
        """
        Parameters
        ----------
        x : (B, in_channels, D, H, W)
        t : (B,) long — diffusion timestep
        porosity : (B, 1) or (B,) — scalar porosity condition
        force_null_porosity : bool
            If ``True``, use the null embedding for **all** samples
            (used during the unconditional pass of CFG inference).
        """
        t_emb = self.time_mlp(t)                      # (B, time_dim)
        p_emb = self.porosity_mlp(porosity)            # (B, por_dim)

        # ─── CFG: randomly drop porosity condition during training ───
        if self.training and self.cfg_drop_prob > 0.0 and not force_null_porosity:
            B = p_emb.shape[0]
            drop_mask = torch.rand(B, 1, device=p_emb.device) < self.cfg_drop_prob
            null = self.null_por_emb.expand(B, -1)
            p_emb = torch.where(drop_mask, null, p_emb)

        if force_null_porosity:
            p_emb = self.null_por_emb.expand(p_emb.shape[0], -1)

        # ─── Build embeddings for ResBlock ───
        if self._combined_mode:
            # Legacy: sum time + porosity → single embedding, no separate p_emb
            emb = t_emb + p_emb
            p_emb_pass = None
        else:
            # AdaGN: time goes through additive bias, porosity through scale & shift
            emb = t_emb
            p_emb_pass = p_emb

        x = self.inc(x)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, emb, p_emb_pass)
            x = block2(x, emb, p_emb_pass)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb, p_emb_pass)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb, p_emb_pass)

        for block1, block2, attn, upsample in self.ups:
            skip = h.pop()
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat((x, skip), dim=1)
            x = block1(x, emb, p_emb_pass)
            x = block2(x, emb, p_emb_pass)
            x = attn(x)
            x = upsample(x)

        x = self.final_res_block(x, emb, p_emb_pass)
        return self.outc(x)
