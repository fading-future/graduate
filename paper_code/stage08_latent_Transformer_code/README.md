# stage08_latent_Transformer_code

This directory contains a faithful, appendix‑driven implementation of the two‑stage pipeline described in **Ren 等 (2024)**:

1. **VQ‑VAE** compresses 3D binary porous media into discrete latent codebook indices.
2. **Transformer (NanoGPT‑style)** autoregressively models sequences of codebook indices, conditioned on spatial porosity.

The default hyperparameters and architecture settings below are taken from the paper’s **Appendix A/B** and wired into the configs.

---

## Model Fidelity to Appendix

### VQ‑VAE (Appendix A)
- Encoder channels: `[16, 64, 128, 256, 512]`
- Decoder channels: `[512, 256, 256, 64, 16, 16]`
- Residual blocks: **2 (encoder)**, **3 (decoder)**
- GroupNorm groups: **16**
- Latent dimension: **256**
- Codebook size: **3000**
- Commitment beta: **1**
- Training: **25 epochs**, **batch=20**, **lr=5e‑4**, **betas=(0.9,0.999)**
- Codebook loss weight schedule: `0.02 + 0.02/epoch` up to `2`

### Transformer (Appendix B)
- Vocab size: **3001** (3000 codebook + SOS)
- Block size: **512** (= 8 patches × 64 tokens)
- Layers/heads/emb: **12 / 12 / 1080**
- Dropout: **0.01**
- Conditional dim: **1**, conditional embedding: **100**
- SOS token: **3000**
- Training: **50 epochs**, **batch=32**, **lr=2e‑4**

### Important Implementation Notes
- **Latent tokens per 64³ patch = 4×4×4 = 64**, matching Appendix B (`Number of Features = 64`).
- Decoder upsampling flags are set so that **64³ → 4³ → 64³** round‑trip is preserved. For decoder channels `[512,256,256,64,16,16]`, we use one non‑upsampling stage at 256 and upsample the remaining transitions.
- The transformer is **NanoGPT‑style causal** with porosity conditioning added as an embedding sum. Conditioning is aligned to **target positions** (not shifted), following Eq.(7) logic in the paper.

---

## Data Paths (your setup)
The configs are already set to your paths:
- `raw_data_dir: /chendou_space/data/binary_Training_Data`
- `porosity_csv: /chendou_space/data/aligned_Training_Data/processing_report.csv`

By default the pipeline **computes patch porosity directly from binary voxels** (`porosity_source: compute`).  
If your CSV contains **per‑file porosity only** (like your `file, rel_path, porosity, ...`), you can set `porosity_source: csv` and the loader will **broadcast the file‑level porosity to all 8 sub‑patches** inside each 128³ sample.  
If you later add per‑patch indices to the CSV, the loader will automatically use them instead.

---

## Project Layout
```
configs/
  vqvae.yaml
  transformer.yaml
src/
  data/
    dataset.py
    patching.py
    porosity.py
  models/
    vqvae.py
    transformer.py
    quantizer.py
  utils/
    checkpoint.py
    config.py
    seed.py
  train_vqvae.py
  train_transformer.py
  infer_generate.py
requirements.txt
```

---

## Training

### 1) Train VQ‑VAE
```
python src/train_vqvae.py --config configs/vqvae.yaml --out_dir outputs/vqvae
```

### 2) Train Transformer
```
python src/train_transformer.py \
  --config configs/transformer.yaml \
  --vqvae_ckpt outputs/vqvae/vqvae_epoch_25.pt \
  --out_dir outputs/transformer
```

---

## Inference (Generate Large Volumes)
You need a porosity grid `.npy` (shape: `[Z, Y, X]` in patch units). Example for a 256³ target with 64³ patches → grid shape `4×4×4`.

```
python src/infer_generate.py \
  --config configs/transformer.yaml \
  --vqvae_ckpt outputs/vqvae/vqvae_epoch_25.pt \
  --transformer_ckpt outputs/transformer/transformer_epoch_50.pt \
  --porosity_grid /path/to/porosity_grid.npy \
  --out_path outputs/generated_volume.npy
```

---

## If You Want Me To Extend
I can add:
- precompute token cache for transformer training
- automatic porosity‑grid extraction from any NPY volume
- distributed / mixed‑precision training

Just say the word and I’ll wire it in.
