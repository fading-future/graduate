import os
import glob
import random
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from src.config import CONFIG

class LatentDataset(Dataset):
    def __init__(self, data_dir, augment=True, overfit_num_samples: int = 0):
        self.augment = augment 
        self.file_list = []

        if isinstance(data_dir, str):
            data_dir_list = [data_dir]
        else:
            data_dir_list = data_dir
            
        print(f"Loading data from {len(data_dir_list)} directories...")
        for d in data_dir_list:
            if not os.path.exists(d):
                print(f"⚠️ Warning: Directory not found: {d}")
                continue
            files = sorted(glob.glob(os.path.join(d, "*.npy")))
            self.file_list.extend(files)
            
        if len(self.file_list) == 0:
            raise ValueError("No .npy files found! Check config paths.")

        if overfit_num_samples and overfit_num_samples > 0:
            self.file_list = self.file_list[:overfit_num_samples]
            print(f"🧪 Overfit mode: using first {len(self.file_list)} samples (augment={self.augment})")

        print(f"Total LatentDataset size: {len(self.file_list)} files.")

    def __len__(self):
        return len(self.file_list)

    def extract_porosity(self, filename):
        # 支持 porosity_0.123 / porosity_1 / porosity_.5 等
        match = re.search(r'porosity_([0-9]*\.?[0-9]+)', filename)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        return 0.5

    def apply_mask(
        self,
        latent,
        force_mode: str = None,
        force_axis: str = None,
        force_side: str = None,
        force_cut: int = None,
    ):
        """
        增强版：支持 overfit 时强制固定 mask
        force_mode: 'half' / 'two_halves' / 'corner_missing' / 'random_box'
        force_axis: 'D'/'H'/'W'（对 half / two_halves）
        force_side: 'low'/'high'（对 half）
        force_cut : half 模式切分点（整数）
        """
        C, D, H, W = latent.shape
        mask = torch.zeros((1, D, H, W), dtype=torch.float32)  # 默认全未知

        # ---- 模式选择 ----
        if force_mode is None:
            mode = random.choices(
                ['half', 'two_halves', 'corner_missing', 'random_box'],
                weights=[80, 10, 5, 5],
                k=1
            )[0]
        else:
            mode = force_mode

        # ========== A) 已知一半 -> 推理另一半 ==========
        if mode == 'half':
            axis = force_axis if force_axis is not None else 'D'   # 你原来强制 D，这里保持默认 D
            side = force_side if force_side is not None else 'low' # 你原来强制 low，这里保持默认 low

            if axis == 'D':
                if force_cut is None:
                    cut = random.randint(int(D * 0.45), int(D * 0.55))
                else:
                    cut = int(force_cut)
                cut = max(1, min(cut, D - 1))
                if side == 'low':
                    mask[:, :cut, :, :] = 1.0
                else:
                    mask[:, cut:, :, :] = 1.0

            elif axis == 'H':
                if force_cut is None:
                    cut = random.randint(int(H * 0.45), int(H * 0.55))
                else:
                    cut = int(force_cut)
                cut = max(1, min(cut, H - 1))
                if side == 'low':
                    mask[:, :, :cut, :] = 1.0
                else:
                    mask[:, :, cut:, :] = 1.0

            else:  # 'W'
                if force_cut is None:
                    cut = random.randint(int(W * 0.45), int(W * 0.55))
                else:
                    cut = int(force_cut)
                cut = max(1, min(cut, W - 1))
                if side == 'low':
                    mask[:, :, :, :cut] = 1.0
                else:
                    mask[:, :, :, cut:] = 1.0

        # ========== B) 已知两半 -> 推理中间 ==========
        elif mode == 'two_halves':
            axis = force_axis if force_axis is not None else random.choice(['D', 'H', 'W'])
            middle_ratio = random.uniform(0.25, 0.50)
            mask.fill_(1.0)
            if axis == 'D':
                mid = int(D * middle_ratio); start = (D - mid) // 2; end = start + mid
                mask[:, start:end, :, :] = 0.0
            elif axis == 'H':
                mid = int(H * middle_ratio); start = (H - mid) // 2; end = start + mid
                mask[:, :, start:end, :] = 0.0
            else:
                mid = int(W * middle_ratio); start = (W - mid) // 2; end = start + mid
                mask[:, :, :, start:end] = 0.0

        # ========== C) 缺角 ==========
        elif mode == 'corner_missing':
            mask.fill_(1.0)
            corner_ratio = random.uniform(0.25, 0.55)
            cd = max(2, int(D * corner_ratio))
            ch = max(2, int(H * corner_ratio))
            cw = max(2, int(W * corner_ratio))

            d_side = random.choice(['low', 'high'])
            h_side = random.choice(['low', 'high'])
            w_side = random.choice(['low', 'high'])

            d_slice = slice(0, cd) if d_side == 'low' else slice(D - cd, D)
            h_slice = slice(0, ch) if h_side == 'low' else slice(H - ch, H)
            w_slice = slice(0, cw) if w_side == 'low' else slice(W - cw, W)
            mask[:, d_slice, h_slice, w_slice] = 0.0

        # ========== D) 随机挖洞 ==========
        else:  # 'random_box'
            mask.fill_(1.0)
            hole_d = random.randint(max(2, int(D * 0.25)), max(3, int(D * 0.70)))
            hole_h = random.randint(max(2, int(H * 0.25)), max(3, int(H * 0.70)))
            hole_w = random.randint(max(2, int(W * 0.25)), max(3, int(W * 0.70)))
            hole_d = min(hole_d, D - 1); hole_h = min(hole_h, H - 1); hole_w = min(hole_w, W - 1)
            z0 = random.randint(0, D - hole_d); y0 = random.randint(0, H - hole_h); x0 = random.randint(0, W - hole_w)
            mask[:, z0:z0 + hole_d, y0:y0 + hole_h, x0:x0 + hole_w] = 0.0

        masked_latent = latent * mask
        return masked_latent, mask


    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        filename = os.path.basename(file_path)

        try:
            data_numpy = np.load(file_path)
        except:
            return self.__getitem__(random.randint(0, len(self.file_list)-1))

        latent = torch.from_numpy(data_numpy).float()

        latent = latent * CONFIG['scale_factor']
        latent = torch.clamp(latent, min=-CONFIG['safe_threshold'], max=CONFIG['safe_threshold'])
        if latent.dim() == 5:
            latent = latent.squeeze(0)

        # ---------------- overfit 模式：强制可控 ----------------
        overfit_n = int(CONFIG.get("overfit_num_samples", 0))
        overfit_fixed = bool(CONFIG.get("overfit_fixed_mask", False))
        overfit_seed = int(CONFIG.get("overfit_seed", 1234))

        if overfit_n > 0:
            # 1) 关闭增强（确保同一输入）
            do_augment = False

            # 2) 固定随机性：保证每次 __getitem__ 返回同一个 mask
            #    这里用 idx 固定，确保不同样本也可控
            random.seed(overfit_seed + idx)
            np.random.seed(overfit_seed + idx)
            torch.manual_seed(overfit_seed + idx)
        else:
            do_augment = self.augment

        # ---------------- 数据增强（原逻辑不动，只是受 do_augment 控制） ----------------
        if do_augment:
            if random.random() > 0.5: latent = torch.flip(latent, dims=[1])
            if random.random() > 0.5: latent = torch.flip(latent, dims=[2])
            if random.random() > 0.5: latent = torch.flip(latent, dims=[3])

        porosity = self.extract_porosity(filename)
        porosity = torch.tensor([porosity], dtype=torch.float32)

        # ---------------- mask（overfit 时固定成你真实任务：沿 D 轴切一半） ----------------
        if overfit_n > 0 and overfit_fixed:
            # 固定：D 轴，low 已知，cut = D//2
            cut = latent.shape[1] // 2
            condition_latent, mask = self.apply_mask(
                latent,
                force_mode="half",
                force_axis="D",
                force_side="low",
                force_cut=cut,
            )
        else:
            condition_latent, mask = self.apply_mask(latent)

        return {
            "GT": latent,
            "Condition": condition_latent,
            "Mask": mask,
            "Porosity": porosity
        }
