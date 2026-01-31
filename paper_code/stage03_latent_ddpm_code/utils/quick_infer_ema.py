# quick_infer_ema.py
"""
Quick inference on EMA model:
- Finds latest unet_epoch_*.pth in exp_results/<experiment>/models
- Loads diffusion model + vqvae (load_models from inference.py handles EMA weights if present)
- Randomly selects 5 samples from processed_data_dir[0], runs DDIM sampling (default steps from CONFIG)
- Decodes with VQ-VAE and computes statistics:
    - GT mean/std, GEN mean/std
    - Unknown-region MSE, MAE, MaxAbs
- Saves visualizations (by calling inference.visualize_inference_results)
- Writes CSV summary to exp_results/.../quick_infer_summary.csv
"""
import os
import glob
import random
import csv
import numpy as np
import torch
from src.config import CONFIG
from utils.get_root_path import get_project_root
import src.inference as infer_mod  # reuse functions: load_models, ddim_sample, visualize_inference_results, etc.

def compute_stats(recon_gt, recon_gen, mask_pixel_np):
    # recon_* are numpy arrays (D,H,W)
    gt = recon_gt
    gen = recon_gen
    gt_flat = gt.flatten()
    gen_flat = gen.flatten()
    stats = {}
    stats['gt_mean'] = float(np.mean(gt_flat))
    stats['gt_std'] = float(np.std(gt_flat))
    stats['gen_mean'] = float(np.mean(gen_flat))
    stats['gen_std'] = float(np.std(gen_flat))
    # unknown region (mask_pixel_np == 0)
    unknown_idx = (mask_pixel_np == 0)
    if unknown_idx.sum() > 0:
        diff = gen[unknown_idx] - gt[unknown_idx]
        stats['unknown_mse'] = float(np.mean((diff) ** 2))
        stats['unknown_mae'] = float(np.mean(np.abs(diff)))
        stats['unknown_maxabs'] = float(np.max(np.abs(diff)))
    else:
        stats['unknown_mse'] = None
        stats['unknown_mae'] = None
        stats['unknown_maxabs'] = None
    return stats

def main():
    device = CONFIG['device']
    root = get_project_root()
    models_dir = os.path.join(root, "exp_results", CONFIG['experiment_name'], "models")
    model_files = sorted(glob.glob(os.path.join(models_dir, "unet_epoch_*.pth")), key=os.path.getmtime)
    if not model_files:
        print("❌ No models found")
        return
    model_path = model_files[-1]
    print(f"Using model: {model_path}")

    diffusion_model, vqvae_model = infer_mod.load_models(model_path, device)

    # select sample files
    data_dir = CONFIG['processed_data_dir'][0]
    data_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    if not data_files:
        print("❌ No data files found in", data_dir)
        return

    sample_count = min(10, len(data_files))
    chosen = random.sample(data_files, sample_count)

    save_dir = os.path.join(root, "exp_results", CONFIG['experiment_name'], "quick_infer")
    os.makedirs(save_dir, exist_ok=True)

    summary_path = os.path.join(save_dir, "quick_infer_summary.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['fname', 'gt_mean', 'gt_std', 'gen_mean', 'gen_std', 'unknown_mse', 'unknown_mae', 'unknown_maxabs']
            writer.writerow(header)

    for sample_path in chosen:
        fname = os.path.basename(sample_path)
        print(f"\n🔎 Processing sample: {fname}")

        # load latent GT
        gt_raw = np.load(sample_path)
        gt_tensor = torch.from_numpy(gt_raw).float().to(device)

        # scale
        scale = CONFIG['scale_factor']
        gt_scaled = gt_tensor * scale
        safe_thresh = CONFIG.get('safe_threshold', 6.0)
        gt_scaled = torch.clamp(gt_scaled, min=-safe_thresh, max=safe_thresh)

        # build mask (top 50% known)
        D = CONFIG['image_size']
        mask = torch.zeros_like(gt_scaled)
        split_point = int(D * 0.5)
        mask[..., :split_point, :, :] = 1.0
        mask_input = mask[:, 0:1, ...]
        condition = gt_scaled * mask_input

        # porosity
        import re
        m = re.search(r'porosity_(\d+\.\d+)', fname)
        poro_val = float(m.group(1)) if m else 0.15
        porosity = torch.tensor([poro_val]).to(device).view(1,1)

        # sampling (DDIM by default)
        with torch.no_grad():
            x_gen_scaled = infer_mod.ddim_sample(diffusion_model, condition, mask_input, porosity, device,
                                                 ddim_steps=CONFIG.get('ddim_steps_infer', 200))
            gen_restored = x_gen_scaled / scale
            recon_gen = vqvae_model.decode(gen_restored).cpu().numpy()[0,0]
            recon_gt = vqvae_model.decode(gt_tensor).cpu().numpy()[0,0]

            # mask to pixel
            mask_pixel = torch.nn.functional.interpolate(mask_input, scale_factor=4, mode='nearest')
            mask_pixel_np = mask_pixel[0,0].cpu().numpy()

            # compute stats
            stats = compute_stats(recon_gt, recon_gen, mask_pixel_np)
            row = [fname, stats['gt_mean'], stats['gt_std'], stats['gen_mean'], stats['gen_std'],
                   stats['unknown_mse'], stats['unknown_mae'], stats['unknown_maxabs']]
            with open(summary_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # save visualizations using inference's helper
            viz_path = os.path.join(save_dir, f"{fname}_quick_LDM.png")
            infer_mod.visualize_inference_results(recon_gt, (recon_gt * 0), recon_gen, mask_pixel_np, viz_path, fname)

            # also save nifti of gen
            try:
                import nibabel as nib
                nib.save(nib.Nifti1Image(recon_gen, np.eye(4)), os.path.join(save_dir, f"{fname}_gen.nii.gz"))
            except Exception as e:
                print("Could not save nifti:", e)

            print("Saved results & stats for", fname)

    print("\n✅ Quick inference finished. Summary saved to:", summary_path)

if __name__ == "__main__":
    main()
