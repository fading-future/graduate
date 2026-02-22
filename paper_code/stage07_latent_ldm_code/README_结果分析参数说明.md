# Stage07 结果分析参数说明（单样本 + 批量）

本文档用于解释以下目录中的评估结果字段，并按类别归档：

- `eval_batch_e40_ema_fix_s666666`
- `exp_results/stage07_patch_ldm_v3/eval/eval_ema_12_6-6-22_Global_Consistency_z3008_y128_x384`

对应脚本：

- 单样本评估：`src/evaluate_generated_sample.py`
- 批量评估：`src/evaluate_batch_generated.py`
- 批量可视化：`src/visualize_batch_eval.py`

---

## 1. 输出文件与用途

### 1.1 单样本目录（`eval_ema_xxx`）

常见文件：

- `metrics.json`：该样本的全部数值指标
- `pred_voxel_prob.npy/.png`：解码后体素概率
- `pred_voxel_bin.npy/.png`：阈值后二值体
- `voxel_bin_pred_vs_gt.png`：Pred/GT 三视图对比
- `phi_from_bin_pred_vs_gt.png`：Phi map（三维网格）对比
- `latent_ch0_pred_vs_gt.png`：latent 通道切片对比

### 1.2 批量目录（`eval_batch_xxx`）

常见文件：

- `per_sample_metrics.csv/jsonl`：每个样本一行
- `summary.json`：批量聚合统计（`_mean/_std/_min/_max`）
- `phi_cell_metrics.csv/jsonl`：每个样本、每个 Phi 单元格的一行
- `phi_cell_summary.csv`：同一单元格跨样本聚合
- `figures/*.png`：科研风格汇总图

---

## 2. 语义约定（必须先看）

你当前命令常用设置：

- `--pore-value 0`：把值 `0` 视作“目标相（孔隙相）”
- `--gt-phi-semantic rock_rate`：GT 的 phi_map 数值语义是“岩石率”

因此会出现两套“看起来相反”的字段：

- `porosity_*`：默认按“值为 1 的体积分数”计算（常对应岩石率）
- `pore_porosity_*` 或 `target_phase_fraction_*`：按 `pore-value` 指定的目标相计算（你这里更应关注这组）

结论：你的项目里建议优先看 `pore_*` / `phase_*` / `target_phase_fraction_*`。

---

## 3. 指标分组总览（怎么看好坏）

### 3.1 Latent 重建误差（越低越好）

- `latent_mae`, `latent_mse`, `latent_rmse`

含义：生成 latent 与 GT latent 的差距。  
用途：看扩散模型在 latent 空间的拟合程度。

### 3.2 Voxel 二值重叠指标（Dice/IoU 越高越好）

- `voxel_dice`, `voxel_iou`, `voxel_precision`, `voxel_recall`

含义：

- `precision` 高：预测为正的体素更准
- `recall` 高：GT 正类被覆盖得更全
- `dice/iou`：总体重叠质量

### 3.3 全局体积分数（孔隙率/岩石率）误差

- 默认语义：`porosity_pred`, `porosity_gt`, `porosity_abs_err`
- 目标相语义：`target_phase_fraction_pred(_aligned)`, `target_phase_fraction_gt_aligned`, `target_phase_fraction_abs_err`
- 别名（`pore-value=0` 时）：`pore_porosity_pred(_aligned)`, `pore_porosity_gt_aligned`, `pore_porosity_abs_err`

说明：`_aligned` 表示与 GT 做了居中裁剪对齐后再算。

### 3.4 Phi map（网格级）误差

- 基础：`bin_phi_mae/mse/rmse/corr`, `prob_phi_*`
- 目标相版本：`phase_bin_phi_*`, `phase_prob_phi_*`
- 孔隙别名（`pore-value=0`）：`pore_phi_mae/rmse/corr`

解释：

- `bin`：来自二值体统计
- `prob`：来自概率体统计
- `corr` 高说明空间分布趋势一致
- `mae/rmse` 低说明数值接近

### 3.5 边界与层面偏差诊断（判断是否有“头尾偏差”）

- `phi_layer0_bias`, `phi_layer_last_bias`
- `phi_layer0_mae`, `phi_layer_last_mae`
- `z_head_phase_gap`, `z_tail_phase_gap`
- `z_head_porosity_gap`, `z_tail_porosity_gap`
- 以及对应 `..._pred`, `..._gt`

解释：

- `bias/gap > 0`：预测值高于 GT
- `bias/gap < 0`：预测值低于 GT
- 常用于定位“固定侧面黑块/白块”系统偏差

### 3.6 两点相关函数 TP2（结构统计一致性）

- `tp2_corr`, `tp2_mae`, `tp2_rmse`, `tp2_mse`, `tp2_bias_mean`
- 分轴：`tp2_x_*`, `tp2_y_*`, `tp2_z_*`

解释：

- `tp2_corr` 高：空间相关结构形状接近
- `tp2_bias_mean`：曲线整体偏高/偏低

### 3.7 物理模拟（OpenPNM，可选）

仅在开启 `--physics-abs-k` 后出现：

- `kabs_pred_x/y/z`, `kabs_gt_x/y/z`
- `kabs_abs_err_x/y/z`, `kabs_rel_err_x/y/z`
- `kabs_pred_mean`, `kabs_gt_mean`, `kabs_abs_err_mean`, `kabs_rel_err_mean`
- `kabs_pred_x_ok`, `kabs_pred_x_err`（是否成功和失败原因）

### 3.8 运行与配置记录

- `time_sec`
- `seed`, `seed_mode`, `sample_seed`
- `ckpt`, `ddim_steps`, `threshold`
- `infer_order`, `infer_direction`, `infer_random_order`, `infer_random_direction`
- `pore_value`, `gt_phi_semantic`, `tp2_max_lag`, `tp2_phase`

---

## 4. 批量 summary.json 的命名规则

`summary.json` 会把每个数值字段自动聚合成：

- `<key>_mean`
- `<key>_std`
- `<key>_min`
- `<key>_max`

例如：

- `voxel_dice_mean`：全批次平均 Dice
- `bin_phi_corr_std`：全批次 phi 相关系数的离散程度

另外还会有批次级元信息：

- `num_samples`, `ckpt`, `phi_dir`, `num_selected` 等。

---

## 5. Phi 单元格文件（`phi_cell_*`）说明

### 5.1 `phi_cell_metrics.csv`

每一行 = 某个样本的某个网格单元（`cell_i, cell_j, cell_k`）：

- `gt_phi`
- `pred_phi_bin`, `pred_phi_prob`
- `bin_abs_err`, `bin_bias`
- `prob_abs_err`, `prob_bias`

### 5.2 `phi_cell_summary.csv`

每一行 = 固定网格位置跨样本聚合：

- `gt_phi_mean`
- `pred_phi_bin_mean`, `pred_phi_prob_mean`
- `bin_mae_mean`, `bin_bias_mean`
- `prob_mae_mean`, `prob_bias_mean`

用途：定位“总是出问题的网格位置”。

---

## 6. 图像与参数的对应关系

- `voxel_bin_pred_vs_gt.png` 对应 `voxel_*`、`porosity_*`
- `phi_from_bin_pred_vs_gt.png` 对应 `bin_phi_*`
- `phi_from_prob_pred_vs_gt.png` 对应 `prob_phi_*`
- `figures/phi_cell_layers.png` 对应 `phi_cell_summary.csv`
- `figures/overview_panel.png` 汇总 `voxel_* + phi_* + tp2_* + time_sec`

---

## 7. 你当前两类结果目录的解读建议

### 7.1 单样本目录（`eval_ema_12_...`）

优先看：

- `voxel_dice/iou`：样本级重叠质量
- `porosity_abs_err` 与 `bin_phi_mae`：全局量值偏差 + 网格偏差
- `latent_mae`：latent 空间误差是否过大

### 7.2 批量目录（`eval_batch_e40_ema_fix_s666666`）

优先看：

- 稳定性：`*_std` 是否过大
- 系统偏差：`phi_cell_bin_bias_mean`、`z_head/tail_*_gap_mean` 的符号和量级
- 结构一致性：`tp2_corr_mean` 高不代表局部数值就一定好，仍需看 `bin_phi_mae_mean`

---

## 8. 常见误解

1. `tp2_corr` 很高就说明样本完全正确  
不是。它更偏向“统计结构”一致，不保证局部网格数值完全对齐。

2. `porosity_*` 和“孔隙率”一定同义  
不一定，取决于二值语义。你的流程建议以 `pore_*` / `target_phase_fraction_*` 为准。

3. 单样本好看就代表模型稳定  
应以批量 `mean + std + min/max` 判断。

---

## 9. 建议的汇报最小指标集（论文/汇报）

建议至少报告以下 10 个：

- `voxel_dice_mean/std`
- `voxel_iou_mean/std`
- `pore_porosity_abs_err_mean/std`
- `phase_bin_phi_mae_mean/std`
- `phase_bin_phi_corr_mean/std`
- `tp2_corr_mean/std`
- `tp2_mae_mean/std`
- `time_sec_mean`
- `phi_cell_bin_bias_mean`
- `z_head_phase_gap_mean`, `z_tail_phase_gap_mean`

以上可同时覆盖：体素重叠、全局量值、网格分布、结构统计、边界偏差、推理速度。

