import torch
import os

# 1. 检查 PyTorch 是否由 uv 环境提供
print(f"Python 路径: {os.sys.executable}")
print(f"PyTorch 版本: {torch.__version__}")

# 2. 检查你刚刚改名后的数据目录
data_dir = "/chendou_space/data/denoised_unalign_NPYdata"
if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    print(f"✅ 成功！目录已找到，内含 {len(files)} 个文件/文件夹")
else:
    print(f"❌ 错误：找不到路径 {data_dir}，请检查文件夹名是否拼写正确")