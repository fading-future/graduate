# 文件位置: stage02_vqvae_code/utils/get_root_path.py
from pathlib import Path

def get_project_root() -> Path:
    """
    返回当前项目(Stage02)的根目录绝对路径
    逻辑: 当前脚本在 utils 文件夹，根目录是 utils 的上一级
    """
    # 1. 获取当前脚本的绝对路径 (e.g., .../stage02_vqvae_code/utils/get_root_path.py)
    current_path = Path(__file__).resolve()
    
    # 2. 取父目录的父目录 (utils -> stage02_vqvae_code)
    root_path = current_path.parent.parent
    
    return root_path

# 测试代码（如果直接运行这个脚本会打印路径）
if __name__ == "__main__":
    print(f"项目根目录是: {get_project_root()}")