import torch
from utils.get_root_path import get_root_path

PROJECT_ROOT = get_root_path()

# 定义全局变量配置文件
CONFIG = {
    # 实验信息
    "experiment_name": "stage1_vqvae_256",
    "note": "Stage 1: VQ-VAE 训练。输入256x256x256，16-bit归一化到[-1,1]，使用A100",

    # 训练参数
    'batch_size': 4,        # A100 80G 跑 256^3 的 VQVAE，Batch=4 应该很轻松，可以尝试 6 或 8
    'num_workers': 8,       # Windows下如果报错，请改为 0
    'epochs': 200,          
    'lr': 1e-4,             # VQ-VAE 学习率不宜过大
    'device': "cuda" if torch.cuda.is_available() else "cpu",

    # 数据与模型参数
    'image_size': 256,      # 【关键】直接训练 256
    'embedding_dim': 64,    # 潜在特征维度
    'num_embeddings': 2048, # 码本大小
    
    # 【关键】修改为你实际的数据路径
    # 注意：Windows 路径最好前面加 r，或者用双反斜杠
    # 'processed_data_dir': r"C:\\Users\\vipuser\\Desktop\\chendou\\vqvae_code\\data\\Final_Dataset_NPY_9",
    # 指向包含所有子文件夹的根目录 E:\chendou\rock_core_data
    'processed_data_dir': r'E:\\chendou\\rock_core_data\\Final_Dataset_NPY_9',
    
    # 输出目录
    'model_output_dir': './stage1_models',
    'log_output_dir': './stage1_logs',
}