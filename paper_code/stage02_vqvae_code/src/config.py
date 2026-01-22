import torch

# 定义全局变量配置文件
CONFIG = {
    # 实验信息
    "experiment_name": "exp02_256_64_2048",
    "note": "Stage 1: VQ-VAE 训练。输入256x256x256，使用A100 40GB 4卡训练",

    # 训练参数
    'batch_size': 4,        # A100 80G 跑 256^3 的 VQVAE，Batch=4 应该很轻松，可以尝试 6 或 8
    'num_workers': 2,       # Windows下如果报错，请改为 0
    'epochs': 300,          
    'lr': 1e-4,             # VQ-VAE 学习率不宜过大
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    # Config 中增加
    'gradient_accumulation_steps': 4, # 2 * 4卡 * 4累积 = 等效 Batch 32

    # 数据与模型参数
    'image_size': 256,      # 输入体素尺寸
    'embedding_dim': 64,    # 潜在特征维度
    'num_embeddings': 2048, # 码本大小
    

    # 指向包含所有子文件夹的根目录 E:\chendou\rock_core_data
    'processed_data_dir': r'/chendou_space/data/aligned_Training_Data_Interactive',
    
    # 输出目录
    'model_output_dir': './models',
    'log_output_dir': './logs',
}