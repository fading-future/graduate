import torch
from utils.get_root_path import get_root_path

PROJECT_ROOT = get_root_path()

# 定义全局变量配置文件
CONFIG = {
    # 实验信息
    "experiment_name": "exp_01",
    "note": "使用岩心数据集训练",

    # 训练参数
    'batch_size': 1,       # 批次大小 
    'num_workers': 8,      # 数据加载子进程数
    'accumulation_steps': 8,  # 梯度累积步数
    'epochs': 200,          # 训练轮数
    'lr': 0.0002,           # 学习率
    'weight_decay': 0,      # 权重衰减
    'device': "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",  # 训练设备选择

    # 数据与模型参数
    'image_size': 128,        # 输入REV 尺寸
    'base_channels': 64,     # UNet 基础通道数
    'timesteps': 1000,       # 扩散过程时间步数
    'limit_dataset_size': 10000,  # 限制数据集大小，None表示不限制
    'n_resample': 5,           # 每个时间步的重采样次数

    # 输出与日志参数（相对于项目根目录graduation-thesis-code 的）
    'save_model_every': 2,  # 每多少轮保存一次模型
    'num_samples_to_generate': 20,  # 推理时生成样本数量
    
    # 文件路径相关
    'raw_data_dir': f'{PROJECT_ROOT}/data/raw',  # 原始数据集目录
    'processed_data_dir': r'E:\aligned_Training_Data',  # 处理后数据集目录
    'model_output_dir': './models',  # 模型参数保存目录
    'model_checkpoint_path': f'{PROJECT_ROOT}/exp_results/exp_05/models',  # 预训练模型路径，None表示不使用预训练模型
    'log_output_dir': './logs',  # 日志保存目录
    'inference_output_dir': './inference_outputs',  # 推理结果保存目录
}