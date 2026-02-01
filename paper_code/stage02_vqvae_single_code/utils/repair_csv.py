import pandas as pd

# 1. 读取 CSV 文件
df = pd.read_csv(r'E:\chendou\paper_code\stage02_vqvae_single_code\stage1_vqvae_256\stage1_logs\training_log_finetune.csv')

# 2. 修正 Step 逻辑
# 我们遍历数据，如果发现当前的 step 比上一个 step 小，说明发生了断点重置
# 我们记录这个 offset (偏移量)

offset = 0
corrected_steps = []
prev_step = -1

for index, row in df.iterrows():
    current_step = row['Step']
    
    # 如果不是第一行，且当前步数突然变小（比如从 2069 变到了 1）
    if prev_step != -1 and current_step < prev_step:
        # 此时的 offset 应该是上一个正常的 step 值
        # 比如上一行是 2069，当前是 1，我们希望它是 2070
        # 所以 offset 累加量 = 上一步的值 (2069)
        offset += prev_step
        
    # 如果是断点后的数据，由于它重置为 1, 2, 3...
    # 但有时候可能断点恢复的逻辑更复杂，这里我们假设它是一个全新的计数
    # 实际上更稳健的方法是检测到下降后，用 offset 修正
    
    # 简单的修正逻辑：
    # 如果发生了重置，现在的真实步数 = 当前记录步数 + offset
    # 注意：这个简单的逻辑假设重置后的计数是连续的 (1, 2, 3...)
    
    # 为了处理多次断点，我们使用累积偏移量的逻辑：
    # 当检测到 current_step < prev_step (未修正前) 时，更新 offset
    pass 

# 下面是一个更简洁向量化的处理方式：
# 找出 Step 并不是递增的地方
steps = df['Step'].values
diffs = steps[1:] - steps[:-1]
# 找到断点位置（差值为负数的地方）
reset_indices = [i+1 for i, x in enumerate(diffs) if x < 0]

cumulative_offset = 0
fixed_steps = steps.copy()

if len(reset_indices) > 0:
    # 只需要处理第一个断点（如果你的日志只有这一次断点）
    # 或者循环处理多次断点
    last_valid_step = steps[reset_indices[0]-1]
    
    # 从断点行开始，所有的步数都要加上 last_valid_step
    # 注意：这里假设断点后的步数是从 1, 2, 3 开始记录的
    df.loc[reset_indices[0]:, 'Step'] += last_valid_step

# 3. 保存修复后的文件
df.to_csv('fixed_training_log.csv', index=False)
print("修复完成！前 10 行预览：")
print(df.head(10))
print("\n断点处预览：")
print(df.iloc[5:10])