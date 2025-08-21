import pandas as pd

# 1. 读取 Excel 文件（假设文件名为 data.xlsx）
df = pd.read_excel("lightop_gemm_smooth.xlsx")  # 如果没有表头，加参数 header=None

# 按 m、n、k 分组，并保留原始索引
grouped = df.groupby(["m", "n", "k"])
min_indices = grouped["lightop_gemm_smooth"].idxmin()  # 获取最小值的原始索引

# 按原始索引排序
result = df.loc[min_indices].sort_index()

# 重置索引（可选）
result = result.reset_index(drop=True)

# 打印结果
for index, row in result.iterrows():
    # 组合 m,n,k 为 M32N6144K4096 格式
    mnk_str = f"\"M{int(row['m'])}N{int(row['n'])}K{int(row['k'])}\": {int(row['tunning_slide_block'])},"
    
    # 打印组合字符串和整行数据
    print(f"{mnk_str}")
    # print(row.to_string())  # 以字符串形式打印整行数据

# 5. 保存结果到新 Excel 文件（可选）
# result.to_excel("min_time_results.xlsx", index=False)
