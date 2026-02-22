from audio_files_analysis import Config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# 设置图表风格
plt.style.use('default')
plt.rcParams['font.size'] = 10

df = pd.read_csv(os.path.join(Config().output_dir, "delay_analysis_results.csv"))

def format_file_name_sort(files: list) -> list:
    input_files = ['WD_1', 'WD_2', 'WD_3', 'charge_1', 'charge_2', 'charge_3', 'lanlin_1', 'lanlin_2', 'lanlin_3']
    output_files = ['lanlin_1', 'lanlin_2', 'lanlin_3', 'WD_1', 'WD_2', 'WD_3', 'charge_1', 'charge_2', 'charge_3']
    if all([file in input_files for file in files]):
        print('保持和B站视频顺序一致')
        return output_files
    return files

# 解析文件名
def parse_comparison(comparison):
    parts = comparison.split(' vs ')
    return parts[0].replace('.wav', ''), parts[1].replace('.wav', '')

# 提取所有唯一的文件名并排序
all_files = sorted(list(set(
    [f for comp in df['对比项'] for f in parse_comparison(comp)]
)))
all_files = format_file_name_sort(all_files) # 格式化，为了跟B站视频中展示的顺序一致。

# 创建延迟矩阵
delay_matrix = pd.DataFrame(index=all_files, columns=all_files, dtype=float)

for idx, row in df.iterrows():
    file1, file2 = parse_comparison(row['对比项'])
    delay_matrix.loc[file1, file2] = row['延迟总和']
    delay_matrix.loc[file2, file1] = row['延迟总和']

# 创建用于可视化的矩阵（对角线设为NaN）
delay_matrix_plot = delay_matrix.copy()
for file in all_files:
    delay_matrix_plot.loc[file, file] = np.nan

# 计算延迟的最小值和最大值（排除NaN）
delay_min = delay_matrix_plot.min().min()
delay_max = delay_matrix_plot.max().max()
delay_mean = delay_matrix_plot.mean().mean()

# 创建优化后的热力图
fig, ax = plt.subplots(figsize=(12, 9))

# 使用mask屏蔽对角线
mask = np.isnan(delay_matrix_plot)

# 使用更鲜明的colormap和调整范围
sns.heatmap(delay_matrix_plot,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn_r',  # 红黄绿反转，红色表示高延迟
            cbar_kws={'label': 'Delay Sum', 'shrink': 0.8},
            ax=ax,
            annot_kws={'size': 9},
            linewidths=0.5,
            linecolor='white',
            vmin=delay_min,  # 设置最小值
            vmax=delay_max,  # 设置最大值
            mask=mask,  # 屏蔽NaN值
            center=delay_mean,  # 以平均值为中心
            robust=True)  # 使用鲁棒颜色映射

ax.set_title('File Delay Matrix Heatmap (Enhanced Contrast)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('File', fontsize=13)
ax.set_ylabel('File', fontsize=13)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)
plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=11)

plt.tight_layout()
heatmap_file_path = os.path.join(Config().output_dir, "delay_matrix_heatmap_optimized.png")
plt.savefig(heatmap_file_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ 延迟矩阵热力图已保存为 {heatmap_file_path}")
