#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频延迟数据分析脚本
功能：分析各音频对比项的延迟情况，计算总延迟并排序
"""

from audio_files_analysis import Config
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def analyze_delay_data(file_path):
    """
    分析音频延迟数据

    参数:
        file_path: CSV文件路径
    """

    print("=" * 80)
    print("音频延迟数据分析")
    print("=" * 80)

    # 1. 读取数据
    print("\n[1] 读取数据...")
    try:
        df = pd.read_csv(file_path)
        print(f"✓ 成功读取数据！")
        print(f"  - 数据维度：{df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"  - 时间段数量：{df.shape[0]}")
        print(f"  - 对比项数量：{df.shape[1]}")
    except Exception as e:
        print(f"✗ 读取数据失败：{e}")
        return None

    # 2. 数据质量检查
    print("\n[2] 数据质量检查...")
    print(f"  - 缺失值数量：{df.isnull().sum().sum()}")
    print(f"  - 零值比例：{(df == 0).sum().sum() / df.size * 100:.2f}%")

    # 3. 计算各对比项的统计指标
    print("\n[3] 计算延迟统计指标...")

    # 创建结果DataFrame
    results = pd.DataFrame({
        '对比项': df.columns,
        '延迟总和': df.abs().sum()
    })

    # 4. 按总延迟从大到小排序
    print("\n[4] 按总延迟从大到小排序...")
    results_sorted = results.sort_values('延迟总和', ascending=False)

    # 5. 输出详细结果
    print("\n" + "=" * 80)
    print("延迟排序结果（从大到小）")
    print("=" * 80)

    # 设置pandas显示选项
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 100)

    # 格式化输出
    print("\n{:<4} {:<40} {:>15}".format(
        '排名', '对比项', '延迟总和'
    ))
    print("-" * 80)

    for idx, (index, row) in enumerate(results_sorted.iterrows(), 1):
        print("{:<4} {:<40} {:>15.8f}".format(
            idx, row['对比项'][:40], row['延迟总和']
        ))

    # 6. 保存结果到CSV
    output_file = os.path.join(Config().output_dir, 'delay_analysis_results.csv')
    results_sorted.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 详细结果已保存到: {output_file}")

    return results_sorted

# 主程序入口
if __name__ == "__main__":
    # 指定CSV文件路径
    file_path = os.path.join(Config().output_dir, "all_sequences.csv")

    # 运行分析
    results = analyze_delay_data(file_path)

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n输出文件: {file_path} - 详细分析结果")
