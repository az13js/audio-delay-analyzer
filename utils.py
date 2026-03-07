'''
共用的函数
'''
import numpy as np
from typing import Optional
from resample_tool import IntegerUpsampler
from scipy.signal import correlate

def db_to_amplitude(db_val):
    """
    将分贝转换为线性幅度值。
    公式：A = 10^(dB/20)
    """
    return 10 ** (db_val / 20.0)

def calculate_delay_correlation(ref: np.ndarray, y: np.ndarray, window_size: int, ref_sr: int, upsampler: Optional[IntegerUpsampler]=None) -> float:
    """
    计算两个音频文件的延迟时间。

    Args:
        ref: 参考音频
        y: 待比较音频
        window_size: 窗口大小，单位：采样点数量
        ref_sr: 参考音频的采样率
        upsampler: 上采样器，可选，默认为None
    Returns:
        delay_time_ms: 延迟时间，单位：毫秒
    """
    audio_ref = upsampler(ref) if upsampler is not None else ref
    audio_y = upsampler(y) if upsampler is not None else y
    real_window_size = window_size * upsampler.factor if upsampler is not None else window_size
    real_ref_sr = ref_sr * upsampler.factor if upsampler is not None else ref_sr

    # 接下来需要找出audio_ref在audio_y中出现的位置
    corr = correlate(audio_y, audio_ref, mode='valid')
    estimated_offset = np.argmax(corr)
    offset = estimated_offset - real_window_size
    offset = -offset # 正数表示audio_y延迟，负数表示audio_y提前
    delay_time_ms = 1000 * offset / real_ref_sr
    return delay_time_ms
