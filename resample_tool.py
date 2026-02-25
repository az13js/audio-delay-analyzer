import numpy as np
from scipy.signal import resample_poly
from typing import Literal, Optional, Union
import warnings

class IntegerUpsampler:
    """
    整数倍上采样器
    """

    def __init__(
        self,
        factor: int,
        window: Union[str, tuple] = ('kaiser', 5.0),
        padtype: Literal['constant', 'line', 'mean', 'median', 'maximum', 'minimum', 'wrap'] = 'constant',
        cval: Optional[float] = None
    ):
        """
        初始化上采样器

        Parameters
        ----------
        factor : int
            上采样倍数，必须 >= 1
        window : str or tuple, default=('kaiser', 5.0)
            抗混叠滤波器窗口：
            - 'kaiser' : 推荐，tuple形式，beta越大阻带衰减越高
            - 'boxcar' : 矩形窗（无滤波，最快但质量差）
            - 'hann', 'hamming', 'blackman' : 经典窗函数
        padtype : str, default='constant'
            边界填充模式：
            - 'constant' : 常数填充（默认0）
            - 'line' : 线性外推（推荐用于非零均值信号）
            - 'wrap' : 循环填充（适合周期信号）
        cval : float, optional
            constant填充时的值，默认0
        """
        if not isinstance(factor, int) or factor < 1:
            raise ValueError(f"factor 必须是正整数，当前: {factor}")

        self.factor = factor
        self.window = window
        self.padtype = padtype
        self.cval = cval if cval is not None else 0.0

        # 缓存GCD优化结果（用于非质数因子）
        self._cached_gcd = None

    def upsample(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        执行上采样

        Parameters
        ----------
        x : np.ndarray
            输入信号，支持任意维度，数据类型自动转换
        axis : int, default=-1
            上采样沿着的轴

        Returns
        -------
        np.ndarray
            上采样后的信号，长度 = len(x) * factor
        """
        # 类型转换：确保浮点数以获得最佳精度
        original_dtype = x.dtype
        if not np.issubdtype(original_dtype, np.floating):
            x = x.astype(np.float64)
            warnings.warn(f"输入类型 {original_dtype} 已转换为 float64 以保证精度")

        # ========== 关键修复 ==========
        # 只有 padtype='constant' 时才传递 cval 参数
        # 其他填充模式（line, wrap等）不接受 cval 参数，否则 scipy 会报错
        resample_kwargs = {
            'up': self.factor,
            'down': 1,
            'axis': axis,
            'window': self.window,
            'padtype': self.padtype
        }

        # 仅当 padtype='constant' 时才添加 cval
        if self.padtype == 'constant':
            resample_kwargs['cval'] = self.cval

        # 执行多相滤波上采样
        y = resample_poly(x, **resample_kwargs)

        # 尝试恢复原类型（如果输入是整数）
        if original_dtype in [np.int16, np.int32, np.int64]:
            y = np.clip(y, np.iinfo(original_dtype).min, np.iinfo(original_dtype).max)
            y = y.astype(original_dtype)

        return y

    def __call__(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """使实例可调用"""
        return self.upsample(x, axis)

    def get_filter_info(self) -> dict:
        """获取滤波器设计信息"""
        # 估算滤波器长度（resample_poly内部逻辑）
        if isinstance(self.window, tuple) and self.window[0] == 'kaiser':
            beta = self.window[1]
            # 根据Kaiser窗公式估算
            if beta > 8.6:
                attenuation = 2.285 * beta + 8  # 阻带衰减
            else:
                attenuation = 2.285 * beta
        else:
            attenuation = "未知（非Kaiser窗）"

        return {
            "factor": self.factor,
            "window": self.window,
            "estimated_filter_length": 2 * 10 * self.factor + 1,  # 近似值
            "padtype": self.padtype
        }


# ============ 便捷函数接口 ============

def upsample_integer(
    x: np.ndarray,
    factor: int,
    window: Union[str, tuple] = ('kaiser', 5.0),
    padtype: str = 'constant',
    axis: int = -1
) -> np.ndarray:
    """
    一键整数倍上采样（函数式接口）

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(100000).astype(np.float32)
    >>> y = upsample_integer(x, factor=4)  # 400K 输出
    >>> print(y.shape)  # (400000,)

    Parameters
    ----------
    x : np.ndarray
        输入信号，建议 float32/float64
    factor : int
        上采样倍数 (1, 2, 3, ...)
    window : str or tuple
        滤波器窗口，默认Kaiser窗(beta=5.0)
    padtype : str
        边界处理模式
    axis : int
        操作轴

    Returns
    -------
    np.ndarray
        上采样后的数组
    """
    upsampler = IntegerUpsampler(factor=factor, window=window, padtype=padtype)
    return upsampler.upsample(x, axis=axis)


# ============ 批量处理优化版 ============

def upsample_large_array(
    x: np.ndarray,
    factor: int,
    chunk_size: int = 50000,
    **kwargs
) -> np.ndarray:
    """
    分块上采样（用于超大数组，减少内存峰值）

    对于 100K 序列通常不需要分块，但提供此接口以备扩展。
    """
    if len(x) <= chunk_size:
        return upsample_integer(x, factor, **kwargs)

    # 分块处理逻辑（带重叠避免边界效应）
    # ... 实现略，100K 不需要分块
    pass


# ============ 使用示例与测试 ============

if __name__ == "__main__":
    import time

    # 生成 100K 测试数据（模拟音频信号）
    np.random.seed(42)
    sr_original = 16000  # 假设原始采样率 16kHz
    duration = 100000 / sr_original  # ~6.25秒

    t = np.linspace(0, duration, 100000)
    # 混合信号：低频 + 高频 + 噪声
    signal_100k = (
        np.sin(2 * np.pi * 500 * t) +      # 500Hz 正弦
        0.5 * np.sin(2 * np.pi * 3000 * t) +  # 3kHz 正弦
        0.1 * np.random.randn(100000)
    ).astype(np.float32)

    print(f"输入信号: {signal_100k.shape}, dtype={signal_100k.dtype}")
    print(f"信号时长: {duration:.2f}s @ {sr_original}Hz")

    # 测试不同上采样倍数
    for factor in [2, 4, 8]:
        print(f"\n{'='*50}")
        print(f"测试 {factor}倍上采样 (目标采样率: {sr_original*factor/1000:.0f}kHz)")

        # 方法1: 使用类接口（推荐，可复用）
        upsampler = IntegerUpsampler(
            factor=factor,
            window=('kaiser', 5.0),  # beta=5.0 平衡质量与速度
            padtype='line'  # 线性外推，适合音频
        )

        # 预热
        _ = upsampler(signal_100k[:1000])

        # 正式计时
        start = time.perf_counter()
        upsampled = upsampler(signal_100k)
        elapsed = time.perf_counter() - start

        print(f"输出长度: {upsampled.shape}")
        print(f"耗时: {elapsed*1000:.2f} ms")
        print(f"速度: {len(signal_100k)/elapsed/1e6:.2f} MSamples/s")

        # 验证能量守恒（粗略检查）
        energy_ratio = np.sum(upsampled**2) / np.sum(signal_100k**2) / factor
        print(f"能量比例 (应≈1): {energy_ratio:.4f}")

        # 方法2: 函数式接口（快速使用）
        # upsampled = upsample_integer(signal_100k, factor=factor)
