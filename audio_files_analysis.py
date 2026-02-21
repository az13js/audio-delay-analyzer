import os
import soundfile as sf
import pandas as pd
from typing import List, Optional
import numpy as np
from scipy.signal import correlate
from utils import db_to_amplitude

class Config:
    def __init__(self):
        # 数据文件路径
        self.data_path: str = "./data"
        # 采样率
        self.sample_rate: int = 48000
        # 计算使用的声道序号 (索引从0开始)
        self.channel: int = 0
        # 输出数据存放的文件夹
        self.output_dir: str = "./output"
        # 数据对齐时，参考音频文件前后填充的空白采样点数量
        self.align_padding_samples: int = int(self.sample_rate * 0.5)

class AudioAnalysis:
    def __init__(self, config: Config):
        self.config = config
        self.audio_files = self.scan_and_validate_directory(config)
        for file in self.audio_files:
            print(f"音频文件: {file}")

        self.make_output_directory(self.config.output_dir)
        # 对音量进行归一化
        self.normalized_audio_files = self.normalize_audio_volumes(self.audio_files, self.config.output_dir)
        for file in self.normalized_audio_files:
            print(f"音量归一化后的音频文件: {file}")

        self.aligned_files = self.align_audios(self.normalized_audio_files, [os.path.basename(file) for file in self.audio_files])
        for file in self.aligned_files:
            print(f"整体对齐后的音频文件: {file}")

    def delay_correlation(self, idx_ref: int, idx: int, savefig: Optional[str]=None, plot: bool=False, window_size: Optional[int]=None, hard_clipped_delay_ms: Optional[int]=None) -> List[float]:
        """
        利用窗口，比较两个音频文件之间的延迟，并绘制图谱。

        Args:
            idx_ref: 参考音频文件的索引
            idx: 待比较音频文件的索引
            savefig: 保存图片的文件名
            plot: 是否绘制图谱
        Returns:
            delay_samples: 延迟时间列表，单位：秒
        """
        if savefig is None and not plot:
            return
        # 获取音频文件（经过上述处理音频文件采样点数量将一致，音频文件将是单通道）
        ref_file = self.aligned_files[idx_ref]
        file = self.aligned_files[idx]

        ref_y, ref_sr = sf.read(ref_file)
        y, sr = sf.read(file)

        delay_samples = []
        if window_size is None:
            window_size = int(self.config.sample_rate / 2)
        if hard_clipped_delay_ms is None:
            hard_clipped_delay_ms = 1000 * 2 * window_size / self.config.sample_rate

        # 开头和结尾的空间预留出来
        step_count = int(len(ref_y) / window_size) - 2
        sample_begin = window_size
        sample_length = window_size * step_count

        for i in range(step_count):
            ref_y_begin = sample_begin + i * window_size
            audio_ref = ref_y[ref_y_begin: ref_y_begin + window_size]
            audio_y = y[ref_y_begin - window_size: ref_y_begin + window_size + window_size]

            # 居中处理
            audio_ref = audio_ref - np.mean(audio_ref)
            audio_y = audio_y - np.mean(audio_y)

            audio_ref_rms = np.sqrt(np.mean(audio_ref ** 2))
            audio_y_rms = np.sqrt(np.mean(audio_y ** 2))
            if audio_ref_rms < db_to_amplitude(-60) or audio_y_rms < db_to_amplitude(-60): # 音量过小，忽略
                delay_samples.append(0.0)
                continue
            alpha = audio_ref_rms / audio_y_rms
            audio_y = audio_y * alpha

            # 接下来需要找出audio_ref在audio_y中出现的位置
            corr = correlate(audio_y, audio_ref, mode='valid')
            estimated_offset = np.argmax(corr)
            offset = estimated_offset - window_size
            offset = -offset # 正数表示audio_y延迟，负数表示audio_y提前
            delay_time_sec = offset / ref_sr
            delay_samples.append(delay_time_sec)

        # 绘制图谱所需的数据，背景是参考音频区间的FFT图谱，前景是延迟时间，单位是毫秒
        ref_y_samples = ref_y[sample_begin: sample_begin + sample_length]
        ref_y_times = ref_y_samples / ref_sr

        # --- 绘图逻辑 ---
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        # 1. 绘制背景：参考音频的 FFT 图谱
        # 为specgram指定明确的时间轴，确保与延迟线对齐
        time_axis_specgram = np.linspace(sample_begin / ref_sr,
                                    (sample_begin + sample_length) / ref_sr,
                                    len(ref_y_samples))
        plt.specgram(ref_y_samples, Fs=ref_sr, NFFT=1024,
                    cmap='Greys', aspect='auto',
                    xextent=(time_axis_specgram[0], time_axis_specgram[-1]))
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Intensity (dB)')

        # 2. 绘制前景：延迟时间曲线 (单位 ms)
        ax2 = plt.gca().twinx()

        # 构建延迟曲线的时间轴 - 保持原有逻辑
        time_axis = (np.arange(step_count) * window_size + sample_begin + window_size / 2) / ref_sr

        # 将延迟秒数转换为毫秒
        delay_ms = np.array(delay_samples) * 1000
        #delay_ms = np.clip(delay_ms, -hard_clipped_delay_ms, hard_clipped_delay_ms)
        delay_ms[np.abs(delay_ms) > hard_clipped_delay_ms] = None

        # 绘制曲线，颜色设为醒目的红色
        ax2.plot(time_axis, delay_ms, color='red', linewidth=2, marker='o', markersize=3, label='Delay (ms)')
        ax2.set_ylabel('Delay (ms)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # 设置标题和坐标轴
        plt.title(f'Delay Correlation: Reference {os.path.basename(ref_file)} vs File {os.path.basename(file)}')
        plt.xlabel('Time (s)')

        # 根据参数保存或显示
        if savefig:
            plt.savefig(savefig)
            print(f"Figure saved to {savefig}")

        if plot:
            plt.show()
        else:
            plt.close()
        return delay_samples

    @staticmethod
    def validate_single_wav(filepath: str, config: Config) -> None:
        """
        校验单个WAV文件是否满足配置要求。

        Args:
            filepath: WAV文件路径
            config: 配置对象
        """
        info = sf.info(filepath)

        # 1. 校验采样率
        if info.samplerate != config.sample_rate:
            raise ValueError(f"采样率错误: 文件{filepath}为 {info.samplerate}Hz, 要求 {config.sample_rate}Hz")

        # 2. 校验声道
        # config.channel 是索引，例如索引为0，则至少需要1个声道
        # 如果 config.channel = 1 (第二声道)，文件至少需要2个声道
        if info.channels <= config.channel:
            raise ValueError(f"声道数不足: 文件{filepath}包含 {info.channels} 个声道, 无法获取索引为 {config.channel} 的声道")

    def scan_and_validate_directory(self, config: Config) -> List[str]:
        """
        遍历目录及子目录，校验所有WAV文件。

        Args:
            config: 配置对象
        Returns:
            List[str]: 所有WAV文件的路径列表
        """
        data_path = config.data_path

        if not os.path.exists(data_path):
            raise ValueError(f"数据目录不存在: {data_path}")

        count = 0
        results: List[str] = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    full_path = os.path.join(root, file)
                    print(f"正在校验文件: {full_path}")
                    self.validate_single_wav(full_path, config)
                    count += 1
                    results.append(full_path)

        print(f"扫描完成，共发现 {count} 个 WAV 文件。")
        if count < 2:
            raise ValueError("WAV文件过少，需要大于或等于2个文件。")
        return results

    @staticmethod
    def make_output_directory(output_dir: str) -> None:
        """
        生成输出目录。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def normalize_audio_volumes(self, audio_files: List[str], output_dir: str) -> List[str]:
        """
        对音频文件进行音量归一化。
        Args:
            audio_files: 音频文件列表
            output_dir: 输出目录
        Returns:
            List[str]: 音量归一化后的音频文件列表
        """
        # 新建输出目录
        normalized_dir = os.path.join(output_dir, 'normalized_audio_volumes')
        self.make_output_directory(normalized_dir)
        # 如所有文件存在那么直接返回
        results = []
        not_exists = False
        for i, file in enumerate(audio_files):
            result_file = os.path.join(normalized_dir, self.md5(file) + '-' + os.path.basename(file))
            if not os.path.exists(result_file):
                not_exists = True
            results.append(result_file)
        if not not_exists:
            print(f"所有文件已存在，无需再次归一化处理。")
            return results

        # 对所有音频文件计算音量的RMS数值（只取指定声道）
        rms_values = [self.rms_value(f, self.config.channel) for f in audio_files]
        # 获取最大音量
        max_rms = max(rms_values)

        # 对音量进行放大，统一到 max_rms
        for i, file in enumerate(audio_files):
            result_file = os.path.join(normalized_dir, self.md5(file) + '-' + os.path.basename(file))
            if not os.path.exists(result_file):
                y, sr = sf.read(file)
                y = y if 1 == y.ndim else y[:, self.config.channel]

                y_norm = y * (max_rms / rms_values[i])
                sf.write(result_file, y_norm, sr, subtype=sf.info(file).subtype)

        return results

    @staticmethod
    def md5(filepath: str) -> str:
        """
        计算文件的MD5值。

        Args:
            filepath: 文件路径
        Returns:
            str: 文件的MD5值
        """
        import hashlib
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def rms_value(filepath: str, channel: int) -> float:
        """
        计算单个音频文件的音量RMS数值。

        Args:
            filepath: 音频文件路径
            channel: 声道索引
        Returns:
            float: 音量RMS数值
        """
        y, sr = sf.read(filepath)
        y = y if 1 == y.ndim else y[:, channel]
        rms = np.sqrt(np.mean(y**2))
        return rms

    def align_audios(self, audio_files: List[str], rename_filename: List[str]) -> None:
        """
        对齐所有音频文件。
        Args:
            audio_files: 音频文件列表
            rename_filename: 音频文件名称列表
        """
        self.make_output_directory(os.path.join(self.config.output_dir, 'aligned_audio'))
        print(f"正在对齐音频文件...")

        results = []
        reference_file = audio_files[0]
        for i, file in enumerate(audio_files):
            output_file = os.path.join(self.config.output_dir, 'aligned_audio', rename_filename[i])
            print(f"\n正在对齐音频: {file}\n输出文件: {output_file}")

            if not os.path.exists(output_file):
                self.align_audio(file, reference_file, output_file, self.config.align_padding_samples)
            else:
                print('已存在，跳过')
            results.append(output_file)
        return results

    @staticmethod
    def align_audio(filepath: str, reference_file: str, output_file: str, padding_samples: int) -> None:
        """
        对齐单个音频文件。（单声道）

        Args:
            filepath: 音频文件路径
            reference_file: 参考音频文件路径
            output_file: 输出文件路径
            padding_samples: 数据对齐时，参考音频文件前后填充的空白采样点数量
        """
        print(f"参考音频: {reference_file} 正在对齐音频: {filepath}")
        y, sr = sf.read(filepath)
        y_ref, sr_ref = sf.read(reference_file)
        if y.ndim != 1:
            raise ValueError(f"音频文件 {filepath} 不是单声道")
        if y_ref.ndim != 1:
            raise ValueError(f"音频文件 {reference_file} 不是单声道")
        # 检查采样率
        if sr != sr_ref:
            raise ValueError(f"音频文件 {filepath} 和 {reference_file} 的采样率不一致")

        # 在参考文件的前后添加空白采样点
        y_ref = np.pad(y_ref, (padding_samples, padding_samples), mode='constant')
        # 计算互相关
        corr = correlate(y_ref, y, mode='full')
        # 找到最大值的位置
        max_corr_index = np.argmax(corr)
        delay_samples = max_corr_index - (len(y) - 1) # 这里算出 y 需要往右移动的采样点数
        if delay_samples > 0:
            # 需要在 y 的左侧添加空采样点
            y = np.pad(y, (delay_samples, 0), mode='constant')
        if delay_samples < 0:
            # 需要截断y前面的采样点
            y = y[abs(delay_samples):]
        # 最后对y进行处理，使得y与y_ref长度一致
        if len(y) > len(y_ref):
            y = y[:len(y_ref)]
        elif len(y) < len(y_ref):
            # 需要在y的右侧添加空采样点
            y = np.pad(y, (0, len(y_ref) - len(y)), mode='constant')
        sf.write(output_file, y, sr, subtype=sf.info(reference_file).subtype)

if __name__ == "__main__":
    config = Config()
    analysis = AudioAnalysis(config)
    m = analysis.audio_files
    os.makedirs("output/figure", exist_ok=True)
    results = {}
    for i in range(0, len(m) - 1):
        for j in range(i+1, len(m)):
            file_name = f"{i:02d}_{j:02d}_" + os.path.basename(m[i]) + '_vs_' + os.path.basename(m[j]) + ".png"
            data:List[float] = analysis.delay_correlation(i, j, savefig=f"output/figure/{file_name}", window_size=config.sample_rate, hard_clipped_delay_ms=5)
            title = f"{os.path.basename(m[i])} vs {os.path.basename(m[j])}"
            results[title] = data # 延迟时间，秒

    df = pd.DataFrame({title: pd.Series(data) for title, data in results.items()})
    df.to_csv("output/all_sequences.csv", index=False)
