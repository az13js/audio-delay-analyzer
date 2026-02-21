'''脚本用于模拟真实录制环境下录得的测试信号录音

提示词：
实现下述数据生成的任务：
编写一个Python脚本生成采样率为48KHz持续时长10秒的单声道音频文件，文件中包含5秒的从20Hz到20KHz的扫频信号。具体要求如下：
1. 扫频信号的起始时间非固定，在音频文件的1s到4s之间随机（目的是使得采样点不能在后期严格对齐）
2. 每一个音频的扫频信号强度在-3dB到-5dB之间随机，使得文件与文件之间音量存在差异
3. 每个音频文件都独立添加强度为-60dB的随机噪音

特殊要求：
你不需要使用你的虚拟沙箱，直接给出代码内容即可。
你的代码需要在一个文件内容纳，不需要拆分多个文件。
你的代码逻辑需要适当拆分成函数或者类，避免逻辑过于复杂。

以下是你可以使用的软件包，Python版本3.9：

已经安装好的：
python -m pip install pandas matplotlib librosa seaborn
执行命令打印的内容：
```
$ python -m pip freeze
audioread==3.1.0
certifi==2026.1.4
cffi==2.0.0
charset-normalizer==3.4.4
contourpy==1.3.3
cycler==0.12.1
decorator==5.2.1
fonttools==4.61.1
idna==3.11
joblib==1.5.3
kiwisolver==1.4.9
lazy_loader==0.4
librosa==0.11.0
llvmlite==0.46.0
matplotlib==3.10.8
msgpack==1.1.2
numba==0.64.0
numpy==2.4.2
packaging==26.0
pandas==3.0.1
pillow==12.1.1
platformdirs==4.9.2
pooch==1.9.0
pycparser==3.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
requests==2.32.5
scikit-learn==1.8.0
scipy==1.17.0
seaborn==0.13.2
six==1.17.0
soundfile==0.13.1
soxr==1.0.0
threadpoolctl==3.6.0
typing_extensions==4.15.0
tzdata==2025.3
urllib3==2.6.3
(.venv)
```
'''

import numpy as np
import soundfile as sf
from scipy.signal import chirp
import random
import os
from utils import db_to_amplitude

class AudioSynthesizer:
    """
    音频合成器类，用于生成包含扫频信号和噪音的音频文件。
    """

    def __init__(self, sample_rate=48000, duration=10.0):
        """
        初始化合成器参数。

        :param sample_rate: 采样率
        :param duration: 音频总时长
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.total_samples = int(self.sample_rate * self.duration)

    def _generate_noise(self, length, db_level):
        """
        生成指定强度的随机高斯白噪音。

        :param length: 样本点数量
        :param db_level: 噪音强度
        :return: 噪音信号数组
        """
        amplitude = db_to_amplitude(db_level)
        # 生成高斯白噪音
        noise = np.random.randn(length) * amplitude
        return noise

    def _generate_sweep(self, start_freq, end_freq, sweep_duration, db_level, delay_samples=None):
        """
        生成对数扫频信号。

        :param start_freq: 起始频率
        :param end_freq: 结束频率
        :param sweep_duration: 扫频持续时间
        :param db_level: 扫频信号强度
        :param delay_samples: 随机添加的延迟采样点数，会延长扫频持续时长
        :return: 扫频信号数组
        """
        amplitude = db_to_amplitude(db_level)
        t = np.linspace(0, sweep_duration, int(self.sample_rate * sweep_duration), endpoint=False)

        # 使用scipy.signal.chirp生成对数扫频
        # method='logarithmic' 对应于频率随时间指数增长
        sweep_signal = chirp(t, f0=start_freq, f1=end_freq, t1=sweep_duration, method='logarithmic')

        # 如果需要，那么随机插入若干个采样点，其取值为附近的上一个采样点的值，用于模拟真实录制环境中设备时钟偏移产生的延迟
        if delay_samples is not None:
            for _ in range(delay_samples):
                delay_start = random.randint(1, len(sweep_signal) - 1)
                sample_value = sweep_signal[delay_start - 1]
                sweep_signal = np.insert(sweep_signal, delay_start, sample_value)

        return sweep_signal * amplitude

    def generate_audio_file(self, filename, delay_samples=None):
        """
        生成并保存音频文件。

        :param filename: 输出文件名
        """
        print(f"正在生成文件: {filename}")

        # 1. 初始化全时长噪音底噪 (-60dB)
        # 根据要求3：每个音频文件都独立添加强度为-60dB的随机噪音
        audio_data = self._generate_noise(self.total_samples, db_level=-60)

        # 2. 确定扫频信号的随机参数
        # 要求1：扫频信号的起始时间在1s到4s之间随机
        sweep_start_time = random.uniform(1.0, 4.0)
        sweep_duration = 5.0

        # 检查边界，确保扫频不超出文件末尾 (1s起始 + 5s时长 = 6s < 10s，逻辑安全)

        # 要求2：扫频信号强度在-3dB到-5dB之间随机
        sweep_db = random.uniform(-5.0, -3.0)

        print(f"  - 扫频起始时间: {sweep_start_time:.4f}s")
        print(f"  - 扫频信号强度: {sweep_db:.2f}dB")

        # 3. 生成扫频信号
        sweep_signal = self._generate_sweep(
            start_freq=20,
            end_freq=20000,
            sweep_duration=sweep_duration,
            db_level=sweep_db,
            delay_samples=delay_samples
        )

        # 4. 将扫频信号叠加到噪音底噪上
        start_sample = int(sweep_start_time * self.sample_rate)
        end_sample = start_sample + len(sweep_signal)

        # 确保不越界 (虽然逻辑上不会，但保留保护机制)
        if end_sample > self.total_samples:
            end_sample = self.total_samples
            sweep_signal = sweep_signal[:self.total_samples - start_sample]

        # 叠加信号
        audio_data[start_sample:end_sample] += sweep_signal

        # 5. 写入文件
        sf.write(filename, audio_data, self.sample_rate, subtype='PCM_24')
        print(f"  - 文件已保存。")

def main():
    # 实例化合成器
    synthesizer = AudioSynthesizer(sample_rate=48000, duration=10.0)

    os.makedirs(f"data", exist_ok=True)
    # 生成文件夹data/group1、data/group2、data/group3
    for i in range(1, 4):
        os.makedirs(f"data/group{i}", exist_ok=True)

    # 生成音频文件
    for i in range(3):
       synthesizer.generate_audio_file(f"data/group1/A_{i+1}.wav", 5)

    for i in range(3):
       synthesizer.generate_audio_file(f"data/group2/B_{i+1}.wav", 5)

    for i in range(3):
       synthesizer.generate_audio_file(f"data/group3/C_{i+1}.wav", 5)

if __name__ == "__main__":
    main()
