from __future__ import annotations
import numpy as np
# from scipy.signal import butter, filtfilt
from .numpyTool import butter_numpy as butter
from .numpyTool import filtfilt_numpy as filtfilt


class WAVE:
    def __init__(self, wave: np.array = None, x: np.array = None, y: np.array = None, name: str = None):
        if wave is None and x is not None and y is not None:
            self.wave_x = x
            self.wave_y = y
            self.wave = np.array([y, x]).T
        elif wave is not None:
            self.wave = wave
            self.wave_x = self.wave[:, 1]
            self.wave_y = wave[:, 0]
        else:
            self.wave = None
            self.wave_x = None
            self.wave_y = None

        self.name = name if name is not None else "WAVE"

    def __str__(self):
        return f"WAVE: {self.name}, len :{self.len()}"

    def len(self):
        try:
            if self.wave is not None:
                return len(self.wave_x)
            else:
                return 0
        except Exception as e:
            print("something get worry", e)

        return self.wave.shape

    # def plot(self, waveList: list = None, title: str = None):
    #     plt.figure(figsize=(10, 5))
    #
    #     plt.plot(self.wave_x, self.wave_y, label=self.name)
    #     if waveList is not None:
    #         for wave in waveList:
    #             plt.plot(wave.wave_x, wave.wave_y, label=wave.name)
    #
    #     plt.title(title)
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    def crop(self, start_index: int, end_index: int = None, length: int = None):
        try:
            if end_index is not None:
                if end_index > self.len() or end_index < start_index:
                    return None

                wave = self.wave[start_index:end_index]
            elif length is not None and end_index is None:
                if start_index + length > self.len():
                    return None

                wave = self.wave[start_index:start_index + length]
            else:
                return None
            return WAVE(wave=wave)
        except Exception as e:
            print("Cannot crop:", e)

    def alignToZero(self):
        targetStart = self.wave_x[0]
        wave = WAVE(x=self.wave_x - targetStart, y=self.wave_y)
        return wave

    # 根据某个信号对齐
    def alignWith(self, wave: WAVE):
        try:
            targetStart = wave.wave_x[0]
            currentStart = self.wave_x[0]

            shift = targetStart - currentStart
            newWave = WAVE(x=self.wave_x + shift, y=self.wave_y)

            return newWave
        except Exception as e:
            print("Cannot align:", e)
            return None

    def fft(self):
        """
        绘制 WAVE 对象的 FFT 幅度谱（单边）
        要求 wave.wave_x 是时间（单位：秒）
        """
        if self.wave_y is None or self.wave_x is None:
            raise ValueError("wave_x (time) and wave_y (signal) must be set")

        y = self.wave_y
        t = self.wave_x
        N = len(y)

        # 采样时间间隔（假设均匀采样）
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt  # 采样频率 (Hz)

        # 执行 FFT
        Y = np.fft.fft(y)
        freqs = np.fft.fftfreq(N, d=dt)  # 包含负频率

        # 只取正频率部分（单边谱）
        positive_freq_idx = freqs >= 0
        freqs_pos = freqs[positive_freq_idx]
        Y_pos = Y[positive_freq_idx]

        # 幅度（单边谱通常要乘以 2/N，但首项（DC）不乘）
        magnitude = np.abs(Y_pos)
        magnitude[1:] *= 2  # 除 DC 分量外，其余乘 2 补偿能量
        magnitude /= N

        return WAVE(x=freqs_pos, y=magnitude)

    def lowpass_filter(self, cutoff_freq=100000.0, order=4) -> WAVE:
        if self.wave_y is None or self.wave_x is None:
            raise ValueError("wave_x and wave_y must be set")

        y = self.wave_y
        t = self.wave_x
        N = len(y)

        # 计算采样频率 fs (Hz)
        dt = np.mean(np.diff(t))  # 假设均匀采样
        fs = 1.0 / dt

        # 安全检查：截止频率不能超过奈奎斯特频率 (fs/2)
        nyquist = 0.5 * fs
        if cutoff_freq >= nyquist:
            print(f"Warning: cutoff frequency ({cutoff_freq} Hz) >= Nyquist ({nyquist:.1f} Hz). No filtering applied.")
            return WAVE(x=t.copy(), y=y.copy())
        # 归一化截止频率 (0 ~ 1, where 1 = nyquist)
        normal_cutoff = cutoff_freq / nyquist

        # Butterworth 低通滤波器
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        # 应用零相位滤波（避免时移）
        y_filtered = filtfilt(b, a, y)

        return WAVE(x=t.copy(), y=y_filtered)
