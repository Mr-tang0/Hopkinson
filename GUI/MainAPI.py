import io
import json
import os
from datetime import datetime

import numpy as np
import pandas
import webview
# from scipy.ndimage import gaussian_filter1d
from Core.numpyTool import gaussian_filter1d

from Core.signals import WAVE
from Core.HopkinsonClass import Sample, Hopkinson


def find_peaks_with_smoothing(data, sigma=1.0, height=None):
    # 对数据进行高斯平滑
    smoothed_data = gaussian_filter1d(data, sigma=sigma)

    # 判断信号整体趋势：正数多则找上顶点，负数多则找下顶点
    signal_mean = np.mean(data)
    find_positive_peaks = signal_mean >= 0

    # 在原始数据上查找平滑后的峰值位置
    peaks = []
    for i in range(1, len(smoothed_data) - 1):
        if find_positive_peaks:
            # 检测局部最大值（上顶点）
            if (smoothed_data[i] > smoothed_data[i - 1] and
                    smoothed_data[i] > smoothed_data[i + 1]):
                # 可选：设置高度阈值
                if height is None or data[i] >= height:
                    peaks.append(i)
        else:
            # 检测局部最小值（下顶点）
            if (smoothed_data[i] < smoothed_data[i - 1] and
                    smoothed_data[i] < smoothed_data[i + 1]):
                # 可选：设置高度阈值（对于负值）
                if height is None or data[i] <= (-height if height else 0):
                    peaks.append(i)

    peaks = np.array(peaks)
    # 如果找到了峰值点，过滤掉小于最大幅值25%的点
    if len(peaks) > 0:
        peak_values = data[peaks]
        max_value = np.max(np.abs(peak_values))
        threshold = max_value * 0.25

        # 只保留幅值大于阈值的峰值点
        valid_peaks = peaks[np.abs(peak_values) >= threshold]
        return valid_peaks

    return peaks


def normalize_array(data):
    data_min = np.min(data)
    data_max = np.max(data)

    # 避免除零错误
    if data_max - data_min == 0:
        return np.zeros_like(data)

    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data


class Api:
    def __init__(self):
        # 原始数据
        self.result = None

        self.WAVE_second_ori = WAVE()
        self.WAVE_first_ori = WAVE()

        self.WAVE_second = WAVE()
        self.WAVE_first = WAVE()

        # fft数据
        self.WAVE_first_fft = WAVE()
        self.WAVE_second_fft = WAVE()

        # 滤波后数据
        self.WAVE_first_filtered = WAVE()
        self.WAVE_second_filtered = WAVE()

        # 入射、反射、透射波
        self.WAVE_Inc = WAVE()
        self.WAVE_Ref = WAVE()
        self.WAVE_Trans = WAVE()
        self.cropData = {}
        self.cropDataStartAndEnd = {}  # 截取原始数据起始和结束点， 用于手动调整

        self.sample = Sample()
        self.hopkinson = Hopkinson()

        self.cachePath = "cache.json"  # 历史缓存路径
        self.cache = {}

    def getCacheData(self):
        try:
            with open(self.cachePath, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
            return {"success": True, "message": "历史数据加载成功", "data": self.cache}
        except FileNotFoundError:
            write_file = open(self.cachePath, "w", encoding="utf-8")
            json.dump({}, write_file)

            return {"success": False, "message": "历史数据不存在", "data": {}}

    def saveCacheData(self):
        try:
            with open(self.cachePath, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=4)
            return {"success": True, "message": "历史数据保存成功"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def loadData(self, incid_content: str, trans_content: str):
        print("loadData")
        self.__init__()
        inc_df = pandas.read_csv(io.StringIO(incid_content), skiprows=10)
        # 交换入射波数据的两列
        inc_df.iloc[:, [0, 1]] = inc_df.iloc[:, [1, 0]].values

        self.WAVE_first = WAVE(wave=np.array(inc_df))
        self.WAVE_first_ori = self.WAVE_first

        tra_df = pandas.read_csv(io.StringIO(trans_content), skiprows=10)
        # 交换透射波数据的两列
        tra_df.iloc[:, [0, 1]] = tra_df.iloc[:, [1, 0]].values

        self.WAVE_second = WAVE(wave=np.array(tra_df))
        self.WAVE_second_ori = self.WAVE_second

        self.WAVE_first_fft = self.WAVE_first.fft()
        self.WAVE_second_fft = self.WAVE_second.fft()

        self.WAVE_first_filtered = self.WAVE_first
        self.WAVE_second_filtered = self.WAVE_second

        fft_data = {
            "入射信号": {
                "x": list(self.WAVE_first_fft.wave_x),
                "y": list(self.WAVE_first_fft.wave_y)
            },
            "透射信号": {
                "x": list(self.WAVE_second_fft.wave_x),
                "y": list(self.WAVE_second_fft.wave_y)
            }
        }

        chart_data = {
            "入射信号": {
                "x": list(self.WAVE_first.wave_x),
                "y": list(self.WAVE_first.wave_y)
            },
            "透射信号": {
                "x": list(self.WAVE_second.wave_x),
                "y": list(self.WAVE_second.wave_y)
            }
        }

        return {"success": True, "message": "数据加载完成", "data": [chart_data, fft_data],
                "title": ["原始数据", "频谱数据"]}

    # 基线校正
    def baselineCorrection(self, baseline, wave):
        try:
            print("基线校正", baseline, wave)
            if wave == "inc":
                self.WAVE_first = WAVE(x=self.WAVE_first_ori.wave_x, y=self.WAVE_first_ori.wave_y + baseline)
                self.WAVE_first_filtered = self.WAVE_first
            elif wave == "trans":
                self.WAVE_second = WAVE(x=self.WAVE_second_ori.wave_x, y=self.WAVE_second_ori.wave_y + baseline)
                self.WAVE_second_filtered = self.WAVE_second
            else:
                return {"success": False, "error": "数据类型错误"}

            data = {
                "入射信号": {
                    "x": list(self.WAVE_first_filtered.wave_x),
                    "y": list(self.WAVE_first_filtered.wave_y)
                },
                "透射信号": {
                    "x": list(self.WAVE_second_filtered.wave_x),
                    "y": list(self.WAVE_second_filtered.wave_y)
                }
            }

            return {"success": True, "message": "数据加载完成", "data": data, "title": "原始数据"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def getLocalSampleData(self):
        localSample = {
            "number": self.sample.number,
            "name": self.sample.name,
            "material": self.sample.material,
            "area": self.sample.diameter_mm,
            "length": self.sample.length_mm
        }
        return localSample

    def saveSampleData(self, sample_data):
        try:
            self.sample = Sample(number=sample_data.get('number', ''),
                                 name=sample_data.get('name', ''),
                                 material=sample_data.get('material', ''),
                                 diameter_mm=float(sample_data.get('area', 0)),
                                 length_mm=float(sample_data.get('length', 0)))
            self.cache["sample"] = sample_data
            self.saveCacheData()
            return {"success": True, "message": "样品数据保存成功"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def saveHopkinsonData(self, hopkinson_data):
        try:

            youngS_Pa = float(hopkinson_data.get('youngs', 0)) * 1e9
            self.hopkinson.setHopKinSonDetails(mode=hopkinson_data.get('mode', ''),
                                               diameter_mm=float(hopkinson_data.get('diameter', 0)),
                                               YoungS_Pa=youngS_Pa,
                                               soundVelocity_MPerS=float(hopkinson_data.get('soundVelocity', 0)))

            self.hopkinson.setBridgeDetails(bridgeType=int(hopkinson_data.get('bridgeType', 0)),
                                            gageFactor=float(hopkinson_data.get('gageFactor', 0)),
                                            bridgeTension_v=float(hopkinson_data.get('excitationVoltage', 0)),
                                            coefficient=float(hopkinson_data.get('calibrationFactor', 0)))

            self.hopkinson.setDispersionCorrection(poissonRatio=float(hopkinson_data.get('poissonRatio', 0)),
                                                   dampingCoefficient=float(
                                                       hopkinson_data.get('dampingCoefficient', 0)))

            self.hopkinson.setLengthToSample(float(hopkinson_data.get('lengthA', 0)),
                                             float(hopkinson_data.get('lengthB', 0)))

            self.cache["hopkinson"] = hopkinson_data
            self.saveCacheData()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def applyFilter(self, low_cutoff_freq, high_cutoff_freq):
        self.WAVE_first_filtered = self.WAVE_first.lowpass_filter(cutoff_freq=float(high_cutoff_freq))
        self.WAVE_second_filtered = self.WAVE_second.lowpass_filter(cutoff_freq=float(high_cutoff_freq))

        filtered_data = {
            "入射波": {
                "x": list(self.WAVE_first_filtered.wave_x),
                "y": list(self.WAVE_first_filtered.wave_y)
            },
            "透射波": {
                "x": list(self.WAVE_second_filtered.wave_x),
                "y": list(self.WAVE_second_filtered.wave_y)
            }
        }
        return {"success": True, "message": "滤波完成", "data": filtered_data, "title": "原始数据"}

    def cropSignal(self, cropStart, cropEnd, SignalType):
        if SignalType == "inc":
            self.WAVE_Inc = self.WAVE_first_filtered.crop(cropStart, cropEnd)
            self.cropDataStartAndEnd["inc"] = {
                "start": cropStart,
                "end": cropEnd
            }
            self.cropData["入射波"] = {
                "x": list(self.WAVE_Inc.wave_x),
                "y": list(self.WAVE_Inc.wave_y)
            }
            return {"success": True, "message": "截取完成", "data": self.cropData, "title": "时域对齐"}
        elif SignalType == "trans":
            self.WAVE_Trans = self.WAVE_second_filtered.crop(cropStart, cropEnd)
            self.cropDataStartAndEnd["trans"] = {
                "start": cropStart,
                "end": cropEnd
            }
            self.cropData["透射波"] = {
                "x": list(self.WAVE_Trans.wave_x),
                "y": list(self.WAVE_Trans.wave_y)
            }
            return {"success": True, "message": "截取完成", "data": self.cropData, "title": "时域对齐"}
        elif SignalType == "ref":
            self.WAVE_Ref = self.WAVE_first_filtered.crop(cropStart, cropEnd)
            self.cropDataStartAndEnd["ref"] = {
                "start": cropStart,
                "end": cropEnd
            }
            self.cropData["反射波"] = {
                "x": list(self.WAVE_Ref.wave_x),
                "y": list(self.WAVE_Ref.wave_y)
            }
            return {"success": True, "message": "截取完成", "data": self.cropData, "title": "时域对齐"}
        else:
            return {"success": False, "message": "截取失败"}

    #  使用国标推荐方法，确定各个波的起点，要求此时只标定了入射波
    #  参考文献https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=D484174464EFFF3C22B1B8CEABF87E20
    def autoAlignWithGB(self):
        # 确定基线：取入射波开始点前100个点的平均值
        start = self.cropDataStartAndEnd["inc"].get("start", 0)

        baseline = self.WAVE_first_filtered.wave_y[start - 200:start - 100].mean()

        # 获取入射波幅值最大index
        tempWave = abs(self.WAVE_Inc.wave_y - baseline)
        max_value = tempWave.max()

        # 获取第一次超过最大值十分之一点的index
        threshold = max_value * 0.1
        first_over_threshold_index = np.argmax(tempWave > threshold)
        print("first_over_threshold_index", first_over_threshold_index)

        # 获取此点左右各100个点，根据两点做直线获取与基线相交的点的index
        left_index = max(0, first_over_threshold_index - 100)
        right_index = min(len(self.WAVE_Inc.wave_y) - 1, first_over_threshold_index + 100)

        # 获取左右两点的值
        x1, y1 = left_index, self.WAVE_Inc.wave_y[left_index]
        x2, y2 = right_index, self.WAVE_Inc.wave_y[right_index]

        # 计算直线方程 y = ax + b
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        # 计算与基线相交的点的index 此为入射波开始点
        n1 = int((baseline - b) / a)

        # 采样时间间隔ms
        ti = float(np.mean(np.diff(self.WAVE_Inc.wave_x)) * 1000)

        a1, a2 = self.hopkinson.first_length + self.sample.length_mm / 2, self.hopkinson.second_length + self.sample.length_mm / 2

        cb = self.hopkinson.soundVelocity_MPerS

        l0 = self.sample.length_mm
        cs = cb

        n2 = int(2 * a1 / (cb * ti)) + n1
        n3 = int((a1 + a2) / (cb * ti) + l0 / (cs * ti)) + n1

        # 自动截取入射波透射波、反射波（均向左偏移100个点）
        length = self.WAVE_Inc.len()
        self.cropSignal(cropStart=n1 + start - 100, cropEnd=n1 + start - 100 + length,
                        SignalType="inc")

        self.cropSignal(cropStart=n2 + start - 100, cropEnd=n2 + start - 100 + length,
                        SignalType="ref")

        self.cropSignal(cropStart=n3 + start - 100,
                        cropEnd=n3 + start - 100 + length, SignalType="trans")

        # 记录区间
        self.cropDataStartAndEnd["inc"] = {
            "start": n1,
            "end": n1 + length
        }
        self.cropDataStartAndEnd["ref"] = {
            "start": n2,
            "end": n2 + length
        }
        self.cropDataStartAndEnd["trans"] = {
            "start": n3,
            "end": n3 + length
        }

        # 对入射波时域归零
        self.WAVE_Inc = self.WAVE_Inc.alignToZero()
        # 对反射波进行时域对齐
        self.WAVE_Ref = self.WAVE_Ref.alignWith(self.WAVE_Inc)
        # 对透射波进行时域对齐
        self.WAVE_Trans = self.WAVE_Trans.alignWith(self.WAVE_Inc)

        pass

    def autoAlignWithTime(self):
        # 对三个波进行等长裁剪
        minLength = min(self.WAVE_Inc.len(), self.WAVE_Ref.len(), self.WAVE_Trans.len())
        self.WAVE_Inc = self.WAVE_Inc.crop(0, minLength)
        self.WAVE_Ref = self.WAVE_Ref.crop(0, minLength)
        self.WAVE_Trans = self.WAVE_Trans.crop(0, minLength)

        # 对入射波时域归零
        self.WAVE_Inc = self.WAVE_Inc.alignToZero()
        # 对反射波进行时域对齐
        self.WAVE_Ref = self.WAVE_Ref.alignWith(self.WAVE_Inc)
        # 对透射波进行时域对齐
        self.WAVE_Trans = self.WAVE_Trans.alignWith(self.WAVE_Inc)

        self.cropDataStartAndEnd["trans"] = {
            "start": self.cropDataStartAndEnd["trans"]["start"],
            "end": self.cropDataStartAndEnd["trans"]["start"] + self.WAVE_Trans.len()
        }
        self.cropDataStartAndEnd["ref"] = {
            "start": self.cropDataStartAndEnd["ref"]["start"],
            "end": self.cropDataStartAndEnd["ref"]["start"] + self.WAVE_Ref.len()
        }
        self.cropDataStartAndEnd["inc"] = {
            "start": self.cropDataStartAndEnd["inc"]["start"],
            "end": self.cropDataStartAndEnd["inc"]["start"] + self.WAVE_Inc.len()
        }

    #  根据第一个顶点对齐
    def alignWithWave(self):
        minLength = min(self.WAVE_Inc.len(), self.WAVE_Ref.len(), self.WAVE_Trans.len())
        self.WAVE_Inc = self.WAVE_Inc.crop(0, minLength)  # 先裁剪入射波长度

        # 获取入射波顶点
        peak_inc = find_peaks_with_smoothing(self.WAVE_Inc.wave_y)
        # 获取反射波顶点
        peak_ref = find_peaks_with_smoothing(self.WAVE_Ref.wave_y)
        # 获取透射波顶点
        peak_trans = find_peaks_with_smoothing(self.WAVE_Trans.wave_y)

        # 计算顶点偏差
        shift_ref_to_inc = peak_inc[0] - peak_ref[0]
        shift_trans_to_inc = peak_inc[0] - peak_trans[0]

        # 根据偏差重新从滤波数据中裁剪数据
        start_ref = self.cropDataStartAndEnd["ref"]["start"]
        self.WAVE_Ref = self.WAVE_first_filtered.crop(start_ref - shift_ref_to_inc,
                                                      start_ref + minLength - shift_ref_to_inc)
        start_trans = self.cropDataStartAndEnd["trans"]["start"]
        self.WAVE_Trans = self.WAVE_second_filtered.crop(start_trans - shift_trans_to_inc,
                                                         start_trans + minLength - shift_trans_to_inc)

        # 对入射波时域归零
        self.WAVE_Inc = self.WAVE_Inc.alignToZero()
        # 对反射波进行时域对齐
        self.WAVE_Ref = self.WAVE_Ref.alignWith(self.WAVE_Inc)
        # 对透射波进行时域对齐
        self.WAVE_Trans = self.WAVE_Trans.alignWith(self.WAVE_Inc)

        # 更新起终点
        self.cropDataStartAndEnd["trans"] = {
            "start": start_trans + shift_trans_to_inc,
            "end": start_trans + minLength + shift_trans_to_inc
        }
        self.cropDataStartAndEnd["ref"] = {
            "start": start_ref + shift_ref_to_inc,
            "end": start_ref + minLength + shift_ref_to_inc
        }
        self.cropDataStartAndEnd["inc"] = {
            "start": self.cropDataStartAndEnd["inc"]["start"],
            "end": self.cropDataStartAndEnd["inc"]["start"] + self.WAVE_Inc.len()
        }

        pass

    def alignWithCorr(self):
        # 将数据全部正向翻转并标准化
        reverse_Inc = WAVE(wave=self.WAVE_Inc.wave)
        reverse_ref = WAVE(wave=self.WAVE_Ref.wave)
        reverse_trans = WAVE(wave=self.WAVE_Trans.wave)

        inc_mean = np.mean(self.WAVE_Inc.wave_y)
        reverse_Inc.wave_y = normalize_array(reverse_Inc.wave_y) if inc_mean > 0 else normalize_array(
            -reverse_Inc.wave_y)

        ref_mean = np.mean(self.WAVE_Ref.wave_y)
        reverse_ref.wave_y = normalize_array(reverse_ref.wave_y) if ref_mean > 0 else normalize_array(
            -reverse_ref.wave_y)

        trans_mean = np.mean(self.WAVE_Trans.wave_y)
        reverse_trans.wave_y = normalize_array(reverse_trans.wave_y) if trans_mean > 0 else normalize_array(
            -reverse_trans.wave_y)

        # 计算与reverse_Inc.wave_y的互相关偏差
        # 计算反射波与入射波的互相关
        correlation_ref = np.correlate(reverse_Inc.wave_y, reverse_ref.wave_y, mode='full')
        lag_ref = np.argmax(correlation_ref) - (len(reverse_Inc.wave_y) - 1)
        print("ref_lag:", lag_ref, "correlation_ref", correlation_ref)

        # 计算透射波与入射波的互相关
        correlation_trans = np.correlate(reverse_Inc.wave_y, reverse_trans.wave_y, mode='full')
        lag_trans = np.argmax(correlation_trans) - (len(reverse_Inc.wave_y) - 1)

        print("ref_lag:", lag_ref, "trans_lag:", lag_trans)

        # # 根据偏差重新裁剪数据
        # start_ref = self.cropDataStartAndEnd["ref"]["start"]
        # self.WAVE_Ref = self.WAVE_first_filtered.crop(start_ref - lag_ref,
        #                                               start_ref + len(reverse_Inc.wave_y) - lag_ref)
        #
        # start_trans = self.cropDataStartAndEnd["trans"]["start"]
        # self.WAVE_Trans = self.WAVE_second_filtered.crop(start_trans - lag_trans,
        #                                                  start_trans + len(reverse_Inc.wave_y) - lag_trans)
        #
        # # 对入射波时域归零
        # self.WAVE_Inc = reverse_Inc.alignToZero()
        # # 对反射波进行时域对齐
        # self.WAVE_Ref = self.WAVE_Ref.alignWith(self.WAVE_Inc)
        # # 对透射波进行时域对齐
        # self.WAVE_Trans = self.WAVE_Trans.alignWith(self.WAVE_Inc)
        #
        # # 更新起终点
        # self.cropDataStartAndEnd["ref"] = {
        #     "start": start_ref - lag_ref,
        #     "end": start_ref + len(reverse_Inc.wave_y) - lag_ref
        # }
        # self.cropDataStartAndEnd["trans"] = {
        #     "start": start_trans - lag_trans,
        #     "end": start_trans + len(reverse_Inc.wave_y) - lag_trans
        # }

        pass

    def alignWithMethod(self, method):
        if method == "time":
            self.autoAlignWithTime()

        if method == "gb":
            self.autoAlignWithGB()

        if method == "corr":
            self.alignWithCorr()

        if method == "wave":
            self.alignWithWave()

        self.cropData["入射波"] = {
            "x": list(self.WAVE_Inc.wave_x),
            "y": list(self.WAVE_Inc.wave_y)
        }
        self.cropData["反射波"] = {
            "x": list(self.WAVE_Ref.wave_x),
            "y": list(self.WAVE_Ref.wave_y)
        }
        self.cropData["透射波"] = {
            "x": list(self.WAVE_Trans.wave_x),
            "y": list(self.WAVE_Trans.wave_y)
        }

        return {"success": True, "message": "时域对齐完成", "data": self.cropData, "title": "时域对齐"}

    def alignManually(self, shift_str, waveName, maxRange):
        if self.WAVE_Inc.wave_y is not None and (self.WAVE_Ref.wave_y is None or self.WAVE_Trans.wave_y is None):
            return {"success": False, "message": "调整失败"}

        start = self.cropDataStartAndEnd.get(waveName).get('start')
        end = self.cropDataStartAndEnd.get(waveName).get('end')

        shift = int(int(maxRange) / 2 - int(shift_str))
        # print("alignManually", self.WAVE_Inc.len(), self.WAVE_Ref.len(), self.WAVE_Trans.len(), shift)

        if waveName == "inc":
            self.WAVE_Inc = self.WAVE_first_filtered.crop(start + shift, end + shift)
            self.WAVE_Inc = self.WAVE_Inc.alignToZero()
            self.cropData["入射波"] = {
                "x": list(self.WAVE_Inc.wave_x),
                "y": list(self.WAVE_Inc.wave_y)
            }
            return {"success": True, "message": "调整完成", "data": self.cropData, "title": "时域对齐"}
        elif waveName == "trans":
            self.WAVE_Trans = self.WAVE_second_filtered.crop(start + shift, end + shift)
            self.WAVE_Trans = self.WAVE_Trans.alignToZero()
            self.cropData["透射波"] = {
                "x": list(self.WAVE_Trans.wave_x),
                "y": list(self.WAVE_Trans.wave_y)
            }
            return {"success": True, "message": "调整完成", "data": self.cropData, "title": "时域对齐"}
        elif waveName == "ref":
            self.WAVE_Ref = self.WAVE_first_filtered.crop(start + shift, end + shift)
            self.WAVE_Ref = self.WAVE_Ref.alignToZero()
            self.cropData["反射波"] = {
                "x": list(self.WAVE_Ref.wave_x),
                "y": list(self.WAVE_Ref.wave_y)
            }
            return {"success": True, "message": "调整完成", "data": self.cropData, "title": "时域对齐"}
        else:
            return {"success": False, "message": "调整失败"}

    def startCalculation(self, calculationType: str):
        self.result = self.hopkinson.calculateWith(sample=self.sample,
                                                   wave_inc=self.WAVE_Inc,
                                                   wave_trans=self.WAVE_Trans,
                                                   wave_ref=self.WAVE_Ref,
                                                   calculationType=calculationType)

        result_data = {
            "应力-应变":
                {
                    "x": list(self.result.get("stress_strain").wave_x),
                    "y": list(self.result.get("stress_strain").wave_y)
                },
            "应变-时间":
                {
                    "x": list(self.result.get("strain_time").wave_x),
                    "y": list(self.result.get("strain_time").wave_y)
                },
            "应力-时间":
                {
                    "x": list(self.result.get("stress_time").wave_x),
                    "y": list(self.result.get("stress_time").wave_y)
                },
            "应变率-时间":
                {
                    "x": list(self.result.get("strain_rate_time").wave_x),
                    "y": list(self.result.get("strain_rate_time").wave_y),
                }
        }
        return {"success": True, "message": "计算完成", "data": result_data, "title": "计算结果"}

    def exportResults(self):
        try:
            folder_path = webview.windows[0].create_file_dialog(
                dialog_type=webview.FOLDER_DIALOG,
                directory=''
            )
            folder_path = folder_path[0] if folder_path else None

            if folder_path is not None:
                folder_path = os.path.join(folder_path, f"hopkinson_result_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                detailJson = {
                    "sample": {
                        "number": self.sample.number,
                        "name": self.sample.name,
                        "material": self.sample.material,
                        "diameter_mm": self.sample.diameter_mm,
                        "length_mm": self.sample.length_mm
                    },
                    "hopkinson": {
                        "mode": self.hopkinson.mode,
                        "diameter_mm": self.hopkinson.diameter_mm,
                        "YoungS_Pa": self.hopkinson.YoungS_Pa,
                        "soundVelocity_MPerS": self.hopkinson.soundVelocity_MPerS,

                        "bridgeType": self.hopkinson.bridgeType,
                        "gageFactor": self.hopkinson.gageFactor,
                        "bridgeTension_v": self.hopkinson.bridgeTension_v,
                        "coefficient": self.hopkinson.coefficient,
                        "halfCoefficient": self.hopkinson.halfCoefficient,

                        "poissonRatio": self.hopkinson.poissonRatio,
                        "dampingCoefficient": self.hopkinson.dampingCoefficient
                    }
                }
                json_file_path = os.path.join(folder_path, "detail.json")
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(detailJson, json_file, ensure_ascii=False, indent=4)

                result_data = pandas.DataFrame({
                    'time': self.WAVE_Inc.wave_x,
                    'wave_inc': self.WAVE_Inc.wave_y,
                    'wave_trans': self.WAVE_Trans.wave_y,
                    'wave_ref': self.WAVE_Ref.wave_y,
                    'strain': self.result.get("strain_time").wave_y,
                    'stress': self.result.get("stress_time").wave_y,
                    'strain_rate': self.result.get("strain_rate_time").wave_y
                })
                result_data_path = os.path.join(folder_path, "result.csv")
                result_data.to_csv(result_data_path, index=False)

                return {'success': True, 'message': '结果已导出'}
            else:
                return {'success': False, 'message': '用户取消了操作'}
        except Exception as e:
            return {'success': False, 'message': f'打开对话框时出错: {str(e)}'}
