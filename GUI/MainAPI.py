import io
import json

import numpy as np
import pandas

from Core.signals import WAVE
from Core.HopkinsonClass import Sample, Hopkinson
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class Api:
    def __init__(self):
        # 原始数据
        self.result = None
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
        inc_df = pandas.read_csv(io.StringIO(incid_content))
        self.WAVE_first = WAVE(wave=np.array(inc_df))
        tra_df = pandas.read_csv(io.StringIO(trans_content))
        self.WAVE_second = WAVE(wave=np.array(tra_df))

        self.WAVE_first_fft = self.WAVE_first.fft()
        self.WAVE_second_fft = self.WAVE_second.fft()

        self.WAVE_first_filtered = self.WAVE_first
        self.WAVE_second_filtered = self.WAVE_second

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

        return {"success": True, "message": "数据加载完成", "data": [chart_data, fft_data],
                "title": ["原始数据", "频谱数据"]}

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

            self.cache["hopkinson"] = hopkinson_data
            self.saveCacheData()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def applyFilter(self, low_cutoff_freq, high_cutoff_freq):
        print(low_cutoff_freq, high_cutoff_freq)

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

    def alignWithTime(self):
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
        start = self.cropDataStartAndEnd.get(waveName).get('start')
        end = self.cropDataStartAndEnd.get(waveName).get('end')

        shift = int(int(maxRange / 2) - int(shift_str))

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
            # "应变-时间":
            #     {
            #         "x": list(self.result.get("strain_time").wave_x),
            #         "y": list(self.result.get("strain_time").wave_y)
            #     },
            # "应力-时间":
            #     {
            #         "x": list(self.result.get("stress_time").wave_x),
            #         "y": list(self.result.get("stress_time").wave_y)
            #     },
            # "应变率-时间":
            #     {
            #         "x": list(self.result.get("strain_rate_time").wave_x),
            #     },

        }
        return {"success": True, "message": "计算完成", "data": result_data, "title": "工程应力应变"}
