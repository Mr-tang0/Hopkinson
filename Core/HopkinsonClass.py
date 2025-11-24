import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from Core.signals import WAVE


class Sample:
    def __init__(self, number: int = 0, name: str = "sample", material: str = None, diameter_mm: float = 6,
                 length_mm: float = 4.5):
        self.number = number
        self.name = name
        self.material = material
        self.diameter_mm = diameter_mm
        self.length_mm = length_mm

    def __str__(self):
        return (f""
                f"第{self.number}号样品：{self.name} "
                f"材质：{self.material} 直径：{self.diameter_mm}mm 长度：{self.length_mm}mm")


class Hopkinson:
    def __init__(self):
        # 默认参数
        # 杆参数
        self.second_length = 62
        self.first_length = 62
        self.mode = "compression"  # 压缩或拉伸
        self.diameter_mm = 20.0  # 杆直径 mm
        self.YoungS_Pa = 210e9  # 杆Young's模量Pa
        self.soundVelocity_MPerS = 5200.0  # 声音速度-钢 m/s

        #  桥参数
        self.bridgeType = 2  # 桥类型半桥
        self.gageFactor = 1.8  # Gauge factor
        self.bridgeTension_v = 10  # 激励电压 V
        self.coefficient = 0.011111  # 标定系数
        self.halfCoefficient = self.coefficient / 2.0  # 标定系数一半

        # 色散校正
        self.poissonRatio = 0.25  # 泊松比-钢
        self.dampingCoefficient = 0.0  # 阻尼系数

    def __str__(self):
        return f"压杆 :ø{self.diameter_mm} mm" if self.mode == "compression" else f"拉杆 :ø{self.diameter_mm} mm"

    def setHopKinSonDetails(self, mode: str, diameter_mm: float, YoungS_Pa: float, soundVelocity_MPerS: float):
        self.mode = mode if mode in ["compression", "stretch"] else "compression"
        self.diameter_mm = diameter_mm
        self.YoungS_Pa = YoungS_Pa
        self.soundVelocity_MPerS = soundVelocity_MPerS

    def setBridgeDetails(self, bridgeType: int, gageFactor: float, bridgeTension_v: float, coefficient: float):
        self.bridgeType = bridgeType
        self.gageFactor = gageFactor
        self.bridgeTension_v = bridgeTension_v
        self.coefficient = coefficient
        self.halfCoefficient = self.coefficient / 2.0

    def setDispersionCorrection(self, poissonRatio: float, dampingCoefficient: float):
        self.poissonRatio = poissonRatio
        self.dampingCoefficient = dampingCoefficient

    def setLengthToSample(self, lengthA: float, lengthB: float):
        self.first_length = lengthA
        self.second_length = lengthB

    #  二波法计算 入射波+透射波
    def calculateWith(self, sample: Sample,
                      wave_inc: WAVE,
                      wave_trans: WAVE,
                      wave_ref: WAVE,
                      calculationType: str = "incAndTrans"):

        compression = -1 if self.mode == "compression" else 1
        #  使用桥系数 （理论）
        # denominator = Hp.bridge_type * Hp.gage_factor * Hp.bridge_tension
        # strain_incid = compression * wave_inc.wave_y / denominator
        # strain_reflected = wave_ref.wave_y / denominator

        # 直接使用标定系数 计算应变片曲线
        strain_inc = compression * wave_inc.wave_y * self.halfCoefficient
        strain_trans = compression * wave_trans.wave_y * self.halfCoefficient
        strain_ref = compression * wave_ref.wave_y * self.halfCoefficient

        # 采样时间间隔（假设均匀采样）
        dt = float(np.mean(np.diff(wave_inc.wave_x)))

        # 试样参数
        sample_length_M = sample.length_mm / 1000.0  # m
        area_ratio = (self.diameter_mm / sample.diameter_mm) ** 2

        if calculationType == "incAndTrans":
            eng_stress_Mpa = self.YoungS_Pa * area_ratio * strain_trans / 1e6  # 应力 MPa
            eng_strain_rate = (2 * self.soundVelocity_MPerS / sample_length_M) * (strain_inc - strain_trans)  # 应变率 1/s
            eng_strain = cumtrapz(eng_strain_rate, dx=dt, initial=0)  # 应变 1

        elif calculationType == "refAndTrans":
            eng_stress_Mpa = self.YoungS_Pa * area_ratio * strain_trans / 1e6  # 应力 MPa
            eng_strain_rate = -(2 * self.soundVelocity_MPerS / sample_length_M) * strain_ref  # 应变率 1/s
            eng_strain = cumtrapz(eng_strain_rate, dx=dt, initial=0)  # 应变 1

        elif calculationType == "threeWave":
            eng_stress_Mpa = 0.5 * self.YoungS_Pa * area_ratio * (strain_inc + strain_trans + strain_ref) / 1e6  # 应力 MPa
            eng_strain_rate = (self.soundVelocity_MPerS / sample_length_M) * (strain_inc - strain_ref - strain_trans)
            eng_strain = cumtrapz(eng_strain_rate, dx=dt, initial=0)

        else:  # 默认使用入射波+透射波
            area_ratio = (self.diameter_mm / sample.diameter_mm) ** 2
            eng_stress_Mpa = self.YoungS_Pa * area_ratio * strain_trans / 1e6  # 应力 MPa
            eng_strain_rate = (2 * self.soundVelocity_MPerS / sample_length_M) * (strain_inc - strain_trans)  # 应变率 1/s
            eng_strain = cumtrapz(eng_strain_rate, dx=dt, initial=0)  # 应变 1

        # 应变-时间曲线
        eng_strain_time = WAVE(x=wave_inc.wave_x, y=eng_strain)

        # 应变率-时间曲线
        eng_strain_rate_time = WAVE(x=wave_inc.wave_x, y=eng_strain_rate)

        # 应力-时间曲线
        eng_stress_time = WAVE(x=wave_inc.wave_x, y=eng_stress_Mpa)

        # 应变-应力曲线
        eng_stress_strain = WAVE(x=eng_strain, y=eng_stress_Mpa)

        result = {
            "stress_strain": eng_stress_strain,
            "strain_time": eng_strain_time,
            "strain_rate_time": eng_strain_rate_time,
            "stress_time": eng_stress_time,
        }

        return result
