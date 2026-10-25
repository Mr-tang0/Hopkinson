package backend

import (
	"math"
	"math/cmplx"

	"github.com/mjibson/go-dsp/fft"
)

// DispersionCorrector 色散矫正器
type DispersionCorrector struct {
	// 杆的物理参数
	SoundVelocity float64 // C₀: 一维声速 (m/s)
	PoissonRatio  float64 // ν: 泊松比
	Radius        float64 // a: 杆半径 (m)

	// 传播距离 (应变片到试样端面的距离, m)
	IncPropagationDist   float64 // 入射杆: 应变片到试样入射端面距离 (m)
	TransPropagationDist float64 // 透射杆: 试样透射端面到应变片距离 (m)
}

// PochhammerChreeSolver 求解Pochhammer-Chree方程，返回归一化相速度 C = Cp/C₀
// 使用简化近似: 对于常见材料，可用多项式拟合或查表
// 这里采用Gorham的近似公式
func (dc *DispersionCorrector) PochhammerChreeSolver(nu float64, F float64) float64 {
	// F = f*a/C₀ (归一化频率)
	// 当 F 很小时，C ≈ 1 (无弥散)
	if F < 0.001 {
		return 1.0
	}

	// 使用Davies的近似公式 (适用于第一模态, F < 0.8)
	// C ≈ 1 + A*F² + B*F⁴
	// 其中 A, B 是泊松比ν的函数
	A := -0.5 * nu * nu
	B := 0.125 * nu * nu * (1 - nu*nu)

	// 更精确的近似 (Rayleigh-Love修正)
	C2 := 1.0 + A*F*F + B*F*F*F*F

	// 确保结果合理
	if C2 < 0.01 {
		return 0.01
	}
	return math.Sqrt(C2)
}

// FrequencyCorrection 对单个信号进行频域色散矫正
// 把应变片位置测到的波形还原为试样端面处、按一维声速传播的波形：
//
//	u_corr(ω) = u_meas(ω) * exp[i·ω·Δz·(1/C0 - 1/Cp)]
//
// 其中相位项同时包含：物理传播延迟 Δz/Cp 的补偿，以及按名义一维波速
// 对齐时间轴所需的 Δz/C0 提前量。负号约定与 go-dsp 的 FFT(IFFT) 一致。
//
// propagationDist: 应变片到目标端面的距离 (m)
func (dc *DispersionCorrector) FrequencyCorrection(timeSignal []float64, dt float64, propagationDist float64) []float64 {
	n := len(timeSignal)
	if n < 2 || dt <= 0 || dc.SoundVelocity <= 0 {
		return timeSignal
	}
	// 距离为 0 时无需矫正
	if math.Abs(propagationDist) < 1e-12 {
		return timeSignal
	}

	// 补零到不小于 2n 的 2 的幂，避免相位旋转造成循环卷绕
	nPad := 1
	for nPad < 2*n {
		nPad <<= 1
	}

	complexSignal := make([]complex128, nPad)
	for i := 0; i < n; i++ {
		complexSignal[i] = complex(timeSignal[i], 0)
	}
	freqSignal := fft.FFT(complexSignal)

	// 频率分辨率（按补零后的长度计算）
	df := 1.0 / (float64(nPad) * dt)

	// 逐频率分量做相位矫正（负频率按对称关系单独处理）
	for i := 0; i < nPad; i++ {
		freq := float64(i) * df
		if i > nPad/2 {
			freq = -float64(nPad-i) * df
		}
		if freq == 0 {
			continue
		}

		// 归一化频率 F = |f|·a/C0
		F := math.Abs(freq) * dc.Radius / dc.SoundVelocity

		// 归一化相速度 C = Cp/C0
		C := dc.PochhammerChreeSolver(dc.PoissonRatio, F)
		if C <= 0 {
			continue
		}

		// 相位补偿量 φ = ω·Δz·(1/C0 - 1/Cp) = ω·Δz/C0·(1 - 1/C)
		phi := 2.0 * math.Pi * freq * propagationDist / dc.SoundVelocity * (1.0 - 1.0/C)
		freqSignal[i] *= cmplx.Exp(complex(0, phi))
	}

	// 频域 -> 时域，取前 n 点
	correctedSignal := fft.IFFT(freqSignal)
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = real(correctedSignal[i])
	}
	return result
}

// DispersionCorrectStrains 对三个应变信号进行矫正
func (dc *DispersionCorrector) DispersionCorrectStrains(
	strainInc, strainRef, strainTrans []float64,
	dt float64,
) ([]float64, []float64, []float64) {

	// 对入射波进行矫正 (从应变片到试样入射端面)
	// 使用入射杆的传播距离
	correctedInc := dc.FrequencyCorrection(strainInc, dt, dc.IncPropagationDist)

	// 对反射波进行矫正 (从试样反射端面到应变片)
	// 反射波传播方向相反, 但相位偏移公式相同
	// 使用入射杆的传播距离
	correctedRef := dc.FrequencyCorrection(strainRef, dt, dc.IncPropagationDist)

	// 对透射波进行矫正 (从试样透射端面到应变片)
	// 使用透射杆的传播距离
	correctedTrans := dc.FrequencyCorrection(strainTrans, dt, dc.TransPropagationDist)

	return correctedInc, correctedRef, correctedTrans
}

// NewDispersionCorrector 创建一个新的色散矫正器 (便捷构造函数)
func NewDispersionCorrector(
	soundVelocity, poissonRatio, radius,
	incPropagationDist, transPropagationDist float64,
) *DispersionCorrector {
	return &DispersionCorrector{
		SoundVelocity:        soundVelocity,
		PoissonRatio:         poissonRatio,
		Radius:               radius,
		IncPropagationDist:   incPropagationDist,
		TransPropagationDist: transPropagationDist,
	}
}

// NewDispersionCorrectorWithDefault 使用默认值创建矫正器 (适用于常见钢杆)
func NewDispersionCorrectorWithDefault(
	radius, incPropagationDist, transPropagationDist float64,
) *DispersionCorrector {
	return &DispersionCorrector{
		SoundVelocity:        5190.0, // 钢的典型声速 (m/s)
		PoissonRatio:         0.29,   // 钢的典型泊松比
		Radius:               radius,
		IncPropagationDist:   incPropagationDist,
		TransPropagationDist: transPropagationDist,
	}
}
