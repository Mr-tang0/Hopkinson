package backend

import (
	"math"
	"math/cmplx"
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
// propagationDist: 应变片到目标端面的距离 (m)
func (dc *DispersionCorrector) FrequencyCorrection(timeSignal []float64, dt float64, propagationDist float64) []float64 {
	n := len(timeSignal)
	if n == 0 {
		return timeSignal
	}

	// Step 1: FFT - 将时域信号转换到频域
	complexSignal := make([]complex128, n)
	for i := 0; i < n; i++ {
		complexSignal[i] = complex(timeSignal[i], 0)
	}

	// 执行FFT
	freqSignal := dc.fft(complexSignal)

	// 频率分辨率
	df := 1.0 / (float64(n) * dt)

	// Step 2-4: 对每个频率分量进行相位矫正
	for i := 0; i < n; i++ {
		freq := float64(i) * df

		// 只处理正频率分量 (i < n/2)
		if i > n/2 {
			// 负频率分量: 共轭对称, 自动处理
			continue
		}

		// 归一化频率 F = f*a/C₀
		F := freq * dc.Radius / dc.SoundVelocity

		// 计算归一化相速度 C = Cp/C₀
		C := dc.PochhammerChreeSolver(dc.PoissonRatio, F)

		// 如果C接近1, 不需要矫正
		if math.Abs(C-1.0) < 1e-6 {
			continue
		}

		// 计算相位偏移: φ' = (C₀/Cp - 1) * (2πfΔz/C₀)
		// 即: φ' = (1/C - 1) * (2πfΔz/C₀)
		phiPrime := (1.0/C - 1.0) * (2.0 * math.Pi * freq * propagationDist / dc.SoundVelocity)

		// 矫正相位: φ_corr = φ - φ'
		currentPhase := cmplx.Phase(freqSignal[i])
		currentMag := cmplx.Abs(freqSignal[i])

		// 应用相位矫正 (对于正频率)
		correctedPhase := currentPhase - phiPrime
		freqSignal[i] = cmplx.Rect(currentMag, correctedPhase)

		// 对于负频率, 取共轭以保持信号为实数
		if i > 0 && i < n/2 {
			j := n - i
			freqSignal[j] = cmplx.Conj(freqSignal[i])
		}
	}

	// Step 5: IFFT - 将矫正后的频域信号转换回时域
	correctedSignal := dc.ifft(freqSignal)

	// 提取实部
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = real(correctedSignal[i])
	}

	return result
}

// fft 快速傅里叶变换 (Cooley-Tukey迭代实现)
// 要求 n 必须是2的幂
func (dc *DispersionCorrector) fft(x []complex128) []complex128 {
	n := len(x)

	// 如果n是0或1, 直接返回
	if n <= 1 {
		return x
	}

	// 检查n是否为2的幂, 如果不是则补零
	actualN := dc.nextPowerOfTwo(n)
	if actualN != n {
		// 补零到 actualN
		padded := make([]complex128, actualN)
		copy(padded, x)
		x = padded
		n = actualN
	}

	// 位反转置换 (迭代实现)
	result := make([]complex128, n)
	copy(result, x)

	// 位反转
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for j&bit != 0 {
			j ^= bit
			bit >>= 1
		}
		j ^= bit
		if i < j {
			result[i], result[j] = result[j], result[i]
		}
	}

	// 迭代蝶形运算
	length := 2
	for length <= n {
		// 计算旋转因子
		theta := -2.0 * math.Pi / float64(length)
		wlen := cmplx.Exp(complex(0, theta))

		for i := 0; i < n; i += length {
			w := complex(1, 0)
			half := length >> 1
			for j := 0; j < half; j++ {
				u := result[i+j]
				v := result[i+j+half] * w
				result[i+j] = u + v
				result[i+j+half] = u - v
				w *= wlen
			}
		}
		length <<= 1
	}

	return result
}

// ifft 逆快速傅里叶变换
func (dc *DispersionCorrector) ifft(x []complex128) []complex128 {
	n := len(x)

	// 取共轭, 执行FFT, 再取共轭并除以n
	conj := make([]complex128, n)
	for i := 0; i < n; i++ {
		conj[i] = cmplx.Conj(x[i])
	}

	fftResult := dc.fft(conj)

	result := make([]complex128, n)
	for i := 0; i < n; i++ {
		result[i] = cmplx.Conj(fftResult[i]) / complex(float64(n), 0)
	}

	return result
}

// nextPowerOfTwo 返回大于等于n的最小2的幂
func (dc *DispersionCorrector) nextPowerOfTwo(n int) int {
	if n <= 0 {
		return 1
	}
	// 如果n已经是2的幂, 直接返回
	if n&(n-1) == 0 {
		return n
	}
	// 否则找到下一个2的幂
	power := 1
	for power < n {
		power <<= 1
	}
	return power
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
