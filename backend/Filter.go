package backend

import "math"

// FilterConfig 高频滤波（低通）配置
type FilterConfig struct {
	Type   string  `json:"Type"`   // none / butterworth / chebyshev / fir
	Cutoff float64 `json:"Cutoff"` // 高频截止频率 (Hz)
}

// ApplyLowPass 按配置对信号做零相位低通滤波。
// 采用零相位实现（IIR 前后向各滤波一次、FIR 对称核），避免相位滞后影响对齐与计算。
func ApplyLowPass(signal ShpbSignal, cfg FilterConfig) ShpbSignal {
	if cfg.Type == "" || cfg.Type == "none" || cfg.Cutoff <= 0 || len(signal.Y) < 3 {
		return signal
	}
	dt := averageDt(signal.X)
	if dt <= 0 {
		return signal
	}
	fs := 1.0 / dt
	if cfg.Cutoff >= fs/2.0 {
		return signal
	}

	var y []float64
	switch cfg.Type {
	case "butterworth":
		y = zeroPhaseIIR(signal.Y, butterworthSections(cfg.Cutoff, fs))
	case "chebyshev":
		y = zeroPhaseIIR(signal.Y, chebyshevSections(cfg.Cutoff, fs, 0.5))
	case "fir":
		y = firHammingLowPass(signal.Y, cfg.Cutoff, fs)
	default:
		return signal
	}

	return ShpbSignal{X: signal.X, Y: y}
}

// averageDt 取采样间隔（与 MATLAB 相同，用总时长/点数）
func averageDt(x []float64) float64 {
	n := len(x)
	if n < 2 {
		return 0
	}
	return (x[n-1] - x[0]) / float64(n-1)
}

// ---------------------------------------------------------------
// IIR 低通设计：4 阶巴特沃斯 / 切比雪夫 I 型（双线性变换 + 预畸变）
// ---------------------------------------------------------------

const filterOrder = 4

// butterworthSections 返回 4 阶巴特沃斯低通的二阶节系数 [b0,b1,b2,a1,a2]
func butterworthSections(fc, fs float64) [][5]float64 {
	poles := make([]complex128, 0, filterOrder/2)
	for k := 0; k < filterOrder; k++ {
		theta := math.Pi * float64(2*k+filterOrder+1) / float64(2*filterOrder)
		p := complex(math.Cos(theta), math.Sin(theta))
		if imag(p) > 0 {
			poles = append(poles, p)
		}
	}
	return analogPolesToSections(poles, fc, fs)
}

// chebyshevSections 返回 4 阶切比雪夫 I 型低通的二阶节系数
func chebyshevSections(fc, fs, rippleDB float64) [][5]float64 {
	eps := math.Sqrt(math.Pow(10, rippleDB/10) - 1)
	a := math.Asinh(1/eps) / float64(filterOrder)
	poles := make([]complex128, 0, filterOrder/2)
	for k := 0; k < filterOrder; k++ {
		theta := math.Pi * float64(2*k+1) / float64(2*filterOrder)
		p := complex(-math.Sinh(a)*math.Sin(theta), math.Cosh(a)*math.Cos(theta))
		if imag(p) > 0 {
			poles = append(poles, p)
		}
	}
	return analogPolesToSections(poles, fc, fs)
}

// analogPolesToSections 将归一化模拟极点映射为数字二阶节
func analogPolesToSections(poles []complex128, fc, fs float64) [][5]float64 {
	// 双线性变换预畸变系数
	k := math.Tan(math.Pi * fc / fs)
	sections := make([][5]float64, 0, len(poles))
	for _, p := range poles {
		zp := (1 + complex(k, 0)*p) / (1 - complex(k, 0)*p)
		re, im := real(zp), imag(zp)
		mag2 := re*re + im*im
		// 分子 (1+z^-1)^2，分母由共轭极点对生成
		sections = append(sections, [5]float64{1, 2, 1, -2 * re, mag2})
	}

	// 归一化使直流增益为 1
	gain := 1.0
	for _, s := range sections {
		gain *= (s[0] + s[1] + s[2]) / (1 + s[3] + s[4])
	}
	if gain != 0 && len(sections) > 0 {
		// 整体增益只加在第一个二阶节上
		sections[0][0] /= gain
		sections[0][1] /= gain
		sections[0][2] /= gain
	}
	return sections
}

// zeroPhaseIIR 对 IIR 滤波器做前向+反向两次滤波，得到零相位结果
func zeroPhaseIIR(x []float64, sections [][5]float64) []float64 {
	n := len(x)
	if n < 3 || len(sections) == 0 {
		return x
	}

	// 反射延拓，减小端点瞬态
	pad := 6 * len(sections)
	if pad > n-1 {
		pad = n - 1
	}
	ext := make([]float64, n+2*pad)
	for i := 0; i < pad; i++ {
		ext[i] = 2*x[0] - x[pad-i]
		ext[pad+n+i] = 2*x[n-1] - x[n-2-i]
	}
	copy(ext[pad:pad+n], x)

	y := filterForward(ext, sections)
	reverseInPlace(y)
	y = filterForward(y, sections)
	reverseInPlace(y)

	return y[pad : pad+n]
}

// filterForward 二阶节级联的 DF2T 前向滤波
func filterForward(x []float64, sections [][5]float64) []float64 {
	y := make([]float64, len(x))
	copy(y, x)

	z1 := make([]float64, len(sections))
	z2 := make([]float64, len(sections))

	for s, sec := range sections {
		b0, b1, b2, a1, a2 := sec[0], sec[1], sec[2], sec[3], sec[4]
		for i := range y {
			in := y[i]
			out := b0*in + z1[s]
			z1[s] = b1*in - a1*out + z2[s]
			z2[s] = b2*in - a2*out
			y[i] = out
		}
	}
	return y
}

func reverseInPlace(x []float64) {
	for i, j := 0, len(x)-1; i < j; i, j = i+1, j-1 {
		x[i], x[j] = x[j], x[i]
	}
}

// ---------------------------------------------------------------
// FIR 低通：汉明窗（线性相位、对称核，天然零相位）
// ---------------------------------------------------------------

func firHammingLowPass(y []float64, fc, fs float64) []float64 {
	n := len(y)
	// 过渡带越窄需要的阶数越高，这里按 4 个采样周期估计
	taps := int(math.Round(4.0 * fs / fc))
	if taps%2 == 0 {
		taps++
	}
	if taps < 11 {
		taps = 11
	}
	if taps > 2001 {
		taps = 2001
	}

	m := taps / 2
	kernel := make([]float64, taps)
	normFreq := 2.0 * fc / fs
	sum := 0.0
	for j := 0; j < taps; j++ {
		t := float64(j - m)
		var sinc float64
		if t == 0 {
			sinc = normFreq
		} else {
			sinc = math.Sin(math.Pi*normFreq*t) / (math.Pi * t)
		}
		window := 0.54 - 0.46*math.Cos(2*math.Pi*float64(j)/float64(taps-1))
		kernel[j] = sinc * window
		sum += kernel[j]
	}
	if sum == 0 {
		return y
	}
	for j := range kernel {
		kernel[j] /= sum
	}

	out := make([]float64, n)
	for i := 0; i < n; i++ {
		val := 0.0
		for j := 0; j < taps; j++ {
			idx := i + j - m
			// 镜像延拓，避免边界突变
			if idx < 0 {
				idx = -idx - 1
			}
			if idx >= n {
				idx = 2*n - idx - 1
			}
			if idx < 0 {
				idx = 0
			}
			if idx >= n {
				idx = n - 1
			}
			val += kernel[j] * y[idx]
		}
		out[i] = val
	}
	return out
}
