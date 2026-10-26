package backend

import "math"

type SignalProcessor struct {
	//杆参数
	Hopkinson HopkinsonBar

	// 样品
	Sample Sample

	//原始信号
	IncWave   ShpbSignal
	TransWave ShpbSignal

	//平滑、基线处理后信号用于显示以及计算
	IncWaveShow   ShpbSignal
	TransWaveShow ShpbSignal

	OffsetINC   float64 //基线处理后的偏移
	OffsetTRANS float64

	// 高频滤波（低通）配置，入射波与透射波分别设置
	IncFilter   FilterConfig
	TransFilter FilterConfig

	//截取的波形
	IncWaveCrop   ShpbSignal
	RefWaveCrop   ShpbSignal
	TransWaveCrop ShpbSignal

	//截取起始点
	CropDataStartAndEnd map[string]*Range

	//计算结果
	Result CalculationResult
}

func NewSignalProcessor() *SignalProcessor {
	return &SignalProcessor{
		Hopkinson: NewDefaultHopkinsonBar(),
		Sample:    NewDefaultSample(),

		CropDataStartAndEnd: make(map[string]*Range),

		IncWave:   ShpbSignal{X: []float64{}, Y: []float64{}},
		TransWave: ShpbSignal{X: []float64{}, Y: []float64{}},

		IncFilter:   FilterConfig{Type: "none"},
		TransFilter: FilterConfig{Type: "none"},
	}
}

// CSV 文本解析为向量
func (sp *SignalProcessor) ParseData(incRaw string, transRaw string) error {
	var err error
	sp.IncWave, err = parseCsvToVector(incRaw)
	sp.IncWaveShow = sp.IncWave

	if err != nil {
		return err
	}

	sp.TransWave, err = parseCsvToVector(transRaw)
	sp.TransWaveShow = sp.TransWave

	if err != nil {
		return err
	}
	return err
}

// 对数据进行预处理
func (sp *SignalProcessor) AlignPeak() []ShpbSignal {
	minLen := len(sp.IncWaveCrop.Y)
	if len(sp.RefWaveCrop.Y) < minLen {
		minLen = len(sp.RefWaveCrop.Y)
	}
	if len(sp.TransWaveCrop.Y) < minLen {
		minLen = len(sp.TransWaveCrop.Y)
	}

	peakInc := FindFirstMajorPeak(&sp.IncWaveCrop, 1.0)
	peakRef := FindFirstMajorPeak(&sp.RefWaveCrop, 1.0)
	peakTrans := FindFirstMajorPeak(&sp.TransWaveCrop, 1.0)

	if peakInc == -1 || peakRef == -1 || peakTrans == -1 {
		return nil
	}

	shiftRef := peakInc - peakRef
	shiftTrans := peakInc - peakTrans

	if sp.CropDataStartAndEnd == nil {
		return nil
	}

	// 入射波
	if _, ok := sp.CropDataStartAndEnd["inc"]; !ok {
		println("!!! 错误: Key 'inc' 不存在")
		return nil
	}
	startInc := sp.CropDataStartAndEnd["inc"].Start
	finalInc := cropSlice(&sp.IncWaveShow, startInc, minLen)

	// 反射波
	if _, ok := sp.CropDataStartAndEnd["ref"]; !ok {
		println("!!! 错误: Key 'ref' 不存在")
		return nil
	}
	startRef := sp.CropDataStartAndEnd["ref"].Start
	finalRef := cropSlice(&sp.IncWaveShow, startRef-shiftRef, minLen)

	// 透射波
	if _, ok := sp.CropDataStartAndEnd["trans"]; !ok {
		println("!!! 错误: Key 'trans' 不存在")
		return nil
	}
	startTrans := sp.CropDataStartAndEnd["trans"].Start

	finalTrans := cropSlice(&sp.TransWaveShow, startTrans-shiftTrans, minLen)

	sp.CropDataStartAndEnd["inc"].End = startInc + minLen
	sp.CropDataStartAndEnd["ref"].Start = startRef - shiftRef
	sp.CropDataStartAndEnd["ref"].End = startRef - shiftRef + minLen
	sp.CropDataStartAndEnd["trans"].Start = startTrans - shiftTrans
	sp.CropDataStartAndEnd["trans"].End = startTrans - shiftTrans + minLen

	return []ShpbSignal{finalInc, finalRef, finalTrans}
}

func (sp *SignalProcessor) AlignGB() []ShpbSignal {
	if sp.CropDataStartAndEnd == nil {
		return nil
	}
	if _, ok := sp.CropDataStartAndEnd["inc"]; !ok {
		return nil
	}
	if _, ok := sp.CropDataStartAndEnd["ref"]; !ok {
		return nil
	}
	if _, ok := sp.CropDataStartAndEnd["trans"]; !ok {
		return nil
	}

	peakInc := FindFirstMajorPeak(&sp.IncWaveCrop, 1.0)
	peakRef := FindFirstMajorPeak(&sp.RefWaveCrop, 1.0)
	peakTrans := FindFirstMajorPeak(&sp.TransWaveCrop, 1.0)
	if peakInc == -1 || peakRef == -1 || peakTrans == -1 {
		return nil
	}

	baseInc := findGBBaseIndex(sp.IncWaveCrop.Y, peakInc)
	baseRef := findGBBaseIndex(sp.RefWaveCrop.Y, peakRef)
	baseTrans := findGBBaseIndex(sp.TransWaveCrop.Y, peakTrans)
	if baseInc == -1 || baseRef == -1 || baseTrans == -1 {
		return nil
	}

	minLen := len(sp.IncWaveCrop.Y)
	if len(sp.RefWaveCrop.Y) < minLen {
		minLen = len(sp.RefWaveCrop.Y)
	}
	if len(sp.TransWaveCrop.Y) < minLen {
		minLen = len(sp.TransWaveCrop.Y)
	}

	shiftRef := baseInc - baseRef
	shiftTrans := baseInc - baseTrans

	if sp.CropDataStartAndEnd == nil {
		return nil
	}

	// 入射波
	if _, ok := sp.CropDataStartAndEnd["inc"]; !ok {
		println("!!! 错误: Key 'inc' 不存在")
		return nil
	}
	startInc := sp.CropDataStartAndEnd["inc"].Start
	finalInc := cropSlice(&sp.IncWaveShow, startInc, minLen)

	// 反射波
	if _, ok := sp.CropDataStartAndEnd["ref"]; !ok {
		println("!!! 错误: Key 'ref' 不存在")
		return nil
	}
	startRef := sp.CropDataStartAndEnd["ref"].Start
	finalRef := cropSlice(&sp.IncWaveShow, startRef-shiftRef, minLen)

	// 透射波
	if _, ok := sp.CropDataStartAndEnd["trans"]; !ok {
		println("!!! 错误: Key 'trans' 不存在")
		return nil
	}
	startTrans := sp.CropDataStartAndEnd["trans"].Start

	finalTrans := cropSlice(&sp.TransWaveShow, startTrans-shiftTrans, minLen)

	sp.CropDataStartAndEnd["inc"].End = startInc + minLen
	sp.CropDataStartAndEnd["ref"].Start = startRef - shiftRef
	sp.CropDataStartAndEnd["ref"].End = startRef - shiftRef + minLen
	sp.CropDataStartAndEnd["trans"].Start = startTrans - shiftTrans
	sp.CropDataStartAndEnd["trans"].End = startTrans - shiftTrans + minLen

	return []ShpbSignal{finalInc, finalRef, finalTrans}
}

func (sp *SignalProcessor) AlignTime() []ShpbSignal {
	minLen := len(sp.IncWaveCrop.Y)
	if len(sp.RefWaveCrop.Y) < minLen {
		minLen = len(sp.RefWaveCrop.Y)
	}
	if len(sp.TransWaveCrop.Y) < minLen {
		minLen = len(sp.TransWaveCrop.Y)
	}

	startInc := sp.CropDataStartAndEnd["inc"].Start
	finalInc := cropSlice(&sp.IncWaveShow, startInc, minLen)

	startTrans := sp.CropDataStartAndEnd["trans"].Start
	finalTrans := cropSlice(&sp.TransWaveShow, startTrans, minLen)

	startRef := sp.CropDataStartAndEnd["ref"].Start
	finalRef := cropSlice(&sp.IncWaveShow, startRef, minLen)

	sp.CropDataStartAndEnd["inc"].Start = startInc
	sp.CropDataStartAndEnd["trans"].Start = startTrans
	sp.CropDataStartAndEnd["ref"].Start = startRef
	sp.CropDataStartAndEnd["inc"].End = startInc + minLen
	sp.CropDataStartAndEnd["trans"].End = startTrans + minLen
	sp.CropDataStartAndEnd["ref"].End = startRef + minLen

	return []ShpbSignal{finalInc, finalRef, finalTrans}
}

func (sp *SignalProcessor) AlignManul(data map[string]float64) []ShpbSignal {
	minLen := len(sp.IncWaveCrop.Y)
	if len(sp.RefWaveCrop.Y) < minLen {
		minLen = len(sp.RefWaveCrop.Y)
	}
	if len(sp.TransWaveCrop.Y) < minLen {
		minLen = len(sp.TransWaveCrop.Y)
	}

	startInc := sp.CropDataStartAndEnd["inc"].Start - int(data["inc"])
	finalInc := cropSlice(&sp.IncWaveShow, startInc, minLen)

	startTrans := sp.CropDataStartAndEnd["trans"].Start - int(data["trans"])
	finalTrans := cropSlice(&sp.TransWaveShow, startTrans, minLen)

	startRef := sp.CropDataStartAndEnd["ref"].Start - int(data["ref"])
	finalRef := cropSlice(&sp.IncWaveShow, startRef, minLen)

	sp.CropDataStartAndEnd["inc"].Start = startInc
	sp.CropDataStartAndEnd["trans"].Start = startTrans
	sp.CropDataStartAndEnd["ref"].Start = startRef
	sp.CropDataStartAndEnd["inc"].End = startInc + minLen
	sp.CropDataStartAndEnd["trans"].End = startTrans + minLen
	sp.CropDataStartAndEnd["ref"].End = startRef + minLen

	return []ShpbSignal{finalInc, finalRef, finalTrans}
}

type CalculationResult struct {
	StressStrain   ShpbSignal `json:"stress_strain"`    // 工程应力-工程应变
	StrainTime     ShpbSignal `json:"strain_time"`      // 工程应变-时间
	StrainRateTime ShpbSignal `json:"strain_rate_time"` // 工程应变率-时间
	StressTime     ShpbSignal `json:"stress_time"`      // 工程应力-时间

	TrueStressStrain   ShpbSignal `json:"true_stress_strain"`    // 真应力-真应变
	TrueStrainTime     ShpbSignal `json:"true_strain_time"`      // 真应变-时间
	TrueStrainRateTime ShpbSignal `json:"true_strain_rate_time"` // 真应变率-时间
	TrueStressTime     ShpbSignal `json:"true_stress_time"`      // 真应力-时间

	IncWaveCrop   ShpbSignal `json:"inc_wave_crop"`   // 入射波-裁剪
	RefWaveCrop   ShpbSignal `json:"ref_wave_crop"`   // 反射波-裁剪
	TransWaveCrop ShpbSignal `json:"trans_wave_crop"` // 透射波-裁剪
}

// Calculate 按 MATLAB(PHop) 版本的核心算法计算
//   σ = E·(A_bar/A_sample)·ε_t
//   ε̇ = 2C0/L·(ε_i-ε_t)   (双波法)
//   ε = Σ dt·ε̇            (右端点欧拉积分)
func (sp *SignalProcessor) Calculate(calculationType string) CalculationResult {
	compression := 1.0
	if sp.Hopkinson.Mode == "compression" {
		compression = -1.0
	}

	// 截取波形的数据（使用对齐后的 Crop 数据）
	wInc := sp.IncWaveCrop
	wRef := sp.RefWaveCrop
	wTrans := sp.TransWaveCrop

	// 三路波形长度可能不完全一致，取最短长度避免越界
	n := len(wInc.Y)
	if len(wRef.Y) < n {
		n = len(wRef.Y)
	}
	if len(wTrans.Y) < n {
		n = len(wTrans.Y)
	}
	if n == 0 {
		return CalculationResult{}
	}

	// 采样间隔 dt
	dt := 1.0
	if len(wInc.X) > 1 {
		dt = wInc.X[1] - wInc.X[0]
	}

	// 标定系数已包含桥路/增益等全部硬件因素，这里直接乘电压得到应变，
	// 与 MATLAB 的 stc1/stc2 用法一致，不再按桥型做二次除法。
	incidentCoeff := sp.Hopkinson.IncidentCoefficient
	transmittedCoeff := sp.Hopkinson.TransmittedCoefficient

	strainInc := make([]float64, n)
	strainRef := make([]float64, n)
	strainTrans := make([]float64, n)

	for i := 0; i < n; i++ {
		strainInc[i] = wInc.Y[i] * incidentCoeff * compression
		strainRef[i] = wRef.Y[i] * incidentCoeff * compression
		strainTrans[i] = wTrans.Y[i] * transmittedCoeff * compression
	}

	// 色散修正必须在应变换算完成之后进行
	if sp.Hopkinson.EnableDispersionCorrection {
		corrector := &DispersionCorrector{
			SoundVelocity:        sp.Hopkinson.SoundVelocity,
			PoissonRatio:         sp.Hopkinson.PoissonRatio,
			Radius:               sp.Hopkinson.Diameter / 2.0 / 1000.0,
			IncPropagationDist:   sp.Hopkinson.IncidentGaugeToEndDistance / 1000.0,
			TransPropagationDist: sp.Hopkinson.TransmitGaugeToEndDistance / 1000.0,
		}
		strainInc, strainRef, strainTrans = corrector.DispersionCorrectStrains(
			strainInc, strainRef, strainTrans, dt,
		)
	}

	// 试样参数：截面积由前端计算并传入
	sampleLengthM := sp.Sample.Length / 1000.0 // mm -> m
	barArea := math.Pi * math.Pow(sp.Hopkinson.Diameter/2.0, 2.0)
	areaRatio := 0.0
	if sp.Sample.Area > 0 {
		areaRatio = barArea / sp.Sample.Area
	}

	// 常数准备
	E := sp.Hopkinson.YoungSPa * 1e9
	C := sp.Hopkinson.SoundVelocity

	engStressMpa := make([]float64, n)
	engStrainRate := make([]float64, n)

	switch calculationType {
	case "incAndTrans": // 入射+透射 (两波法，MATLAB 默认算法)
		for i := 0; i < n; i++ {
			engStressMpa[i] = E * areaRatio * strainTrans[i] / 1e6
			engStrainRate[i] = (2 * C / sampleLengthM) * (strainInc[i] - strainTrans[i])
		}

	case "refAndTrans": // 反射+透射
		for i := 0; i < n; i++ {
			engStressMpa[i] = E * areaRatio * strainTrans[i] / 1e6
			engStrainRate[i] = -(2 * C / sampleLengthM) * strainRef[i]
		}

	case "threeWave": // 三波法
		for i := 0; i < n; i++ {
			engStressMpa[i] = 0.5 * E * areaRatio * (strainInc[i] + strainTrans[i] + strainRef[i]) / 1e6
			engStrainRate[i] = (C / sampleLengthM) * (strainInc[i] - strainRef[i] - strainTrans[i])
		}

	default:
		return CalculationResult{}
	}

	// 工程应变：MATLAB 使用右端点欧拉积分
	engStrain := CumulativeEuler(engStrainRate, dt)

	// 真应力/真应变：MATLAB 由未平滑的工程量计算
	trueStrain := make([]float64, n)
	trueStrainRate := make([]float64, n)
	trueStressMpa := make([]float64, n)
	for i := 0; i < n; i++ {
		ratio := 1.0 + engStrain[i]
		if ratio < 1e-6 {
			ratio = 1e-6
		}
		trueStrain[i] = math.Log(ratio)
		trueStressMpa[i] = engStressMpa[i] * ratio
		trueStrainRate[i] = engStrainRate[i] / ratio
	}

	// 结果平滑：对应 MATLAB 的 smooth(x, n/spanfactor, 'sgolay')，
	// spanfactor 默认为 10，MATLAB 会把偶数窗宽 +1 调整为奇数。
	// MATLAB 对工程量与真值共 6 条曲线分别独立平滑。
	const resultSpanFactor = 10
	span := n / resultSpanFactor
	if span%2 == 0 {
		span++
	}
	if span >= 5 {
		engStrain = SavgolSmooth(engStrain, span, 2)
		engStrainRate = SavgolSmooth(engStrainRate, span, 2)
		engStressMpa = SavgolSmooth(engStressMpa, span, 2)
		trueStrain = SavgolSmooth(trueStrain, span, 2)
		trueStrainRate = SavgolSmooth(trueStrainRate, span, 2)
		trueStressMpa = SavgolSmooth(trueStressMpa, span, 2)
	}

	timeAxis := wInc.X[:n]

	sp.Result = CalculationResult{
		StrainTime:         ShpbSignal{X: timeAxis, Y: engStrain},
		StrainRateTime:     ShpbSignal{X: timeAxis, Y: engStrainRate},
		StressTime:         ShpbSignal{X: timeAxis, Y: engStressMpa},
		StressStrain:       ShpbSignal{X: engStrain, Y: engStressMpa},
		TrueStrainTime:     ShpbSignal{X: timeAxis, Y: trueStrain},
		TrueStrainRateTime: ShpbSignal{X: timeAxis, Y: trueStrainRate},
		TrueStressTime:     ShpbSignal{X: timeAxis, Y: trueStressMpa},
		TrueStressStrain:   ShpbSignal{X: trueStrain, Y: trueStressMpa},
		IncWaveCrop:        sp.IncWaveCrop,
		RefWaveCrop:        sp.RefWaveCrop,
		TransWaveCrop:      sp.TransWaveCrop,
	}
	return sp.Result
}
