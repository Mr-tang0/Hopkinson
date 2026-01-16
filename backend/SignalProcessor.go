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

	LowPassFreq  float64 //滤波频率
	HighPassFreq float64

	//频谱信号
	FFTInc   ShpbSignal
	FFTTrans ShpbSignal

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

		LowPassFreq:  0,
		HighPassFreq: 0,
		IncWave:      ShpbSignal{X: []float64{}, Y: []float64{}},
		TransWave:    ShpbSignal{X: []float64{}, Y: []float64{}},
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

	return sp.AlignPeak()
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
	StressStrain   ShpbSignal `json:"stress_strain"`    // 应力-应变
	StrainTime     ShpbSignal `json:"strain_time"`      // 应变-时间
	StrainRateTime ShpbSignal `json:"strain_rate_time"` // 应变率-时间
	StressTime     ShpbSignal `json:"stress_time"`      // 应力-时间
}

func (sp *SignalProcessor) Calculate(calculationType string) CalculationResult {
	compression := 1.0
	if sp.Hopkinson.Mode == "compression" {
		compression = -1.0
	}

	// 截取波形的数据（使用对齐后的 Crop 数据）
	wInc := sp.IncWaveCrop
	wRef := sp.RefWaveCrop
	wTrans := sp.TransWaveCrop

	n := len(wInc.Y)
	if n == 0 {
		return CalculationResult{}
	}

	// 采样间隔 dt
	dt := 1.0
	if len(wInc.X) > 1 {
		dt = wInc.X[1] - wInc.X[0]
	}

	// 应变片产生的应变 (直接使用标定系数)
	// strain = V * coeff * mode_sign
	strainInc := make([]float64, n)
	strainRef := make([]float64, n)
	strainTrans := make([]float64, n)

	coeff := sp.Hopkinson.Coefficient
	if sp.Hopkinson.BridgeType == "HALF" {
		coeff = coeff / 2.0
	} else if sp.Hopkinson.BridgeType == "QURT" {
		coeff = coeff / 4.0
	}

	for i := 0; i < n; i++ {
		strainInc[i] = wInc.Y[i] * coeff * compression
		strainRef[i] = wRef.Y[i] * coeff * compression
		strainTrans[i] = wTrans.Y[i] * coeff * compression
	}

	// 试样参数
	sampleLengthM := sp.Sample.Length / 1000.0 // mm -> m
	// 面积比 = (杆直径 / 试样直径)^2
	areaRatio := math.Pow(sp.Hopkinson.Diameter/sp.Sample.Diameter, 2)

	// 常数准备
	E := sp.Hopkinson.YoungSPa * 1e9
	C := sp.Hopkinson.SoundVelocity

	engStressMpa := make([]float64, n)
	engStrainRate := make([]float64, n)

	switch calculationType {
	case "incAndTrans": // 入射+透射 (两波法)
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
	}

	// 计算工程应变 (速率积分)
	engStrain := CumTrapz(engStrainRate, dt)

	// 结果
	sp.Result = CalculationResult{
		StrainTime:     ShpbSignal{X: wInc.X, Y: engStrain},
		StrainRateTime: ShpbSignal{X: wInc.X, Y: engStrainRate},
		StressTime:     ShpbSignal{X: wInc.X, Y: engStressMpa},
		StressStrain:   ShpbSignal{X: engStrain, Y: engStressMpa},
	}
	return sp.Result
}
