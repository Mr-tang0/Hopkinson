package backend

type HopkinsonBar struct {
	Type                       string  `json:"Type"`
	Mode                       string  `json:"Mode"`
	Material                   string  `json:"Material"`
	Diameter                   float64 `json:"Diameter"`
	YoungSPa                   float64 `json:"YoungSPa"`
	SoundVelocity              float64 `json:"SoundVelocity"`
	IncidentCoefficient        float64 `json:"IncidentCoefficient"`
	TransmittedCoefficient     float64 `json:"TransmittedCoefficient"`
	EnableDispersionCorrection bool    `json:"EnableDispersionCorrection"`
	PoissonRatio               float64 `json:"PoissonRatio"`
	IncidentGaugeToEndDistance float64 `json:"IncidentGaugeToEndDistance"`
	TransmitGaugeToEndDistance float64 `json:"TransmitGaugeToEndDistance"`
}

func NewDefaultHopkinsonBar() HopkinsonBar {
	return HopkinsonBar{
		Type:                       "ALT7075",
		Mode:                       "compression",
		Material:                   "钢",
		Diameter:                   20,        // 19mm 常用规格
		YoungSPa:                   210,       // 210 GPa
		SoundVelocity:              5200.0,    // 5000 m/s
		IncidentCoefficient:        0.0111111, // 默认入射杆标定系数
		TransmittedCoefficient:     0.0111111, // 默认透射杆标定系数
		EnableDispersionCorrection: false,
		PoissonRatio:               0.33,
		IncidentGaugeToEndDistance: 0,
		TransmitGaugeToEndDistance: 0,
	}
}

type Sample struct {
	Number   string  `json:"Number"`
	Name     string  `json:"Name"`
	Material string  `json:"Material"`
	Diameter float64 `json:"Diameter"`
	Length   float64 `json:"Length"`
	// Area 试样实际截面积 (mm²)，对应 MATLAB 版本的 as。
	// 为 0 时退化为按直径计算的圆面积。
	Area float64 `json:"Area"`
}

// NewDefaultSample 创建默认样品参数
func NewDefaultSample() Sample {
	return Sample{
		Number:   "S001",
		Name:     "待测样品",
		Material: "待测材料",
		Diameter: 6,
		Length:   4.5,
	}
}

type Range struct {
	Start int
	End   int
}
