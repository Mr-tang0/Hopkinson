package backend

type HopkinsonBar struct {
	Type           string  `json:"Type"`
	Mode           string  `json:"Mode"`
	Material       string  `json:"Material"`
	Diameter       float64 `json:"Diameter"`
	YoungSPa       float64 `json:"YoungSPa"`
	SoundVelocity  float64 `json:"SoundVelocity"`
	BridgeType     string  `json:"BridgeType"`
	GageFactor     float64 `json:"GageFactor"`
	BridgeTensionV float64 `json:"BridgeTensionV"`
	Coefficient    float64 `json:"Coefficient"`
	FirstLength    float64 `json:"FirstLength"`
	SecondLength   float64 `json:"SecondLength"`
	PoissonRatio   float64 `json:"PoissonRatio"`
	Damping        float64 `json:"Damping"`
}

func NewDefaultHopkinsonBar() HopkinsonBar {
	return HopkinsonBar{
		Type:           "ALT7075",
		Mode:           "compression",
		Material:       "钢",
		Diameter:       20,           // 19mm 常用规格
		YoungSPa:       210,          // 210 GPa
		SoundVelocity:  5200.0,       // 5000 m/s
		BridgeType:     "HalfBridge", // 全桥
		GageFactor:     1.8,
		BridgeTensionV: 10,
		Coefficient:    0.0111111, // 默认标定系数
		PoissonRatio:   0.25,
	}
}

type Sample struct {
	Number   string  `json:"Number"`
	Name     string  `json:"Name"`
	Material string  `json:"Material"`
	Diameter float64 `json:"Diameter"`
	Length   float64 `json:"Length"`
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
