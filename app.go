package main

import (
	"Hopkinson_Decoder/backend"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/wailsapp/wails/v2/pkg/runtime"
	"github.com/xuri/excelize/v2"
)

// App struct
type App struct {
	ctx           context.Context
	hopBar        *backend.SignalProcessor
	updateService *backend.UpdateService
}

type APIResponse struct {
	Status  int    `json:"status"`
	Message string `json:"message"`
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// startup is called when the app starts. The context is saved
// so we can call the runtime methods
func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	a.hopBar = backend.NewSignalProcessor()
	a.updateService = &backend.UpdateService{}
}

// 定义配置文件的总结构
type ConfigData struct {
	HopkinsonList map[string]backend.HopkinsonBar `json:"hopkinsonList"`
	Sample        backend.Sample                  `json:"sample"`
}

func (a *App) GetLatestRelease() (backend.UpdateResult, error) {
	return a.updateService.GetUpdateInfo()
}

func (a *App) GetConfigPath() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		// 如果获取失败，降级使用当前目录
		return "config.json"
	}
	configDir := filepath.Join(homeDir, "Tang", "HopkinsonDecoder")

	_ = os.MkdirAll(configDir, 0755)
	return filepath.Join(configDir, "config.json")
}

// LoadConfig 读取并解析配置
func (a *App) LoadConfig() (ConfigData, error) {
	configPath := a.GetConfigPath()

	// 读取文件
	data, err := os.ReadFile(configPath)

	// 如果文件不存在，则返回一个空的配置结构体
	if err != nil {
		return ConfigData{
			HopkinsonList: make(map[string]backend.HopkinsonBar),
			Sample:        backend.NewDefaultSample(),
		}, nil
	}

	// 解析JSON数据

	var raw struct {
		HopkinsonList map[string]map[string]string `json:"hopkinsonList"`
		Sample        map[string]string            `json:"sample"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		println("解析JSON数据失败")
		return ConfigData{}, err
	}

	println("解析JSON数据成功")

	res := ConfigData{
		HopkinsonList: make(map[string]backend.HopkinsonBar),
		Sample:        backend.NewDefaultSample(),
	}

	for name, m := range raw.HopkinsonList {
		bar := backend.HopkinsonBar{
			Type:       name,
			Mode:       m["mode"],
			Material:   m["material"],
			BridgeType: m["bridgeType"],
		}
		// 转换数值型字段
		bar.Diameter, _ = strconv.ParseFloat(m["diameter"], 64)
		bar.YoungSPa, _ = strconv.ParseFloat(m["youngs"], 64)
		bar.SoundVelocity, _ = strconv.ParseFloat(m["soundVelocity"], 64)
		bar.IncidentCoefficient, _ = strconv.ParseFloat(m["incidentCalibrationFactor"], 64)
		bar.TransmittedCoefficient, _ = strconv.ParseFloat(m["transmittedCalibrationFactor"], 64)
		if bar.IncidentCoefficient == 0 && m["calibrationFactor"] != "" {
			bar.IncidentCoefficient, _ = strconv.ParseFloat(m["calibrationFactor"], 64)
		}
		if bar.TransmittedCoefficient == 0 && m["calibrationFactor"] != "" {
			bar.TransmittedCoefficient, _ = strconv.ParseFloat(m["calibrationFactor"], 64)
		}

		res.HopkinsonList[name] = bar
	}

	res.Sample.Name = raw.Sample["name"]
	res.Sample.Number = raw.Sample["number"]
	res.Sample.Material = raw.Sample["material"]
	res.Sample.Diameter, _ = strconv.ParseFloat(raw.Sample["diameter"], 64)
	res.Sample.Length, _ = strconv.ParseFloat(raw.Sample["length"], 64)

	return res, nil
}

func (a *App) SaveBarConfig(name string, bar backend.HopkinsonBar) (bool, error) {
	var fullConfig map[string]interface{}
	data, err := os.ReadFile(a.GetConfigPath())
	if err == nil {
		json.Unmarshal(data, &fullConfig)
	}

	if fullConfig == nil {
		fullConfig = make(map[string]interface{})
	}

	if _, ok := fullConfig["hopkinsonList"]; !ok {
		fullConfig["hopkinsonList"] = make(map[string]interface{})
	}

	strBar := map[string]string{
		"mode":                         bar.Mode,
		"material":                     bar.Material,
		"diameter":                     fmt.Sprintf("%v", bar.Diameter),
		"youngs":                       fmt.Sprintf("%v", bar.YoungSPa),
		"soundVelocity":                fmt.Sprintf("%v", bar.SoundVelocity),
		"bridgeType":                   bar.BridgeType,
		"incidentCalibrationFactor":    fmt.Sprintf("%v", bar.IncidentCoefficient),
		"transmittedCalibrationFactor": fmt.Sprintf("%v", bar.TransmittedCoefficient),
	}

	list := fullConfig["hopkinsonList"].(map[string]interface{})
	list[name] = strBar

	finalData, err := json.MarshalIndent(fullConfig, "", "    ")
	if err != nil {
		return false, err
	}

	err = os.WriteFile(a.GetConfigPath(), finalData, 0644)
	return err == nil, err
}

func (a *App) DeleteBarConfig(name string) (bool, error) {
	var fullConfig map[string]interface{}
	data, err := os.ReadFile(a.GetConfigPath())
	if err != nil {
		return false, err
	}

	if err := json.Unmarshal(data, &fullConfig); err != nil {
		return false, err
	}

	list, ok := fullConfig["hopkinsonList"].(map[string]interface{})
	if !ok {
		return false, nil
	}

	delete(list, name)

	finalData, err := json.MarshalIndent(fullConfig, "", "    ")
	if err != nil {
		return false, err
	}

	err = os.WriteFile(a.GetConfigPath(), finalData, 0644)
	return err == nil, err
}

func (app *App) SaveSampleConfig(sample backend.Sample) (bool, error) {
	var fullConfig map[string]interface{}
	data, err := os.ReadFile(app.GetConfigPath())
	if err == nil {
		json.Unmarshal(data, &fullConfig)
	}

	if fullConfig == nil {
		fullConfig = make(map[string]interface{})
	}

	strSample := map[string]string{
		"number":   fmt.Sprintf("%v", sample.Number),
		"name":     sample.Name,
		"material": sample.Material,
		"diameter": fmt.Sprintf("%v", sample.Diameter),
		"length":   fmt.Sprintf("%v", sample.Length),
	}

	fullConfig["sample"] = strSample

	finalData, err := json.MarshalIndent(fullConfig, "", "    ")
	if err != nil {
		return false, err
	}

	err = os.WriteFile(app.GetConfigPath(), finalData, 0644)
	if err != nil {
		return false, err
	}

	return true, nil
}

// API
type ShpbData struct {
	IncWave   backend.ShpbSignal `json:"incWave"`
	TransWave backend.ShpbSignal `json:"transWave"`
	IncFFT    backend.ShpbSignal `json:"incFFT"`
	TransFFT  backend.ShpbSignal `json:"transFFT"`
	Message   string             `json:"message"`
	Success   bool               `json:"success"`
}

func (a *App) LoadShpbData(incData string, transData string) ShpbData {
	err := a.hopBar.ParseData(incData, transData)
	if err != nil {
		return ShpbData{Success: false, Message: "数据加载失败: " + err.Error()}
	}

	incFFT := a.hopBar.IncWave.FFT()
	transFFT := a.hopBar.TransWave.FFT()

	return ShpbData{
		IncWave:   a.hopBar.IncWaveShow,
		TransWave: a.hopBar.TransWaveShow,
		IncFFT:    *incFFT,
		TransFFT:  *transFFT,
		Success:   true,
		Message:   "数据加载完成",
	}
}

type CropData struct {
	Wave     backend.ShpbSignal `json:"Wave"`
	WaveType string             `json:"string"`
	Message  string             `json:"message"`
	Success  bool               `json:"success"`
}

func (a *App) SetWaveRange(waveType string, start float64, end float64) CropData {
	if a.hopBar.CropDataStartAndEnd == nil {
		a.hopBar.CropDataStartAndEnd = make(map[string]*backend.Range)
	}

	getIdx := func(s *backend.ShpbSignal, timeVal float64) int {
		if len(s.X) < 2 {
			return 0
		}
		dt := s.X[1] - s.X[0]
		idx := int(math.Round((timeVal - s.X[0]) / dt))
		if idx < 0 {
			return 0
		}
		if idx >= len(s.X) {
			return len(s.X) - 1
		}
		return idx
	}

	var res *backend.ShpbSignal
	var startIdx int

	switch waveType {
	case "incident", "reflected":
		res = a.hopBar.IncWaveShow.CropSignal(start, end)
		startIdx = getIdx(&a.hopBar.IncWaveShow, start)

		if waveType == "incident" {
			a.hopBar.IncWaveCrop = *res
			a.hopBar.CropDataStartAndEnd["inc"] = &backend.Range{Start: startIdx, End: startIdx + len(res.Y)}
		} else {
			a.hopBar.RefWaveCrop = *res
			a.hopBar.CropDataStartAndEnd["ref"] = &backend.Range{Start: startIdx, End: startIdx + len(res.Y)}
		}

	case "transmitted":
		res = a.hopBar.TransWaveShow.CropSignal(start, end)
		startIdx = getIdx(&a.hopBar.TransWaveShow, start)

		a.hopBar.TransWaveCrop = *res
		a.hopBar.CropDataStartAndEnd["trans"] = &backend.Range{Start: startIdx, End: startIdx + len(res.Y)}

	default:
		return CropData{Message: "无效波形类型", Success: false, WaveType: waveType}
	}

	return CropData{
		Message:  "截取成功并记录起始点",
		Success:  true,
		Wave:     *res,
		WaveType: waveType,
	}
}

func (a *App) SaveSampleDetails(sample backend.Sample) bool {
	print(sample.Diameter)
	if a.hopBar == nil {
		return false
	}
	a.hopBar.Sample = sample
	return true
}

func (a *App) SaveHopkinsonDetails(bar backend.HopkinsonBar) bool {
	if a.hopBar == nil {
		return false
	}
	a.hopBar.Hopkinson = bar
	return true
}

// 基线校准
func (a *App) updateProcessedWaves() {
	a.hopBar.IncWaveShow = a.hopBar.IncWave.DeepCopy()
	a.hopBar.IncWaveShow.ApplyBaseline(a.hopBar.OffsetINC / 1000.0)
	a.hopBar.IncWaveShow.ApplyLowPassFilter(a.hopBar.LowPassFreq)
	a.hopBar.IncWaveShow.ApplyHighPassFilter(a.hopBar.HighPassFreq)

	a.hopBar.TransWaveShow = a.hopBar.TransWave.DeepCopy()
	a.hopBar.TransWaveShow.ApplyBaseline(a.hopBar.OffsetTRANS / 1000.0)
	a.hopBar.TransWaveShow.ApplyLowPassFilter(a.hopBar.LowPassFreq)
	a.hopBar.TransWaveShow.ApplyHighPassFilter(a.hopBar.HighPassFreq)
}

type BaselineData struct {
	Wave     backend.ShpbSignal `json:"Wave"`
	WaveType string             `json:"WaveType"`
	Message  string             `json:"Message"`
	Success  bool               `json:"Success"`
}

func (a *App) BaselineCorrection(data map[string]float64) BaselineData {
	offsetIn := data["inc"] - a.hopBar.OffsetINC
	offsetTrans := data["trans"] - a.hopBar.OffsetTRANS

	if offsetIn != 0 {
		a.hopBar.OffsetINC = data["inc"]
		a.updateProcessedWaves()
		return BaselineData{
			Wave:     a.hopBar.IncWaveShow,
			WaveType: "inc",
			Message:  "success",
			Success:  true,
		}
	}

	if offsetTrans != 0 {
		a.hopBar.OffsetTRANS = data["trans"]
		a.updateProcessedWaves()
		return BaselineData{
			Wave:     a.hopBar.TransWaveShow,
			WaveType: "trans",
			Message:  "success",
			Success:  true,
		}
	}
	return BaselineData{}
}

// 频谱滤波

type FilterData struct {
	Waves    []backend.ShpbSignal `json:"Waves"`
	WaveType string               `json:"WaveType"`
	Message  string               `json:"Message"`
	Success  bool                 `json:"Success"`
}

func (a *App) SpectrumFilter(data map[string]float64) FilterData {
	a.hopBar.LowPassFreq = float64(data["high"])
	a.hopBar.HighPassFreq = float64(data["low"])
	a.updateProcessedWaves()

	return FilterData{
		Message:  "Success",
		Success:  true,
		WaveType: "Spectrum",
		Waves:    []backend.ShpbSignal{a.hopBar.IncWaveShow, a.hopBar.TransWaveShow},
	}

}

// 时域对齐
type AlignResult struct {
	Success bool                 `json:"Success"`
	Waves   []backend.ShpbSignal `json:"Waves"` // 顺序：0-入射, 1-反射, 2-透射
	Message string               `json:"Message"`
}

func (a *App) TimeAlign(AlignType string) AlignResult {
	println(AlignType)

	var alignedWaves []backend.ShpbSignal

	switch AlignType {
	case "peak":
		alignedWaves = a.hopBar.AlignPeak()
	case "time":
		alignedWaves = a.hopBar.AlignTime()
	case "gb":
		alignedWaves = a.hopBar.AlignGB()
	default:
		return AlignResult{
			Success: false,
			Message: "无效的对齐类型",
		}

	}

	a.hopBar.IncWaveCrop = alignedWaves[0]
	a.hopBar.RefWaveCrop = alignedWaves[1]
	a.hopBar.TransWaveCrop = alignedWaves[2]

	return AlignResult{
		Success: true,
		Waves:   alignedWaves,
		Message: "三波对齐成功",
	}
}

func (a *App) ManulAlign(data map[string]float64) AlignResult {
	alignedWaves := a.hopBar.AlignManul(data)
	if alignedWaves == nil {
		return AlignResult{
			Success: false,
			Message: "对齐失败：未找到有效峰值",
		}
	}

	a.hopBar.IncWaveCrop = alignedWaves[0]
	a.hopBar.RefWaveCrop = alignedWaves[1]
	a.hopBar.TransWaveCrop = alignedWaves[2]

	return AlignResult{
		Success: true,
		Waves:   alignedWaves,
		Message: "三波对齐成功",
	}
}

func (a *App) InitSignalProcessor() {
	a.hopBar.CropDataStartAndEnd = make(map[string]*backend.Range)

	a.hopBar.CropDataStartAndEnd["inc"] = &backend.Range{Start: 0, End: 0}
	a.hopBar.CropDataStartAndEnd["ref"] = &backend.Range{Start: 0, End: 0}
	a.hopBar.CropDataStartAndEnd["trans"] = &backend.Range{Start: 0, End: 0}

}

func (a *App) StartCalculate(calcType string) backend.CalculationResult {
	println(calcType)
	result := a.hopBar.Calculate(calcType)
	return result
}

func (a *App) ExportData() string {
	filePath, err := runtime.SaveFileDialog(a.ctx, runtime.SaveDialogOptions{
		Title:           "保存实验结果",
		DefaultFilename: "SHPB_Results.csv",
		Filters: []runtime.FileFilter{
			{DisplayName: "CSV Files (*.csv)", Pattern: "*.csv"},
			{DisplayName: "Excel Files (*.xlsx)", Pattern: "*.xlsx"},
			{DisplayName: "MatLab Files (*.mat)", Pattern: "*.mat"},
		},
	})

	if err != nil {
		return "error: 对话框打开失败"
	}
	if filePath == "" {
		return "cancelled" // 用户取消了
	}

	err = a.saveToFile(filePath)
	if err != nil {
		log.Println(err)
		return "error: 保存文件失败"
	}

	return filePath
}

func (a *App) saveToFile(path string) error {
	res := a.hopBar.Result

	// 检查是否有计算结果 或 裁剪波形数据
	hasCalc := len(res.StressTime.X) > 0
	hasCrop := len(res.IncWaveCrop.Y) > 0 || len(res.RefWaveCrop.Y) > 0 || len(res.TransWaveCrop.Y) > 0

	if !hasCalc && !hasCrop {
		return fmt.Errorf("没有可导出的数据，请先完成计算或截取波形")
	}

	// 获取后缀名并转为小写
	ext := strings.ToLower(filepath.Ext(path))

	// 分支处理
	switch ext {
	case ".csv":
		return a.saveAsCSV(path, res)
	case ".mat":
		return a.saveAsMatLab(path, res)
	case ".xlsx":
		return a.saveAsExcel(path, res)
	default:
		return fmt.Errorf("不支持的文件格式")
	}
}

// 保存为真正的 CSV 格式
func (a *App) saveAsCSV(path string, res backend.CalculationResult) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// 写入 UTF-8 BOM，防止 Excel 打开 CSV 时乱码
	file.WriteString("\xEF\xBB\xBF")

	writer := csv.NewWriter(file)
	defer writer.Flush()

	hasCalc := len(res.StressTime.X) > 0

	if hasCalc {
		// 计算结果表头
		writer.Write([]string{"时间 (s)", "应变 (Strain)", "应变率 (1/s)", "应力 (MPa)"})
		for i := 0; i < len(res.StressTime.X); i++ {
			writer.Write([]string{
				fmt.Sprintf("%.8f", res.StrainTime.X[i]),
				fmt.Sprintf("%.8f", res.StrainTime.Y[i]),
				fmt.Sprintf("%.8f", res.StrainRateTime.Y[i]),
				fmt.Sprintf("%.8f", res.StressTime.Y[i]),
			})
		}
		writer.Write(nil) // 空行
	}

	// 裁剪波形数据（各自独立 X 轴）
	type cropInfo struct {
		name string
		x, y []float64
	}
	crops := []cropInfo{
		{"入射波", res.IncWaveCrop.X, res.IncWaveCrop.Y},
		{"反射波", res.RefWaveCrop.X, res.RefWaveCrop.Y},
		{"透射波", res.TransWaveCrop.X, res.TransWaveCrop.Y},
	}
	for _, c := range crops {
		if len(c.x) == 0 {
			continue
		}
		writer.Write([]string{c.name + " 时间 (s)", c.name + " 幅值 (V)"})
		for i := 0; i < len(c.x); i++ {
			writer.Write([]string{
				fmt.Sprintf("%.8f", c.x[i]),
				fmt.Sprintf("%.8f", c.y[i]),
			})
		}
		writer.Write(nil)
	}

	return nil
}

// 保存为 Excel 格式
func (a *App) saveAsExcel(path string, res backend.CalculationResult) error {
	f := excelize.NewFile()
	defer f.Close()

	hasCalc := len(res.StressTime.X) > 0

	if hasCalc {
		sheet := "TimeHistory"
		f.SetSheetName("Sheet1", sheet)

		sw, err := f.NewStreamWriter(sheet)
		if err != nil {
			return err
		}

		sw.SetRow("A1", []interface{}{"时间 (s)", "应变", "应变率 (1/s)", "应力 (MPa)"})
		for i := 0; i < len(res.StressTime.X); i++ {
			row := i + 2
			cell, _ := excelize.CoordinatesToCellName(1, row)
			sw.SetRow(cell, []interface{}{
				res.StrainTime.X[i],
				res.StrainTime.Y[i],
				res.StrainRateTime.Y[i],
				res.StressTime.Y[i],
			})
		}
		sw.Flush()
	}

	// 裁剪波形写入单独 sheet
	type cropInfo struct {
		name string
		x, y []float64
	}
	crops := []cropInfo{
		{"IncWave", res.IncWaveCrop.X, res.IncWaveCrop.Y},
		{"RefWave", res.RefWaveCrop.X, res.RefWaveCrop.Y},
		{"TransWave", res.TransWaveCrop.X, res.TransWaveCrop.Y},
	}
	hasAnyCrop := false
	for _, c := range crops {
		if len(c.x) > 0 {
			hasAnyCrop = true
			break
		}
	}
	if hasAnyCrop {
		sheet := "CropWaves"
		f.NewSheet(sheet)
		sw, err := f.NewStreamWriter(sheet)
		if err != nil {
			return err
		}

		// 表头：三组 (时间, 幅值)
		headers := []interface{}{}
		for _, c := range crops {
			headers = append(headers, c.name+" 时间", c.name+" 幅值")
		}
		sw.SetRow("A1", headers)

		maxLen := 0
		for _, c := range crops {
			if len(c.x) > maxLen {
				maxLen = len(c.x)
			}
		}

		for i := 0; i < maxLen; i++ {
			row := i + 2
			cell, _ := excelize.CoordinatesToCellName(1, row)
			vals := []interface{}{}
			for _, c := range crops {
				if i < len(c.x) {
					vals = append(vals, c.x[i], c.y[i])
				} else {
					vals = append(vals, "", "")
				}
			}
			sw.SetRow(cell, vals)
		}
		sw.Flush()
	}

	return f.SaveAs(path)
}

func (a *App) saveAsMatLab(path string, res backend.CalculationResult) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	if err := writeMatlabHeader(file); err != nil {
		return err
	}

	hasCalc := len(res.StressTime.X) > 0

	if hasCalc {
		timeData := res.StrainTime.X
		timeHistory := interleaveColumns(timeData, res.StrainTime.Y, res.StrainRateTime.Y, res.StressTime.Y)
		calcVars := []struct {
			name string
			rows int
			cols int
			data []float64
		}{
			{name: "time_s", rows: len(timeData), cols: 1, data: timeData},
			{name: "strain", rows: len(res.StrainTime.Y), cols: 1, data: res.StrainTime.Y},
			{name: "strain_rate_1_s", rows: len(res.StrainRateTime.Y), cols: 1, data: res.StrainRateTime.Y},
			{name: "stress_mpa", rows: len(res.StressTime.Y), cols: 1, data: res.StressTime.Y},
			{name: "time_history", rows: len(timeHistory) / 4, cols: 4, data: timeHistory},
		}
		for _, v := range calcVars {
			if err := writeMatlabDoubleMatrix(file, v.name, v.rows, v.cols, v.data); err != nil {
				return err
			}
		}
	}

	// 裁剪波形：分别保存 X 和 Y
	type cropInfo struct {
		name string
		x, y []float64
	}
	crops := []cropInfo{
		{"inc_wave", res.IncWaveCrop.X, res.IncWaveCrop.Y},
		{"ref_wave", res.RefWaveCrop.X, res.RefWaveCrop.Y},
		{"trans_wave", res.TransWaveCrop.X, res.TransWaveCrop.Y},
	}
	for _, c := range crops {
		if len(c.x) == 0 {
			continue
		}
		// 每条波写入 (X, Y) 为 Nx2 矩阵
		xy := interleaveColumns(c.x, c.y)
		if err := writeMatlabDoubleMatrix(file, c.name, len(c.x), 2, xy); err != nil {
			return err
		}
	}

	return nil
}

func writeMatlabHeader(file *os.File) error {
	description := fmt.Sprintf("MATLAB 5.0 MAT-file, Created by Hopkinson Decoder on %s", time.Now().Format(time.RFC3339))
	header := make([]byte, 128)
	copy(header[:116], []byte(description))
	header[124] = 0
	header[125] = 1
	header[126] = 'I'
	header[127] = 'M'
	_, err := file.Write(header)
	return err
}

func writeMatlabDoubleMatrix(file *os.File, name string, rows int, cols int, data []float64) error {
	if rows < 0 || cols < 0 || len(data) != rows*cols {
		return fmt.Errorf("invalid MATLAB matrix %q dimensions", name)
	}

	var payload bytes.Buffer

	writeMatlabDataElement(&payload, miUINT32, []uint32{mxDOUBLE_CLASS, 0})
	writeMatlabDataElement(&payload, miINT32, []int32{int32(rows), int32(cols)})
	writeMatlabDataElement(&payload, miINT8, []byte(name))
	writeMatlabDataElement(&payload, miDOUBLE, data)

	matrixBytes := payload.Bytes()
	if err := binary.Write(file, binary.LittleEndian, uint32(miMATRIX)); err != nil {
		return err
	}
	if err := binary.Write(file, binary.LittleEndian, uint32(len(matrixBytes))); err != nil {
		return err
	}
	_, err := file.Write(matrixBytes)
	return err
}

func writeMatlabDataElement[T byte | int32 | uint32 | float64](builder *bytes.Buffer, dataType uint32, values []T) {
	var raw bytes.Buffer
	for _, value := range values {
		_ = binary.Write(&raw, binary.LittleEndian, value)
	}

	data := raw.Bytes()
	_ = binary.Write(builder, binary.LittleEndian, dataType)
	_ = binary.Write(builder, binary.LittleEndian, uint32(len(data)))
	builder.Write(data)
	writeMatlabPadding(builder, len(data))
}

func writeMatlabPadding(builder *bytes.Buffer, size int) {
	for i := 0; i < (8-size%8)%8; i++ {
		builder.WriteByte(0)
	}
}

func interleaveColumns(columns ...[]float64) []float64 {
	if len(columns) == 0 {
		return nil
	}

	rows := len(columns[0])
	data := make([]float64, 0, rows*len(columns))
	for _, column := range columns {
		if len(column) < rows {
			rows = len(column)
		}
	}

	for col := 0; col < len(columns); col++ {
		for row := 0; row < rows; row++ {
			data = append(data, columns[col][row])
		}
	}
	return data
}

const (
	miINT8         = 1
	miUINT32       = 6
	miINT32        = 5
	miDOUBLE       = 9
	miMATRIX       = 14
	mxDOUBLE_CLASS = 6
)
