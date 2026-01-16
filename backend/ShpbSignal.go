package backend

import (
	"math"
	"math/cmplx"

	"github.com/mjibson/go-dsp/fft"
)

type ShpbSignal struct {
	X []float64
	Y []float64
}

func (s *ShpbSignal) getDT() float64 {
	if len(s.X) < 2 {
		return 0.000001 // 默认 1MHz 采样率的间隔
	}
	return (s.X[100] - s.X[0]) / 100
}

// 指定向量进行基线校准
func (s *ShpbSignal) ApplyBaseline(offset float64) {

	for i := range s.Y {
		s.Y[i] -= offset
	}
}

// LowPassFilter 低通滤波：允许低于截止频率的信号通过，消除高频噪声
func (s *ShpbSignal) ApplyLowPassFilter(cutoff float64) {
	if len(s.Y) == 0 || cutoff <= 0 {
		return
	}

	dt := s.getDT()
	// 计算平滑系数 alpha = dt / (RC + dt) = 2π * dt * f / (2π * dt * f + 1)
	rc := 1.0 / (2 * math.Pi * cutoff)
	alpha := dt / (rc + dt)

	newY := make([]float64, len(s.Y))
	newY[0] = s.Y[0]

	for i := 1; i < len(s.Y); i++ {
		// y[i] = α * x[i] + (1-α) * y[i-1]
		newY[i] = alpha*s.Y[i] + (1-alpha)*newY[i-1]
	}
	s.Y = newY
}

// HighPassFilter 高通滤波：允许高于截止频率的信号通过，消除基线漂移或低频干扰
func (s *ShpbSignal) ApplyHighPassFilter(cutoff float64) {
	if len(s.Y) == 0 || cutoff <= 0 {
		return
	}

	dt := s.getDT()
	rc := 1.0 / (2 * math.Pi * cutoff)
	alpha := rc / (rc + dt)

	newY := make([]float64, len(s.Y))
	newY[0] = s.Y[0]

	for i := 1; i < len(s.Y); i++ {
		// y[i] = α * (y[i-1] + x[i] - x[i-1])
		newY[i] = alpha * (newY[i-1] + s.Y[i] - s.Y[i-1])
	}
	s.Y = newY
}

// 截取片段
// CropSignal 根据起始和结束的 X 值（如时间点）截取信号片段
func (s *ShpbSignal) CropSignal(startX float64, endX float64) *ShpbSignal {
	if len(s.X) == 0 || len(s.Y) == 0 {
		return &ShpbSignal{}
	}

	// 确保 start < end
	if startX > endX {
		startX, endX = endX, startX
	}

	startIndex := -1
	endIndex := -1

	// 寻找索引（由于 X 是有序的，我们可以快速定位）
	for i, x := range s.X {
		// 寻找第一个大于等于 startX 的点
		if startIndex == -1 && x >= startX {
			startIndex = i
		}
		// 寻找第一个大于等于 endX 的点
		if endIndex == -1 && x >= endX {
			endIndex = i
			break // 找到了终点，退出循环
		}
	}

	// 边界处理
	if startIndex == -1 {
		return &ShpbSignal{} // 范围内没有数据
	}
	if endIndex == -1 {
		endIndex = len(s.X) - 1 // 如果 endX 超出范围，截取到最后
	}

	// 执行截取 (注意：Go 的切片是左闭右开 [start:end])
	// 为了包含最后一个点，我们使用 endIndex + 1
	actualEnd := endIndex + 1
	if actualEnd > len(s.X) {
		actualEnd = len(s.X)
	}

	// 注意：为了避免内存泄露或意外修改原数据，建议 Copy 切片
	newX := make([]float64, actualEnd-startIndex)
	newY := make([]float64, actualEnd-startIndex)
	copy(newX, s.X[startIndex:actualEnd])
	copy(newY, s.Y[startIndex:actualEnd])

	return &ShpbSignal{
		X: newX,
		Y: newY,
	}
}

// fft
func (s *ShpbSignal) FFT() *ShpbSignal {
	N := len(s.Y)
	if N < 2 {
		return &ShpbSignal{}
	}

	sumDt := 0.0
	for i := 1; i < N; i++ {
		sumDt += (s.X[i] - s.X[i-1])
	}
	avgDt := sumDt / float64(N-1)
	Y := fft.FFTReal(s.Y)

	halfN := N/2 + 1
	freqsPos := make([]float64, halfN)
	magnitude := make([]float64, halfN)

	for i := 0; i < halfN; i++ {
		freqsPos[i] = float64(i) / (float64(N) * avgDt)

		mag := cmplx.Abs(Y[i])

		if i == 0 {
			magnitude[i] = mag / float64(N)
		} else {
			magnitude[i] = (mag * 2.0) / float64(N)
		}
	}

	return &ShpbSignal{
		X: freqsPos,
		Y: magnitude,
	}
}

func (s *ShpbSignal) DeepCopy() ShpbSignal {
	// 必须为 Y 轴重新分配独立的内存空间
	newY := make([]float64, len(s.Y))
	copy(newY, s.Y)

	// 必须为 X 轴重新分配独立的内存空间
	newX := make([]float64, len(s.X))
	copy(newX, s.X)

	return ShpbSignal{
		X: newX,
		Y: newY,
	}
}
