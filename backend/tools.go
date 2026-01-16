package backend

import (
	"errors"
	"math"
	"strconv"
	"strings"
)

// 解析csv文件
func parseCsvToVector(raw string) (ShpbSignal, error) {
	lines := strings.Split(strings.TrimSpace(raw), "\n")
	if len(lines) == 0 {
		return ShpbSignal{}, errors.New("empty data")
	}

	timeVec := make([]float64, 0, len(lines))
	valVec := make([]float64, 0, len(lines))

	for _, line := range lines {
		fields := strings.Fields(strings.ReplaceAll(line, ",", " "))
		if len(fields) < 2 {
			continue
		}

		t, err1 := strconv.ParseFloat(fields[0], 64)
		v, err2 := strconv.ParseFloat(fields[1], 64)

		if err1 == nil && err2 == nil {
			timeVec = append(timeVec, t)
			valVec = append(valVec, v)
		}
	}
	return ShpbSignal{X: timeVec, Y: valVec}, nil
}

// GaussianFilter1d 简单实现一维高斯平滑
func GaussianFilter1d(data []float64, sigma float64) []float64 {
	size := int(math.Ceil(sigma * 6)) // 取 6 倍标准差保证覆盖大部分权重
	if size%2 == 0 {
		size++
	}
	kernel := make([]float64, size)
	half := size / 2
	sum := 0.0

	// 生成高斯核
	for i := 0; i < size; i++ {
		x := float64(i - half)
		kernel[i] = math.Exp(-(x * x) / (2 * sigma * sigma))
		sum += kernel[i]
	}
	// 归一化
	for i := range kernel {
		kernel[i] /= sum
	}

	smoothed := make([]float64, len(data))
	for i := range data {
		val := 0.0
		for j := 0; j < size; j++ {
			idx := i + j - half
			// 边界处理：使用最近邻填充
			if idx < 0 {
				idx = 0
			} else if idx >= len(data) {
				idx = len(data) - 1
			}
			val += data[idx] * kernel[j]
		}
		smoothed[i] = val
	}
	return smoothed
}

// FindFirstMajorPeak 查找平滑后的第一个幅值最大的点
// FindFirstMajorPeak 查找平滑后的第一个有效顶点（导数为0的点）
func FindFirstMajorPeak(s *ShpbSignal, sigma float64) int {
	if len(s.Y) < 3 {
		return -1
	}

	// 1. 高斯平滑
	smoothed := GaussianFilter1d(s.Y, sigma)

	// 2. 判断趋势 (均值) 用于确定是找上顶点还是下顶点
	sum := 0.0
	for _, v := range s.Y {
		sum += v
	}
	findPositivePeaks := (sum / float64(len(s.Y))) >= 0

	// 3. 计算全局最大绝对幅值，用于设定噪声过滤阈值
	globalMaxAbs := 0.0
	for _, v := range s.Y {
		if math.Abs(v) > globalMaxAbs {
			globalMaxAbs = math.Abs(v)
		}
	}
	threshold := globalMaxAbs * 0.25 // 过滤掉小于最大幅值25%的干扰点

	// 4. 按顺序查找第一个满足条件的顶点
	for i := 1; i < len(smoothed)-1; i++ {
		isPeak := false
		if findPositivePeaks {
			// 检测上顶点 (波峰)
			if smoothed[i] > smoothed[i-1] && smoothed[i] > smoothed[i+1] {
				isPeak = true
			}
		} else {
			// 检测下顶点 (波谷)
			if smoothed[i] < smoothed[i-1] && smoothed[i] < smoothed[i+1] {
				isPeak = true
			}
		}

		// 💡 关键改动：找到第一个满足幅值阈值的顶点就立即返回
		if isPeak && math.Abs(s.Y[i]) >= threshold {
			println("找到第一个有效顶点索引:", i, "幅值:", s.Y[i])
			return i
		}
	}

	return -1
}

// cropSlice 辅助函数：根据起始索引截取固定长度，并进行时间归零
func cropSlice(src *ShpbSignal, start int, length int) ShpbSignal {
	newY := make([]float64, length)
	newX := make([]float64, length)

	// 获取采样间隔 dt (假设 X 轴单位是微秒或秒)
	dt := 1.0
	if len(src.X) > 1 {
		dt = src.X[1] - src.X[0]
	}

	for i := 0; i < length; i++ {
		srcIdx := start + i
		// 边界保护
		if srcIdx >= 0 && srcIdx < len(src.Y) {
			newY[i] = src.Y[srcIdx]
		} else {
			newY[i] = 0 // 越界填0
		}
		// 时间归零：每个点都是从 0 开始累计
		newX[i] = float64(i) * dt
	}

	return ShpbSignal{
		X: newX,
		Y: newY,
	}
}

// CumTrapz 梯形数值积分
func CumTrapz(y []float64, dx float64) []float64 {
	n := len(y)
	if n == 0 {
		return []float64{}
	}
	result := make([]float64, n)
	result[0] = 0 // initial=0
	for i := 1; i < n; i++ {
		// 梯形公式: (y[i-1] + y[i]) * dx / 2
		result[i] = result[i-1] + (y[i-1]+y[i])*dx/2.0
	}
	return result
}
