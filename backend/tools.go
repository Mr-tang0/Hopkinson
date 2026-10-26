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
	separators := strings.NewReplacer(",", " ", ";", " ")

	for _, line := range lines {
		fields := strings.Fields(separators.Replace(line))
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

func findGBBaseIndex(y []float64, peak int) int {
	if peak <= 0 || peak >= len(y) {
		return -1
	}

	threshold := math.Abs(y[peak]) / 10.0
	for i := peak; i >= 0; i-- {
		if math.Abs(y[i]) <= threshold {
			println("找到第一个有效顶点索引:", i, "幅值:", y[i], "peak:", y[peak])
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

// CumulativeEuler 右端点欧拉累积积分，与 MATLAB 版本
// strain(i) = strain(i-1) + dt * strain_r(i) 完全一致
func CumulativeEuler(y []float64, dx float64) []float64 {
	n := len(y)
	if n == 0 {
		return []float64{}
	}
	result := make([]float64, n)
	result[0] = y[0] * dx
	for i := 1; i < n; i++ {
		result[i] = result[i-1] + y[i]*dx
	}
	return result
}

// SavgolSmooth 对信号进行 Savitzky-Golay 平滑
// 对应 MATLAB: smooth(x, span, 'sgolay')，默认多项式阶数为 2
// span 会自动调整为不超过信号长度的奇数
func SavgolSmooth(y []float64, span int, order int) []float64 {
	n := len(y)
	if n == 0 {
		return []float64{}
	}
	out := make([]float64, n)
	copy(out, y)

	if span < 3 || n < 3 {
		return out
	}
	if order < 0 {
		order = 2
	}
	if span > n {
		span = n
	}
	// 与 MATLAB smooth 一致：偶数窗宽 +1 调整为奇数
	if span%2 == 0 {
		if span < n {
			span++
		} else {
			span--
		}
	}
	if span < 3 {
		return out
	}
	if order >= span {
		order = span - 1
	}

	half := span / 2

	// 中间点：使用以当前点为中心的对称窗口
	centerWeights := savgolWeights(span, order, half)
	if centerWeights == nil {
		return out
	}
	for i := half; i < n-half; i++ {
		val := 0.0
		base := i - half
		for j, w := range centerWeights {
			val += w * y[base+j]
		}
		out[i] = val
	}

	// 头部：固定使用前 span 个点，等价于 MATLAB sgolayfilt 的边缘处理
	for i := 0; i < half; i++ {
		weights := savgolWeights(span, order, i)
		if weights == nil {
			continue
		}
		val := 0.0
		for j, w := range weights {
			val += w * y[j]
		}
		out[i] = val
	}

	// 尾部：固定使用最后 span 个点
	for i := n - half; i < n; i++ {
		center := span - 1 - (n - 1 - i)
		weights := savgolWeights(span, order, center)
		if weights == nil {
			continue
		}
		val := 0.0
		base := n - span
		for j, w := range weights {
			val += w * y[base+j]
		}
		out[i] = val
	}

	return out
}

// savgolWeights 计算 Savitzky-Golay 拟合在窗口内 center 处的权重
func savgolWeights(m int, order int, center int) []float64 {
	if m <= 0 || center < 0 || center >= m {
		return nil
	}
	if order >= m {
		order = m - 1
	}
	size := order + 1

	// 构造正规方程 A^T A
	ata := make([][]float64, size)
	for i := range ata {
		ata[i] = make([]float64, size)
	}
	powers := make([]float64, 2*size-1)
	for j := 0; j < m; j++ {
		t := float64(j - center)
		powers[0] = 1
		for p := 1; p < len(powers); p++ {
			powers[p] = powers[p-1] * t
		}
		for a := 0; a < size; a++ {
			for b := 0; b < size; b++ {
				ata[a][b] += powers[a+b]
			}
		}
	}

	// 求解 (A^T A) * sol = e0，得到逆矩阵第 0 列
	rhs := make([]float64, size)
	rhs[0] = 1
	sol := solveLinearSystem(ata, rhs)
	if sol == nil {
		return nil
	}

	weights := make([]float64, m)
	for j := 0; j < m; j++ {
		t := float64(j - center)
		val := 0.0
		p := 1.0
		for a := 0; a < size; a++ {
			val += sol[a] * p
			p *= t
		}
		weights[j] = val
	}
	return weights
}

// solveLinearSystem 高斯消元求解线性方程组（部分主元）
func solveLinearSystem(a [][]float64, b []float64) []float64 {
	n := len(a)
	if n == 0 {
		return nil
	}

	// 复制，避免修改传入矩阵
	mat := make([][]float64, n)
	for i := 0; i < n; i++ {
		if len(a[i]) != n {
			return nil
		}
		mat[i] = make([]float64, n+1)
		copy(mat[i], a[i])
		mat[i][n] = b[i]
	}

	for col := 0; col < n; col++ {
		// 选主元
		pivot := col
		for row := col + 1; row < n; row++ {
			if math.Abs(mat[row][col]) > math.Abs(mat[pivot][col]) {
				pivot = row
			}
		}
		if math.Abs(mat[pivot][col]) < 1e-12 {
			return nil
		}
		mat[col], mat[pivot] = mat[pivot], mat[col]

		// 归一化主元行
		div := mat[col][col]
		for j := col; j <= n; j++ {
			mat[col][j] /= div
		}

		// 消去其余行
		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			factor := mat[row][col]
			if factor == 0 {
				continue
			}
			for j := col; j <= n; j++ {
				mat[row][j] -= factor * mat[col][j]
			}
		}
	}

	sol := make([]float64, n)
	for i := 0; i < n; i++ {
		sol[i] = mat[i][n]
	}
	return sol
}
