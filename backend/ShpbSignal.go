package backend

type ShpbSignal struct {
	X []float64
	Y []float64
}

// 指定向量进行基线校准
func (s *ShpbSignal) ApplyBaseline(offset float64) {

	for i := range s.Y {
		s.Y[i] -= offset
	}
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

// DeepCopy 深拷贝信号，避免显示数据与原始数据互相影响
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
