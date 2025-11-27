import numpy as np


def gaussian_kernel(sigma, truncate=4.0):
    """
    生成一维高斯核。

    参数:
    sigma (float): 高斯分布的标准差，决定了平滑程度。
    truncate (float): 核的裁剪因子。核的长度为 2 * int(truncate * sigma) + 1。
                      默认为 4.0，与 SciPy 的默认行为一致。

    返回:
    ndarray: 归一化后的一维高斯核。
    """
    # 计算核的半径
    radius = int(truncate * sigma + 0.5)

    # 创建核的 x 轴范围，从 -radius 到 radius
    x = np.arange(-radius, radius + 1)

    # 避免除以零的错误，如果 sigma 极小，则使用一个窄核
    if sigma < 1e-8:
        # 如果 sigma 接近零，核中心为 1，其他为 0 (即不平滑)
        g = np.zeros_like(x, dtype=float)
        g[radius] = 1.0
        return g

    # 高斯函数公式: exp(-0.5 * (x / sigma)^2)
    g = np.exp(-0.5 * (x / sigma) ** 2)

    # 归一化: 使核的和为 1
    return g / np.sum(g)


def gaussian_filter1d(input_array, sigma, mode='reflect', cval=0.0, truncate=4.0):
    """
    使用 NumPy 实现的一维高斯平滑滤波。

    参数:
    input_array (array_like): 要平滑的数据。
    sigma (float): 高斯核的标准差。
    mode (str, optional): 边界处理模式。NumPy 的 np.convolve 只支持 'full', 'same', 'valid'。
                          此处我们使用 'same' 并进行填充来模拟 SciPy 的边界模式。
                          默认为 'reflect' (与 SciPy 默认值不同，但这里使用 'reflect' 来模拟)。
    cval (float, optional): 仅用于 'constant' 模式的填充值。
    truncate (float, optional): 核的裁剪因子。

    返回:
    ndarray: 平滑后的数据。
    """

    input_array = np.asarray(input_array)
    kernel = gaussian_kernel(sigma, truncate=truncate)

    # --- 边界处理模拟 (模拟 SciPy 的填充模式) ---

    # 计算需要填充的长度 (半核长度)
    radius = (len(kernel) - 1) // 2

    # SciPy 的默认模式是 'reflect'。我们使用 np.pad 来实现常见的边界模式。
    if mode == 'reflect':
        # 在两端反射填充，填充长度为 radius
        padded_array = np.pad(input_array, (radius, radius), mode='reflect')
    elif mode == 'wrap':
        # 循环填充
        padded_array = np.pad(input_array, (radius, radius), mode='wrap')
    elif mode == 'constant':
        # 常量填充
        padded_array = np.pad(input_array, (radius, radius), mode='constant', constant_values=cval)
    elif mode == 'nearest':
        # 边界值填充
        padded_array = np.pad(input_array, (radius, radius), mode='edge')
    else:
        # 如果无法识别或使用了复杂的 SciPy 模式，退回 'reflect' 或根据你的需求修改
        padded_array = np.pad(input_array, (radius, radius), mode='reflect')

    # --- 卷积 ---

    # 使用 np.convolve 进行卷积。
    # mode='valid' 确保输出的有效部分 (即没有核接触到填充边界的部分)。
    # 但是我们已经手动填充了，所以需要调整切片。

    # 1. 对填充后的数组和核进行全卷积
    smoothed_full = np.convolve(padded_array, kernel, mode='valid')

    # 2. 截取到原始长度 (因为填充的长度是 radius * 2，并且卷积后长度会变化)

    # 在 'valid' 模式下，卷积的长度是 len(padded_array) - len(kernel) + 1
    # 对于我们的填充方式，这个长度正好是 len(input_array)
    return smoothed_full


def lfilter_numpy(b, a, x):
    """
    一个简化的 lfilter 替代品，实现了 IIR/FIR 滤波器的差分方程。
    y[n] = (b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]) -
           (a[1]*y[n-1] + a[2]*y[n-2] + ... + a[N]*y[n-N])

    参数:
    b (array): 前馈系数 (分子)。
    a (array): 反馈系数 (分母)。a[0] 必须是 1.0 (或归一化后)。
    x (array): 输入信号。
    """

    # 确保 a[0] 归一化为 1
    if a[0] != 1.0:
        a = np.asarray(a) / a[0]
        b = np.asarray(b) / a[0]

    M = len(b) - 1  # 分子阶数
    N = len(a) - 1  # 分母阶数

    # 初始化输出数组
    y = np.zeros_like(x, dtype=x.dtype)

    # 状态变量 (用于保存过去 M/N 个输入/输出值)
    # 对于简单的 lfilter，我们直接使用差分方程迭代

    # 迭代计算
    for n in range(len(x)):
        # 前馈项 (输入 x)
        sum_b = 0.0
        for i in range(M + 1):
            if n - i >= 0:
                sum_b += b[i] * x[n - i]

        # 反馈项 (输出 y)
        sum_a = 0.0
        for i in range(1, N + 1):
            if n - i >= 0:
                sum_a += a[i] * y[n - i]

        # 差分方程
        y[n] = sum_b - sum_a

    return y


# --- 2. 巴特沃斯系数设计 (butter 替代品) ---

def butter_numpy(N, Wn, btype='low', analog=False):
    """
    一个简化的巴特沃斯滤波器系数计算替代品 (仅支持低通)。
    仅使用 NumPy 和 numpy.lib.scimath (用于复数)。

    参数:
    N (int): 滤波器阶数。
    Wn (float): 归一化截止频率 (0 < Wn < 1)。
    btype (str): 滤波器类型 ('low' for lowpass)。

    返回:
    b (ndarray): 分子系数。
    a (ndarray): 分母系数。
    """
    if btype != 'low':
        raise NotImplementedError("butter_numpy only supports lowpass ('low') type.")

    # 计算极点 (poles)
    # k 从 0 到 N-1
    k = np.arange(N)

    # 巴特沃斯极点在 s 平面 (模拟域) 上的位置
    # 极点位于单位圆上，角度为 (2k + N - 1) / (2N) * pi
    analog_poles = np.exp(1j * np.pi * (2 * k + N + 1) / (2 * N))

    # 转换为数字滤波器：使用双线性变换 (Bilinear Transform)
    # 公式：z = (1 + s * Wn / 2) / (1 - s * Wn / 2)
    # 我们的 Wn 是归一化数字频率，需要先转换为预畸变频率

    # 预畸变频率 (pre-warped frequency): omega_a = 2 * tan(pi * Wn / 2)
    omega_a = 2.0 * np.tan(np.pi * Wn / 2.0)

    # 将模拟极点 scaled_poles 转换为数字极点 digital_poles
    # scaled_poles = analog_poles * omega_a
    # digital_poles = (1 + scaled_poles / 2) / (1 - scaled_poles / 2)

    # 简化公式：
    poles = (1 + analog_poles * (omega_a / 2.0)) / (1 - analog_poles * (omega_a / 2.0))

    # 极点转换为分母多项式 (a 系数)
    # np.poly: 从根 (poles) 还原多项式系数
    a = np.poly(poles)

    # 归一化因子 (增益 G)
    # 对于低通，零点在 z=-1。增益 G 使滤波器在 DC (z=1) 处增益为 1。

    # 增益 G = 1 / abs(H(z=1))
    # H(z) = G * (z - z0)^N / (z - p)^N

    # 计算 z=1 时的增益：
    # 分母在 z=1 时的值
    a_at_1 = np.polyval(a, 1.0)

    # 零点在 z=-1 (对于低通)
    # 分子多项式 (z+1)^N 在 z=1 时的值
    b_at_1 = 2.0 ** N  # (1 - (-1))^N

    # 增益 G (用于归一化 b 系数)
    G = a_at_1 / b_at_1

    # 计算分子多项式 (b 系数)
    # 零点在 z=-1, 即 (z - (-1))^N = (z + 1)^N
    b = G * np.poly([-1.0] * N)

    # 由于数值精度问题，通常需要将系数中的极小虚部清除 (scipy 会做这个处理)
    a = np.real(a)
    b = np.real(b)

    return b, a


# --- 3. 零相位滤波 (filtfilt 替代品) ---

def filtfilt_numpy(b, a, x):
    """
    一个简化的 filtfilt 替代品，通过两次调用 lfilter_numpy 实现零相位滤波。
    注意：没有 SciPy 的状态初始化功能，结果可能与 SciPy 版本在信号两端有细微差异。
    """

    # 第一次正向滤波
    y_forward = lfilter_numpy(b, a, x)

    # 反转信号
    y_reversed = y_forward[::-1]

    # 第二次反向滤波
    y_final_reversed = lfilter_numpy(b, a, y_reversed)

    # 再次反转回原方向
    return y_final_reversed[::-1]


def cumtrapz(y, x=None, dx=1.0, axis=-1, initial=None):
    """
    使用 NumPy 实现的累积梯形积分。
    功能上与 scipy.integrate.cumulative_trapezoid 相似。

    参数:
    y (array_like): 要积分的数值。
    x (array_like, optional): 坐标。如果为 None，使用 dx 作为间距。
    dx (float, optional): x 为 None 时，y 元素之间的间距。
    axis (int, optional): 积分的轴向。默认为 -1 (最后一个轴)。
    initial (scalar, optional): 如果给定，将此值插入到结果的开头 (作为积分的起始点)。

    返回:
    ndarray: 累积积分结果。
    """

    # 确保 y 是 NumPy 数组
    y = np.asarray(y)

    # 1. 计算每个区间的宽度 (dx 或 np.diff(x))
    if x is not None:
        x = np.asarray(x)
        # 计算 x 轴上的间距
        d = np.diff(x, axis=axis)
    else:
        # 使用常量间距 dx
        d = dx

    # 2. 计算每个梯形的面积 (0.5 * (y[i] + y[i+1]) * d)

    # 沿着积分轴，从第二个元素开始切片 (y[i+1])
    slice1 = [slice(None)] * y.ndim
    slice1[axis] = slice(1, None)

    # 沿着积分轴，从第一个元素开始切片 (y[i])
    slice2 = [slice(None)] * y.ndim
    slice2[axis] = slice(None, -1)

    # 梯形面积数组
    trapezoid_areas = 0.5 * (y[tuple(slice1)] + y[tuple(slice2)]) * d

    # 3. 累积求和
    res = np.cumsum(trapezoid_areas, axis=axis)

    # 4. 处理 initial 参数 (将初始值插入到结果的开头)
    if initial is not None:
        # 为 initial 值创建一个占位符数组，其维度与其他轴相同，积分轴长度为 1
        shape = list(res.shape)
        shape[axis] = 1

        # 将 initial 值广播到占位符
        initial_array = np.full(shape, initial, dtype=res.dtype)

        # 沿着积分轴将 initial_array 和 res 拼接
        res = np.concatenate((initial_array, res), axis=axis)

    return res
