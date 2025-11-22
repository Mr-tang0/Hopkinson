import numpy as np
from scipy.fft import fft, ifft, fftfreq


def bancroft_coefficients(poisson_ratio: float):
    nu = poisson_ratio
    if not (0 <= nu < 0.5):
        raise ValueError("Poisson's ratio must be in [0, 0.5)")

    A = (1 - 2 * nu) / (2 * (1 - nu))
    B = -((1 - 2 * nu) ** 2) / (8 * (1 - nu) ** 2)
    return A, B


def phase_velocity_from_coeffs(freq: np.ndarray, a: float, c0: float, A: float, B: float):
    omega = 2 * np.pi * freq
    # Avoid division by zero; compute k0 = omega / c0 first
    k0a = (omega * a) / c0  # dimensionless

    # Use series expansion: (cp/c0)^2 ≈ 1 + A*(k0a)^2 + B*(k0a)^4
    ratio_sq = 1.0 + A * (k0a ** 2) + B * (k0a ** 4)

    # Ensure non-negative (numerical safety)
    ratio_sq = np.maximum(ratio_sq, 1e-6)

    cp = c0 * np.sqrt(ratio_sq)

    # Enforce cp = c0 at f = 0 (though math should already give this)
    cp = np.where(freq == 0, c0, cp)
    return cp


def correct_dispersion(
        signal: np.ndarray,  # 信号
        dt: float,  # dt
        distance_to_specimen: float,
        bar_diameter: float,
        sound_velocity: float,  # c0
        poisson_ratio: float,
        damping_coeff: float = 0.0,  # alpha, optional
) -> np.ndarray:

    signal = np.asarray(signal, dtype=np.float64)
    L = float(distance_to_specimen)
    a = float(bar_diameter) / 2.0
    c0 = float(sound_velocity)
    nu = float(poisson_ratio)
    alpha = float(damping_coeff)
    dt = float(dt)

    if L < 0:
        raise ValueError("Distance to specimen must be non-negative.")
    if a <= 0:
        raise ValueError("Bar diameter must be positive.")
    if c0 <= 0:
        raise ValueError("Sound velocity must be positive.")
    if not (0 <= nu < 0.5):
        raise ValueError("Poisson's ratio must be in [0, 0.5).")

    n = len(signal)
    if n == 0:
        return np.array([])

    # Step 1: FFT
    Y = fft(signal)

    # Step 2: Frequency vector (includes negative frequencies)
    freqs = fftfreq(n, d=dt)  # shape (n,)

    # Step 3: Compute phase velocity c_p(f) for all frequencies
    A, B = bancroft_coefficients(nu)
    cp = phase_velocity_from_coeffs(np.abs(freqs), a, c0, A, B)  # c_p is even function

    # Step 4: Compute phase correction Δφ = 2πf * L * (1/c0 - 1/cp)
    # Note: sign convention — we are "back-propagating" the wave to the specimen,
    # so we add the phase that was accumulated during forward propagation.
    delta_phi = 2 * np.pi * freqs * L * (1.0 / c0 - 1.0 / cp)

    # Step 5: Apply phase shift in frequency domain
    Y_corrected = Y * np.exp(1j * delta_phi)

    # Step 6: Inverse FFT
    signal_time = np.real(ifft(Y_corrected))

    # Step 7: Damping correction — amplify to compensate for exp(-αL) loss
    gain = np.exp(alpha * L)
    signal_corrected = signal_time * gain

    return signal_corrected
