import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SimParams:
    gamma: float = 1.0          # linewidth
    alpha: float = 1.5          # coupling / optical-depth-like scale
    delta0: float = 0.0         # pulse carrier detuning from resonance
    sigma: float = 0.4          # spectral bandwidth
    n_freq: int = 2**14         # FFT grid size
    w_max: float = 25.0         # frequency window
    z: float = 1.0              # propagation depth / OD-like distance


def lorentzian_susceptibility(w: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    """
    Complex susceptibility-like response centered at resonance w = 0.
    """
    return alpha / (w + 1j * gamma)


def gaussian_spectrum(w: np.ndarray, delta0: float, sigma: float) -> np.ndarray:
    """
    Input pulse in frequency domain.
    """
    return np.exp(-0.5 * ((w - delta0) / sigma) ** 2)


def transmission_function(
    w: np.ndarray,
    gamma: float,
    alpha: float,
    z: float,
) -> np.ndarray:
    """
    Simple propagation model:
    H(w) = exp(i * chi(w) * z)

    Re[chi] -> phase dispersion
    Im[chi] -> attenuation / reshaping
    """
    chi = lorentzian_susceptibility(w, gamma, alpha)
    return np.exp(1j * chi * z)


def time_grid_from_freq(w: np.ndarray) -> np.ndarray:
    dw = w[1] - w[0]
    n = len(w)
    dt = 2 * np.pi / (n * dw)
    return (np.arange(n) - n // 2) * dt


def ifft_shifted(spec: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(spec)))


def compute_group_delay_from_phase(
    w: np.ndarray,
    H: np.ndarray,
    delta0: float,
) -> float:
    """
    Group delay from phase slope at the pulse carrier.
    tau_g = dphi/dw evaluated near delta0
    """
    phase = np.unwrap(np.angle(H))
    dphi_dw = np.gradient(phase, w)
    idx = np.argmin(np.abs(w - delta0))
    return float(dphi_dw[idx])


def compute_peak_delay(
    t: np.ndarray,
    e_in_t: np.ndarray,
    e_out_t: np.ndarray,
) -> float:
    """
    Delay from intensity centroids instead of raw peak indices.
    This avoids grid snapping artifacts.
    """
    i_in = np.abs(e_in_t) ** 2
    i_out = np.abs(e_out_t) ** 2

    norm_in = np.trapezoid(i_in, t)
    norm_out = np.trapezoid(i_out, t)
    if norm_in <= 1e-15 or norm_out <= 1e-15:
        return 0.0

    t_in = np.trapezoid(t * i_in, t) / norm_in
    t_out = np.trapezoid(t * i_out, t) / norm_out
    return float(t_out - t_in)


def medium_excitation_proxy(
    w: np.ndarray,
    e_in_w: np.ndarray,
    gamma: float,
    alpha: float,
) -> np.ndarray:
    """
    Positive absorptive weighting as a proxy for excitation participation.
    """
    chi = lorentzian_susceptibility(w, gamma, alpha)
    absorption = -np.imag(chi)
    return np.abs(e_in_w) ** 2 * absorption


def excitation_time_proxy(t: np.ndarray, excitation_t: np.ndarray) -> float:
    """
    Mean time of excitation participation.
    """
    weight = np.maximum(excitation_t, 0.0)
    norm = np.trapezoid(weight, t)
    if norm <= 1e-15:
        return 0.0
    return float(np.trapezoid(t * weight, t) / norm)


def run_single_sim(params: SimParams) -> Dict[str, float]:
    w = np.linspace(-params.w_max, params.w_max, params.n_freq)
    t = time_grid_from_freq(w)

    # Input pulse
    e_in_w = gaussian_spectrum(w, params.delta0, params.sigma)
    e_in_t = ifft_shifted(e_in_w)

    # Medium propagation
    H = transmission_function(w, params.gamma, params.alpha, params.z)
    e_out_w = e_in_w * H
    e_out_t = ifft_shifted(e_out_w)

    # Delays
    tau_g_phase = compute_group_delay_from_phase(w, H, params.delta0)
    tau_g_peak = compute_peak_delay(t, e_in_t, e_out_t)

    # Excitation proxy in time domain
    exc_w = medium_excitation_proxy(w, e_in_w, params.gamma, params.alpha)
    exc_t = np.abs(ifft_shifted(exc_w))
    tau_exc = excitation_time_proxy(t, exc_t)

    # Ratios and absolute comparisons
    if abs(tau_exc) > 1e-15:
        r_phase = tau_g_phase / tau_exc
        r_peak = tau_g_peak / tau_exc
        abs_diff_phase = abs(abs(tau_g_phase) - tau_exc)
        abs_diff_peak = abs(abs(tau_g_peak) - tau_exc)
    else:
        r_phase = np.nan
        r_peak = np.nan
        abs_diff_phase = np.nan
        abs_diff_peak = np.nan

    return {
        "delta0": params.delta0,
        "sigma": params.sigma,
        "alpha": params.alpha,
        "z": params.z,
        "tau_g_phase": tau_g_phase,
        "tau_g_peak": tau_g_peak,
        "tau_exc": tau_exc,
        "R_phase": r_phase,
        "R_peak": r_peak,
        "abs_diff_phase": abs_diff_phase,
        "abs_diff_peak": abs_diff_peak,
    }


def sweep_detuning(detunings: np.ndarray, base: SimParams) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    results: List[Dict[str, float]] = []
    for d in detunings:
        p = SimParams(
            gamma=base.gamma,
            alpha=base.alpha,
            delta0=float(d),
            sigma=base.sigma,
            n_freq=base.n_freq,
            w_max=base.w_max,
            z=base.z,
        )
        results.append(run_single_sim(p))
    return detunings, results


def main() -> None:
    base = SimParams(
        gamma=1.0,
        alpha=1.5,
        delta0=0.0,
        sigma=0.4,
        n_freq=2**14,
        w_max=25.0,
        z=1.0,
    )

    detunings = np.linspace(-4.0, 4.0, 25)
    _, results = sweep_detuning(detunings, base)

    print("delta\ttau_g_phase\ttau_g_peak\ttau_exc\tR_phase\tR_peak")
    for r in results:
        print(
            f"{r['delta0']:+.2f}\t"
            f"{r['tau_g_phase']:+.4f}\t"
            f"{r['tau_g_peak']:+.4f}\t"
            f"{r['tau_exc']:+.4f}\t"
            f"{r['R_phase']:+.4f}\t"
            f"{r['R_peak']:+.4f}"
        )


if __name__ == "__main__":
    main()
