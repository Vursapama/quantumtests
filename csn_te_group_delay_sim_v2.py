import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class SimParams:
    gamma: float = 1.0          # linewidth / decay rate
    alpha: float = 1.0          # coupling / optical depth scale
    delta0: float = 0.0         # pulse carrier detuning from resonance
    sigma: float = 0.5          # spectral bandwidth
    n_freq: int = 2**14         # FFT grid size
    w_max: float = 20.0         # frequency window
    z: float = 1.0              # propagation depth / OD-like distance
    kappa: float = 1.0          # drive strength for excitation ODE


def lorentzian_susceptibility(w: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    """
    Resonance centered at w=0. Pulse detuning is applied by shifting the input pulse center.
    """
    return alpha / (w + 1j * gamma)


def gaussian_spectrum(w: np.ndarray, delta0: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((w - delta0) / sigma) ** 2)


def transmission_function(w: np.ndarray, gamma: float, alpha: float, z: float) -> np.ndarray:
    """
    H(w) = exp(i * chi(w) * z)
    Re[chi] -> phase dispersion
    Im[chi] -> attenuation / amplitude shaping
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


def compute_group_delay_from_phase(w: np.ndarray, H: np.ndarray, delta0: float) -> float:
    phase = np.unwrap(np.angle(H))
    dphi_dw = np.gradient(phase, w)
    idx = np.argmin(np.abs(w - delta0))
    return float(dphi_dw[idx])


def compute_centroid_delay(t: np.ndarray, e_in_t: np.ndarray, e_out_t: np.ndarray) -> float:
    i_in = np.abs(e_in_t) ** 2
    i_out = np.abs(e_out_t) ** 2
    in_norm = np.trapezoid(i_in, t)
    out_norm = np.trapezoid(i_out, t)
    if in_norm <= 1e-15 or out_norm <= 1e-15:
        return 0.0
    t_in = np.trapezoid(t * i_in, t) / in_norm
    t_out = np.trapezoid(t * i_out, t) / out_norm
    return float(t_out - t_in)


def solve_atomic_excitation(t: np.ndarray, e_in_t: np.ndarray, gamma: float, delta0: float, kappa: float) -> np.ndarray:
    """
    Time-domain driven response:
        da/dt = -(gamma + i*delta0) a + kappa * E_in(t)
    We integrate forward in time to get the excitation amplitude a(t),
    then use |a(t)|^2 as the excitation participation profile.
    """
    dt = t[1] - t[0]
    a = np.zeros_like(e_in_t, dtype=np.complex128)
    # forward Euler is sufficient for this exploratory solver if dt is small.
    for i in range(len(t) - 1):
        da = (-(gamma + 1j * delta0) * a[i] + kappa * e_in_t[i]) * dt
        a[i + 1] = a[i] + da
    return np.abs(a) ** 2


def excitation_time_proxy(t: np.ndarray, excitation_t: np.ndarray) -> float:
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

    # Medium transmission and output pulse
    H = transmission_function(w, params.gamma, params.alpha, params.z)
    e_out_w = e_in_w * H
    e_out_t = ifft_shifted(e_out_w)

    # Delay metrics
    tau_g_phase = compute_group_delay_from_phase(w, H, params.delta0)
    tau_g_peak = compute_centroid_delay(t, e_in_t, e_out_t)

    # Excitation as part of the same interaction event
    exc_t = solve_atomic_excitation(t, e_in_t, params.gamma, params.delta0, params.kappa)
    tau_exc = excitation_time_proxy(t, exc_t)

    # Center the excitation time relative to the input pulse centroid for fair comparison
    i_in = np.abs(e_in_t) ** 2
    in_norm = np.trapezoid(i_in, t)
    t_in = np.trapezoid(t * i_in, t) / in_norm if in_norm > 1e-15 else 0.0
    tau_exc_rel = tau_exc - t_in

    # Ratios and absolute-difference tests
    def safe_div(num: float, den: float) -> float:
        return float(num / den) if abs(den) > 1e-12 else np.nan

    r_phase = safe_div(tau_g_phase, tau_exc_rel)
    r_peak = safe_div(tau_g_peak, tau_exc_rel)

    return {
        "delta0": params.delta0,
        "sigma": params.sigma,
        "alpha": params.alpha,
        "z": params.z,
        "tau_g_phase": tau_g_phase,
        "tau_g_peak": tau_g_peak,
        "tau_exc": tau_exc_rel,
        "R_phase": r_phase,
        "R_peak": r_peak,
        "abs_diff_phase": abs(abs(tau_g_phase) - abs(tau_exc_rel)),
        "abs_diff_peak": abs(abs(tau_g_peak) - abs(tau_exc_rel)),
    }


def sweep_detuning(detunings: np.ndarray, base: SimParams) -> Tuple[np.ndarray, list]:
    results = []
    for d in detunings:
        p = SimParams(
            gamma=base.gamma,
            alpha=base.alpha,
            delta0=float(d),
            sigma=base.sigma,
            n_freq=base.n_freq,
            w_max=base.w_max,
            z=base.z,
            kappa=base.kappa,
        )
        results.append(run_single_sim(p))
    return detunings, results


if __name__ == "__main__":
    base = SimParams(
        gamma=1.0,
        alpha=1.5,
        delta0=0.0,
        sigma=0.4,
        n_freq=2**14,
        w_max=25.0,
        z=1.0,
        kappa=2.0,
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
