import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SimParams:
    gamma: float = 1.0              # linewidth / decay rate
    alpha: float = 1.5              # coupling / optical depth scale
    delta0: float = 0.0             # pulse carrier detuning from resonance
    sigma: float = 0.4              # spectral bandwidth
    n_freq: int = 2**15             # FFT grid size
    w_max: float = 25.0             # frequency window
    z: float = 1.0                  # propagation depth / OD-like distance
    kappa: float = 2.0              # drive strength for excitation ODE
    support_frac: float = 0.01      # support threshold for excitation timing
    support_min_pts: int = 32        # minimum number of support points
    use_output_drive: bool = False   # whether to drive excitation with transmitted pulse instead of input pulse


def lorentzian_susceptibility(w: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    """Resonance centered at w=0; pulse detuning is applied by shifting the input pulse center."""
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


def centroid_time(t: np.ndarray, weight: np.ndarray) -> float:
    norm = np.trapezoid(weight, t)
    if norm <= 1e-15:
        return 0.0
    return float(np.trapezoid(t * weight, t) / norm)


def second_moment_width(t: np.ndarray, weight: np.ndarray, center: float | None = None) -> float:
    norm = np.trapezoid(weight, t)
    if norm <= 1e-15:
        return 0.0
    mu = centroid_time(t, weight) if center is None else center
    var = np.trapezoid(((t - mu) ** 2) * weight, t) / norm
    return float(np.sqrt(max(var, 0.0)))


def compute_group_delay_from_phase(w: np.ndarray, H: np.ndarray, delta0: float) -> float:
    phase = np.unwrap(np.angle(H))
    dphi_dw = np.gradient(phase, w)
    idx = np.argmin(np.abs(w - delta0))
    return float(dphi_dw[idx])


def compute_centroid_delay(t: np.ndarray, e_in_t: np.ndarray, e_out_t: np.ndarray) -> float:
    i_in = np.abs(e_in_t) ** 2
    i_out = np.abs(e_out_t) ** 2
    return centroid_time(t, i_out) - centroid_time(t, i_in)


def compute_peak_delay_parabolic(t: np.ndarray, e_in_t: np.ndarray, e_out_t: np.ndarray) -> float:
    def refined_peak(tt: np.ndarray, ii: np.ndarray) -> float:
        idx = int(np.argmax(ii))
        if idx == 0 or idx == len(ii) - 1:
            return float(tt[idx])
        y0, y1, y2 = ii[idx - 1], ii[idx], ii[idx + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) < 1e-15:
            return float(tt[idx])
        frac = 0.5 * (y0 - y2) / denom
        return float(tt[idx] + frac * (tt[1] - tt[0]))

    i_in = np.abs(e_in_t) ** 2
    i_out = np.abs(e_out_t) ** 2
    return refined_peak(t, i_out) - refined_peak(t, i_in)


def support_window_indices(weight: np.ndarray, frac: float, min_pts: int) -> np.ndarray:
    peak = float(np.max(weight))
    if peak <= 1e-15:
        return np.zeros_like(weight, dtype=bool)
    mask = weight >= frac * peak
    idx = np.where(mask)[0]
    if len(idx) == 0:
        center = int(np.argmax(weight))
        lo = max(0, center - min_pts // 2)
        hi = min(len(weight), lo + min_pts)
        out = np.zeros_like(weight, dtype=bool)
        out[lo:hi] = True
        return out
    lo = max(0, idx[0] - min_pts // 2)
    hi = min(len(weight), idx[-1] + min_pts // 2 + 1)
    out = np.zeros_like(weight, dtype=bool)
    out[lo:hi] = True
    return out


def solve_atomic_excitation(t: np.ndarray, drive_t: np.ndarray, gamma: float, delta0: float, kappa: float) -> np.ndarray:
    """
    Exact-step update for the linear ODE
        da/dt = -(gamma + i*delta0) a + kappa * E(t)
    using a piecewise-constant drive over each dt.
    """
    dt = t[1] - t[0]
    lam = gamma + 1j * delta0
    decay = np.exp(-lam * dt)
    # limit for small |lam| isn't needed here since gamma > 0 in all runs
    drive_gain = (kappa / lam) * (1 - decay)

    a = np.zeros_like(drive_t, dtype=np.complex128)
    for i in range(len(t) - 1):
        a[i + 1] = a[i] * decay + drive_gain * drive_t[i]
    return np.abs(a) ** 2


def excitation_time_proxy(t: np.ndarray, excitation_t: np.ndarray, support_frac: float, support_min_pts: int) -> Tuple[float, Dict[str, float]]:
    weight = np.maximum(excitation_t, 0.0)
    mask = support_window_indices(weight, support_frac, support_min_pts)
    if not np.any(mask):
        return 0.0, {"support_norm": 0.0, "support_width": 0.0}
    t_win = t[mask]
    w_win = weight[mask]
    mu = centroid_time(t_win, w_win)
    width = second_moment_width(t_win, w_win, mu)
    norm = float(np.trapezoid(w_win, t_win))
    return mu, {"support_norm": norm, "support_width": width}


def run_single_sim(params: SimParams) -> Dict[str, float]:
    w = np.linspace(-params.w_max, params.w_max, params.n_freq)
    t = time_grid_from_freq(w)

    # Input pulse
    e_in_w = gaussian_spectrum(w, params.delta0, params.sigma)
    e_in_t = ifft_shifted(e_in_w)
    i_in = np.abs(e_in_t) ** 2
    t_in = centroid_time(t, i_in)
    w_in = second_moment_width(t, i_in, t_in)

    # Medium transmission and output pulse
    H = transmission_function(w, params.gamma, params.alpha, params.z)
    e_out_w = e_in_w * H
    e_out_t = ifft_shifted(e_out_w)
    i_out = np.abs(e_out_t) ** 2
    t_out = centroid_time(t, i_out)
    w_out = second_moment_width(t, i_out, t_out)

    # Delay metrics
    tau_g_phase = compute_group_delay_from_phase(w, H, params.delta0)
    tau_g_centroid = compute_centroid_delay(t, e_in_t, e_out_t)
    tau_g_peak = compute_peak_delay_parabolic(t, e_in_t, e_out_t)

    # Excitation solved as part of the same interaction event
    drive_t = e_out_t if params.use_output_drive else e_in_t
    exc_t = solve_atomic_excitation(t, drive_t, params.gamma, params.delta0, params.kappa)
    tau_exc_abs, exc_diag = excitation_time_proxy(t, exc_t, params.support_frac, params.support_min_pts)
    tau_exc_rel = tau_exc_abs - t_in

    def safe_div(num: float, den: float) -> float:
        return float(num / den) if abs(den) > 1e-12 else np.nan

    return {
        "delta0": float(params.delta0),
        "sigma": float(params.sigma),
        "alpha": float(params.alpha),
        "z": float(params.z),
        "gamma": float(params.gamma),
        "tau_g_phase": float(tau_g_phase),
        "tau_g_centroid": float(tau_g_centroid),
        "tau_g_peak": float(tau_g_peak),
        "tau_exc": float(tau_exc_rel),
        "R_phase": safe_div(tau_g_phase, tau_exc_rel),
        "R_centroid": safe_div(tau_g_centroid, tau_exc_rel),
        "R_peak": safe_div(tau_g_peak, tau_exc_rel),
        "abs_diff_phase": float(abs(abs(tau_g_phase) - abs(tau_exc_rel))),
        "abs_diff_centroid": float(abs(abs(tau_g_centroid) - abs(tau_exc_rel))),
        "abs_diff_peak": float(abs(abs(tau_g_peak) - abs(tau_exc_rel))),
        "input_center": float(t_in),
        "output_center": float(t_out),
        "input_width": float(w_in),
        "output_width": float(w_out),
        "exc_support_norm": float(exc_diag["support_norm"]),
        "exc_support_width": float(exc_diag["support_width"]),
    }


def sweep_detuning(detunings: np.ndarray, base: SimParams) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    results: List[Dict[str, float]] = []
    for d in detunings:
        p = SimParams(**{**asdict(base), "delta0": float(d)})
        results.append(run_single_sim(p))
    return detunings, results


def summarize_results(results: List[Dict[str, float]]) -> Dict[str, float]:
    valid = [r for r in results if np.isfinite(r["R_phase"]) and abs(r["tau_exc"]) > 1e-10]
    if not valid:
        return {}
    phase_abs = np.array([abs(r["R_phase"]) for r in valid])
    cent_abs = np.array([abs(r["R_centroid"]) for r in valid])
    peak_abs = np.array([abs(r["R_peak"]) for r in valid])
    return {
        "n_valid": len(valid),
        "phase_abs_ratio_mean": float(np.mean(phase_abs)),
        "phase_abs_ratio_std": float(np.std(phase_abs)),
        "centroid_abs_ratio_mean": float(np.mean(cent_abs)),
        "centroid_abs_ratio_std": float(np.std(cent_abs)),
        "peak_abs_ratio_mean": float(np.mean(peak_abs)),
        "peak_abs_ratio_std": float(np.std(peak_abs)),
        "phase_abs_diff_mean": float(np.mean([r["abs_diff_phase"] for r in valid])),
        "centroid_abs_diff_mean": float(np.mean([r["abs_diff_centroid"] for r in valid])),
        "peak_abs_diff_mean": float(np.mean([r["abs_diff_peak"] for r in valid])),
    }


def print_results_table(results: List[Dict[str, float]]) -> None:
    print(
        "delta\ttau_g_phase\ttau_g_centroid\ttau_g_peak\ttau_exc\tR_phase\tR_centroid\tR_peak"
    )
    for r in results:
        print(
            f"{r['delta0']:+.2f}\t"
            f"{r['tau_g_phase']:+.4f}\t"
            f"{r['tau_g_centroid']:+.4f}\t"
            f"{r['tau_g_peak']:+.4f}\t"
            f"{r['tau_exc']:+.4f}\t"
            f"{r['R_phase']:+.4f}\t"
            f"{r['R_centroid']:+.4f}\t"
            f"{r['R_peak']:+.4f}"
        )


if __name__ == "__main__":
    base = SimParams()
    detunings = np.linspace(-4.0, 4.0, 25)
    _, results = sweep_detuning(detunings, base)
    print_results_table(results)
    print("\nsummary:")
    print(json.dumps(summarize_results(results), indent=2))
