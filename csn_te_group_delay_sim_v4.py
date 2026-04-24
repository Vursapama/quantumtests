import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List
import json


@dataclass
class SimParams:
    gamma: float = 1.0              # linewidth / decay rate
    alpha: float = 1.0              # coupling / optical depth scale
    delta0: float = 0.0             # pulse carrier detuning from resonance
    sigma: float = 0.5              # spectral bandwidth
    n_freq: int = 2**14             # FFT grid size
    w_max: float = 20.0             # frequency window
    z: float = 1.0                  # propagation depth / OD-like distance
    kappa: float = 1.0              # drive strength for excitation ODE
    # New CSN-relevant controls
    memory_decay: float = 0.0       # M: residual excitation carry between pulses (0 = none)
    n_pulses: int = 1               # H: number of repeated pulses in sequence
    pulse_spacing: float = 8.0      # H: spacing between repeated pulses (time units)
    observer_sigma_t: float = 0.0   # R: observer bandwidth / smoothing width in time domain
    support_threshold: float = 0.01 # R: threshold for meaningful support in centroid calculations


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


def fft_shifted(sig: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(sig)))


def compute_group_delay_from_phase(w: np.ndarray, H: np.ndarray, delta0: float) -> float:
    phase = np.unwrap(np.angle(H))
    dphi_dw = np.gradient(phase, w)
    idx = np.argmin(np.abs(w - delta0))
    return float(dphi_dw[idx])


def gaussian_kernel_t(t: np.ndarray, sigma_t: float) -> np.ndarray:
    if sigma_t <= 0:
        k = np.zeros_like(t)
        k[len(t)//2] = 1.0
        return k
    kernel = np.exp(-0.5 * (t / sigma_t) ** 2)
    s = np.sum(kernel)
    return kernel / s if s > 0 else kernel


def apply_observer_filter_t(signal_t: np.ndarray, t: np.ndarray, sigma_t: float) -> np.ndarray:
    """Apply a simple observer bandwidth filter in time domain."""
    if sigma_t <= 0:
        return signal_t.copy()
    k = gaussian_kernel_t(t, sigma_t)
    return np.convolve(signal_t, k, mode="same")


def compute_centroid_time(t: np.ndarray, weight: np.ndarray, support_threshold: float = 0.0) -> float:
    weight = np.maximum(np.asarray(weight), 0.0)
    if weight.size == 0 or np.max(weight) <= 1e-15:
        return 0.0

    if support_threshold > 0:
        mask = weight >= support_threshold * np.max(weight)
        if np.any(mask):
            t_use = t[mask]
            w_use = weight[mask]
        else:
            t_use = t
            w_use = weight
    else:
        t_use = t
        w_use = weight

    norm = np.trapezoid(w_use, t_use)
    if norm <= 1e-15:
        return 0.0
    return float(np.trapezoid(t_use * w_use, t_use) / norm)


def compute_peak_time(t: np.ndarray, weight: np.ndarray, support_threshold: float = 0.0) -> float:
    weight = np.maximum(np.asarray(weight), 0.0)
    if weight.size == 0 or np.max(weight) <= 1e-15:
        return 0.0

    if support_threshold > 0:
        mask = weight >= support_threshold * np.max(weight)
        idxs = np.where(mask)[0]
        if idxs.size > 0:
            local_idx = np.argmax(weight[idxs])
            return float(t[idxs[local_idx]])

    return float(t[np.argmax(weight)])


def compute_delay_metrics(
    t: np.ndarray,
    e_in_t: np.ndarray,
    e_out_t: np.ndarray,
    observer_sigma_t: float,
    support_threshold: float,
) -> Tuple[float, float, Dict[str, float]]:
    i_in = np.abs(e_in_t) ** 2
    i_out = np.abs(e_out_t) ** 2

    i_in_obs = apply_observer_filter_t(i_in, t, observer_sigma_t)
    i_out_obs = apply_observer_filter_t(i_out, t, observer_sigma_t)

    t_in_centroid = compute_centroid_time(t, i_in_obs, support_threshold)
    t_out_centroid = compute_centroid_time(t, i_out_obs, support_threshold)
    tau_centroid = t_out_centroid - t_in_centroid

    t_in_peak = compute_peak_time(t, i_in_obs, support_threshold)
    t_out_peak = compute_peak_time(t, i_out_obs, support_threshold)
    tau_peak = t_out_peak - t_in_peak

    diag = {
        "t_in_centroid": t_in_centroid,
        "t_out_centroid": t_out_centroid,
        "t_in_peak": t_in_peak,
        "t_out_peak": t_out_peak,
        "i_in_width": compute_width(t, i_in_obs, support_threshold),
        "i_out_width": compute_width(t, i_out_obs, support_threshold),
    }
    return float(tau_centroid), float(tau_peak), diag


def exact_atomic_step(a_prev: complex, e_drive: complex, dt: float, gamma: float, delta0: float, kappa: float) -> complex:
    """Exact one-step update for constant drive over dt."""
    lam = gamma + 1j * delta0
    decay = np.exp(-lam * dt)
    if abs(lam) < 1e-15:
        return a_prev + kappa * e_drive * dt
    drive = (kappa / lam) * (1.0 - decay) * e_drive
    return a_prev * decay + drive


def solve_atomic_excitation_sequence(
    t: np.ndarray,
    e_in_t: np.ndarray,
    gamma: float,
    delta0: float,
    kappa: float,
    memory_decay: float,
    n_pulses: int,
    pulse_spacing: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve repeated-pulse excitation with explicit memory carry.
    Returns:
      - total excitation profile over the full sequence
      - per-pulse excitation profile (same grid, accumulated)
    """
    dt = t[1] - t[0]
    n = len(t)

    total_exc = np.zeros(n, dtype=float)
    total_field = np.zeros(n, dtype=np.complex128)

    center_idx = n // 2
    spacing_idx = int(round(pulse_spacing / dt)) if pulse_spacing > 0 else n
    residual_a = 0.0 + 0.0j

    for p in range(max(1, n_pulses)):
        shift = (p - (max(1, n_pulses) - 1) / 2.0) * spacing_idx
        e_shift = np.roll(e_in_t, int(shift))
        a = np.zeros_like(e_shift, dtype=np.complex128)
        a[0] = residual_a

        for i in range(n - 1):
            a[i + 1] = exact_atomic_step(a[i], e_shift[i], dt, gamma, delta0, kappa)

        exc = np.abs(a) ** 2
        total_exc += exc
        total_field += a

        # carry final state into next pulse with memory decay
        if memory_decay > 0:
            residual_a = a[-1] * np.exp(-memory_decay * pulse_spacing)
        else:
            residual_a = 0.0 + 0.0j

    return total_exc, np.abs(total_field) ** 2


def compute_width(t: np.ndarray, weight: np.ndarray, support_threshold: float = 0.01) -> float:
    weight = np.maximum(np.asarray(weight), 0.0)
    if weight.size == 0 or np.max(weight) <= 1e-15:
        return 0.0
    mask = weight >= support_threshold * np.max(weight)
    idx = np.where(mask)[0]
    if idx.size < 2:
        return 0.0
    return float(t[idx[-1]] - t[idx[0]])


def excitation_shape_metrics(t: np.ndarray, exc_t: np.ndarray, support_threshold: float = 0.01) -> Dict[str, float]:
    weight = np.maximum(np.asarray(exc_t), 0.0)
    t_cent = compute_centroid_time(t, weight, support_threshold)
    t_peak = compute_peak_time(t, weight, support_threshold)
    width = compute_width(t, weight, support_threshold)

    if np.max(weight) <= 1e-15:
        return {
            "exc_peak": 0.0,
            "exc_width": 0.0,
            "exc_skew": 0.0,
            "exc_t_peak": 0.0,
            "exc_t_centroid": 0.0,
        }

    mask = weight >= support_threshold * np.max(weight)
    t_use = t[mask] if np.any(mask) else t
    w_use = weight[mask] if np.any(mask) else weight
    norm = np.trapezoid(w_use, t_use)

    if norm <= 1e-15:
        skew = 0.0
    else:
        mu = np.trapezoid(t_use * w_use, t_use) / norm
        var = np.trapezoid(((t_use - mu) ** 2) * w_use, t_use) / norm
        if var <= 1e-15:
            skew = 0.0
        else:
            third = np.trapezoid(((t_use - mu) ** 3) * w_use, t_use) / norm
            skew = float(third / (var ** 1.5))

    return {
        "exc_peak": float(np.max(weight)),
        "exc_width": width,
        "exc_skew": skew,
        "exc_t_peak": t_peak,
        "exc_t_centroid": t_cent,
    }


def safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-12 else np.nan


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
    tau_g_centroid, tau_g_peak, delay_diag = compute_delay_metrics(
        t, e_in_t, e_out_t, params.observer_sigma_t, params.support_threshold
    )

    # Excitation as part of the same interaction event, with explicit memory/history
    exc_t_sum, exc_field_sum = solve_atomic_excitation_sequence(
        t=t,
        e_in_t=e_in_t,
        gamma=params.gamma,
        delta0=params.delta0,
        kappa=params.kappa,
        memory_decay=params.memory_decay,
        n_pulses=params.n_pulses,
        pulse_spacing=params.pulse_spacing,
    )

    # Observer-limited readout of excitation
    exc_obs = apply_observer_filter_t(exc_t_sum, t, params.observer_sigma_t)

    # Relative excitation time against input centroid, preserving difference rather than collapsing it away
    i_in = apply_observer_filter_t(np.abs(e_in_t) ** 2, t, params.observer_sigma_t)
    t_in = compute_centroid_time(t, i_in, params.support_threshold)
    tau_exc_abs = compute_centroid_time(t, exc_obs, params.support_threshold)
    tau_exc = tau_exc_abs - t_in

    # Shape diagnostics
    exc_shape = excitation_shape_metrics(t, exc_obs, params.support_threshold)

    # Ratios and absolute-difference tests
    r_phase = safe_div(tau_g_phase, tau_exc)
    r_centroid = safe_div(tau_g_centroid, tau_exc)
    r_peak = safe_div(tau_g_peak, tau_exc)

    result = {
        "delta0": params.delta0,
        "sigma": params.sigma,
        "alpha": params.alpha,
        "z": params.z,
        "kappa": params.kappa,
        "memory_decay": params.memory_decay,
        "n_pulses": params.n_pulses,
        "pulse_spacing": params.pulse_spacing,
        "observer_sigma_t": params.observer_sigma_t,
        "tau_g_phase": tau_g_phase,
        "tau_g_centroid": tau_g_centroid,
        "tau_g_peak": tau_g_peak,
        "tau_exc": tau_exc,
        "R_phase": r_phase,
        "R_centroid": r_centroid,
        "R_peak": r_peak,
        "abs_diff_phase": abs(abs(tau_g_phase) - abs(tau_exc)),
        "abs_diff_centroid": abs(abs(tau_g_centroid) - abs(tau_exc)),
        "abs_diff_peak": abs(abs(tau_g_peak) - abs(tau_exc)),
    }
    result.update(delay_diag)
    result.update(exc_shape)
    return result


def sweep_detuning(detunings: np.ndarray, base: SimParams) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    results: List[Dict[str, float]] = []
    for d in detunings:
        p = SimParams(**{**asdict(base), "delta0": float(d)})
        results.append(run_single_sim(p))
    return detunings, results


def summarize_results(results: List[Dict[str, float]]) -> Dict[str, float]:
    def finite_vals(key: str) -> np.ndarray:
        vals = np.array([r[key] for r in results], dtype=float)
        return vals[np.isfinite(vals)]

    phase_ratio = np.abs(finite_vals("R_phase"))
    centroid_ratio = np.abs(finite_vals("R_centroid"))
    peak_ratio = np.abs(finite_vals("R_peak"))

    summary = {
        "n_valid": int(len(results)),
        "phase_abs_ratio_mean": float(np.mean(phase_ratio)) if phase_ratio.size else np.nan,
        "phase_abs_ratio_std": float(np.std(phase_ratio)) if phase_ratio.size else np.nan,
        "centroid_abs_ratio_mean": float(np.mean(centroid_ratio)) if centroid_ratio.size else np.nan,
        "centroid_abs_ratio_std": float(np.std(centroid_ratio)) if centroid_ratio.size else np.nan,
        "peak_abs_ratio_mean": float(np.mean(peak_ratio)) if peak_ratio.size else np.nan,
        "peak_abs_ratio_std": float(np.std(peak_ratio)) if peak_ratio.size else np.nan,
        "phase_abs_diff_mean": float(np.mean(finite_vals("abs_diff_phase"))),
        "centroid_abs_diff_mean": float(np.mean(finite_vals("abs_diff_centroid"))),
        "peak_abs_diff_mean": float(np.mean(finite_vals("abs_diff_peak"))),
        "exc_width_mean": float(np.mean(finite_vals("exc_width"))),
        "exc_skew_mean": float(np.mean(finite_vals("exc_skew"))),
        "i_out_width_mean": float(np.mean(finite_vals("i_out_width"))),
    }
    return summary


def print_results_table(results: List[Dict[str, float]]) -> None:
    print("delta\ttau_g_phase\ttau_g_centroid\ttau_g_peak\ttau_exc\tR_phase\tR_centroid\tR_peak")
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


def main() -> None:
    base = SimParams(
        gamma=1.0,
        alpha=1.5,
        delta0=0.0,
        sigma=0.4,
        n_freq=2**14,
        w_max=25.0,
        z=1.0,
        kappa=2.0,
        memory_decay=0.15,
        n_pulses=1,
        pulse_spacing=8.0,
        observer_sigma_t=0.0,
        support_threshold=0.01,
    )

    detunings = np.linspace(-4.0, 4.0, 25)
    _, results = sweep_detuning(detunings, base)

    print_results_table(results)
    print("\nsummary:")
    print(json.dumps(summarize_results(results), indent=2))


if __name__ == "__main__":
    main()
