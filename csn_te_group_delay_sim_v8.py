import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from pathlib import Path


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
    memory_decay: float = 0.15  # TE memory relaxation constant
    memory_gain: float = 1.0    # TE write strength into memory
    te_delta_feedback: float = 0.25  # TE memory -> effective detuning
    te_kappa_feedback: float = 0.25  # TE memory -> effective coupling
    observer_sigma: float = 1.0      # observer / detector smoothing width in samples


def lorentzian_susceptibility(w: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    """Resonance centered at w=0. Pulse detuning is applied by shifting the input pulse center."""
    return alpha / (w + 1j * gamma)


def gaussian_spectrum(w: np.ndarray, delta0: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((w - delta0) / sigma) ** 2)


def transmission_function(w: np.ndarray, gamma: float, alpha: float, z: float) -> np.ndarray:
    """H(w) = exp(i * chi(w) * z)."""
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


def gaussian_kernel_sigma_samples(sigma_samples: float, radius_factor: float = 4.0) -> np.ndarray:
    sigma_samples = max(float(sigma_samples), 1e-9)
    radius = max(1, int(np.ceil(radius_factor * sigma_samples)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma_samples) ** 2)
    k /= np.sum(k)
    return k


def observer_filter(signal: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 1e-9:
        return signal
    kernel = gaussian_kernel_sigma_samples(sigma_samples)
    return np.convolve(signal, kernel, mode="same")


def compute_centroid_delay(t: np.ndarray, e_in_t: np.ndarray, e_out_t: np.ndarray, observer_sigma: float) -> float:
    i_in = np.abs(e_in_t) ** 2
    i_out = np.abs(e_out_t) ** 2
    i_in_f = observer_filter(i_in, observer_sigma)
    i_out_f = observer_filter(i_out, observer_sigma)

    in_norm = np.trapezoid(i_in_f, t)
    out_norm = np.trapezoid(i_out_f, t)
    if in_norm <= 1e-15 or out_norm <= 1e-15:
        return 0.0
    t_in = np.trapezoid(t * i_in_f, t) / in_norm
    t_out = np.trapezoid(t * i_out_f, t) / out_norm
    return float(t_out - t_in)


def compute_peak_delay(t: np.ndarray, e_in_t: np.ndarray, e_out_t: np.ndarray, observer_sigma: float) -> float:
    i_in = observer_filter(np.abs(e_in_t) ** 2, observer_sigma)
    i_out = observer_filter(np.abs(e_out_t) ** 2, observer_sigma)
    return float(t[np.argmax(i_out)] - t[np.argmax(i_in)])


def solve_atomic_excitation(
    t: np.ndarray,
    e_in_t: np.ndarray,
    gamma: float,
    delta0: float,
    kappa: float,
    memory_decay: float,
    memory_gain: float,
    te_delta_feedback: float,
    te_kappa_feedback: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-domain driven response with explicit TE-memory state.

    a(t): excitation amplitude
    TE_current(t): instantaneous mismatch between input and excitation
    TE_memory(t): decaying accumulated TE history
    """
    dt = t[1] - t[0]
    a = np.zeros_like(e_in_t, dtype=np.complex128)
    te_current = np.zeros_like(t, dtype=np.float64)
    te_memory = np.zeros_like(t, dtype=np.float64)

    for i in range(len(t) - 1):
        eff_delta = delta0 + te_delta_feedback * te_memory[i]
        eff_kappa = kappa * (1.0 + te_kappa_feedback * te_memory[i])
        lam = gamma + 1j * eff_delta

        decay = np.exp(-lam * dt)
        drive = (eff_kappa / lam) * (1.0 - decay) * e_in_t[i]
        a[i + 1] = a[i] * decay + drive

        te_current[i] = abs(e_in_t[i] - a[i])
        te_memory[i + 1] = (1.0 - memory_decay) * te_memory[i] + memory_gain * te_current[i] * abs(dt)

    te_current[-1] = abs(e_in_t[-1] - a[-1])
    excitation = np.abs(a) ** 2
    return excitation, te_current, te_memory


def support_window_mask(weight: np.ndarray, frac: float = 0.01) -> np.ndarray:
    peak = float(np.max(weight)) if weight.size else 0.0
    if peak <= 1e-15:
        return np.zeros_like(weight, dtype=bool)
    return weight >= (frac * peak)


def excitation_time_proxy(t: np.ndarray, excitation_t: np.ndarray, support_frac: float = 0.01) -> float:
    weight = np.maximum(excitation_t, 0.0)
    mask = support_window_mask(weight, support_frac)
    if not np.any(mask):
        return 0.0

    t_win = t[mask]
    w_win = weight[mask]
    norm = np.trapezoid(w_win, t_win)
    if norm <= 1e-15:
        return 0.0
    return float(np.trapezoid(t_win * w_win, t_win) / norm)


def weighted_width(t: np.ndarray, weight: np.ndarray, support_frac: float = 0.01) -> float:
    w = np.maximum(weight, 0.0)
    mask = support_window_mask(w, support_frac)
    if not np.any(mask):
        return 0.0
    t_win = t[mask]
    w_win = w[mask]
    norm = np.trapezoid(w_win, t_win)
    if norm <= 1e-15:
        return 0.0
    mu = np.trapezoid(t_win * w_win, t_win) / norm
    var = np.trapezoid(((t_win - mu) ** 2) * w_win, t_win) / norm
    return float(np.sqrt(max(var, 0.0)))


def weighted_skew(t: np.ndarray, weight: np.ndarray, support_frac: float = 0.01) -> float:
    w = np.maximum(weight, 0.0)
    mask = support_window_mask(w, support_frac)
    if not np.any(mask):
        return 0.0
    t_win = t[mask]
    w_win = w[mask]
    norm = np.trapezoid(w_win, t_win)
    if norm <= 1e-15:
        return 0.0
    mu = np.trapezoid(t_win * w_win, t_win) / norm
    var = np.trapezoid(((t_win - mu) ** 2) * w_win, t_win) / norm
    sigma = np.sqrt(max(var, 0.0))
    if sigma <= 1e-15:
        return 0.0
    m3 = np.trapezoid(((t_win - mu) ** 3) * w_win, t_win) / norm
    return float(m3 / (sigma ** 3))


def run_single_sim(params: SimParams) -> Dict[str, float]:
    w = np.linspace(-params.w_max, params.w_max, params.n_freq)
    t = time_grid_from_freq(w)

    e_in_w = gaussian_spectrum(w, params.delta0, params.sigma)
    e_in_t = ifft_shifted(e_in_w)

    H = transmission_function(w, params.gamma, params.alpha, params.z)
    e_out_w = e_in_w * H
    e_out_t = ifft_shifted(e_out_w)

    tau_g_phase = compute_group_delay_from_phase(w, H, params.delta0)
    tau_g_centroid = compute_centroid_delay(t, e_in_t, e_out_t, params.observer_sigma)
    tau_g_peak = compute_peak_delay(t, e_in_t, e_out_t, params.observer_sigma)

    exc_t, te_current, te_memory = solve_atomic_excitation(
        t,
        e_in_t,
        params.gamma,
        params.delta0,
        params.kappa,
        params.memory_decay,
        params.memory_gain,
        params.te_delta_feedback,
        params.te_kappa_feedback,
    )

    tau_exc = excitation_time_proxy(t, exc_t)
    i_in = np.abs(e_in_t) ** 2
    in_norm = np.trapezoid(i_in, t)
    t_in = np.trapezoid(t * i_in, t) / in_norm if in_norm > 1e-15 else 0.0
    tau_exc_rel = tau_exc - t_in

    def safe_div(num: float, den: float) -> float:
        return float(num / den) if abs(den) > 1e-12 else np.nan

    return {
        "delta0": params.delta0,
        "sigma": params.sigma,
        "alpha": params.alpha,
        "z": params.z,
        "memory_decay": params.memory_decay,
        "memory_gain": params.memory_gain,
        "tau_g_phase": tau_g_phase,
        "tau_g_centroid": tau_g_centroid,
        "tau_g_peak": tau_g_peak,
        "tau_exc": tau_exc_rel,
        "R_phase": safe_div(tau_g_phase, tau_exc_rel),
        "R_centroid": safe_div(tau_g_centroid, tau_exc_rel),
        "R_peak": safe_div(tau_g_peak, tau_exc_rel),
        "abs_diff_phase": abs(abs(tau_g_phase) - abs(tau_exc_rel)),
        "abs_diff_centroid": abs(abs(tau_g_centroid) - abs(tau_exc_rel)),
        "abs_diff_peak": abs(abs(tau_g_peak) - abs(tau_exc_rel)),
        "exc_width": weighted_width(t, exc_t),
        "exc_skew": weighted_skew(t, exc_t),
        "i_out_width": weighted_width(t, np.abs(e_out_t) ** 2),
        "te_current_peak": float(np.max(te_current)),
        "te_memory_peak": float(np.max(te_memory)),
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
            kappa=base.kappa,
            memory_decay=base.memory_decay,
            memory_gain=base.memory_gain,
            te_delta_feedback=base.te_delta_feedback,
            te_kappa_feedback=base.te_kappa_feedback,
            observer_sigma=base.observer_sigma,
        )
        results.append(run_single_sim(p))
    return detunings, results


def summarize_results(results: List[Dict[str, float]]) -> Dict[str, float]:
    valid = [r for r in results if not np.isnan(r["R_phase"]) and not np.isnan(r["R_centroid"]) and not np.isnan(r["R_peak"])]
    if not valid:
        return {"n_valid": 0}

    def arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in valid], dtype=float)

    phase_abs_ratio = np.abs(arr("R_phase"))
    centroid_abs_ratio = np.abs(arr("R_centroid"))
    peak_abs_ratio = np.abs(arr("R_peak"))

    return {
        "n_valid": int(len(valid)),
        "phase_abs_ratio_mean": float(np.mean(phase_abs_ratio)),
        "phase_abs_ratio_std": float(np.std(phase_abs_ratio)),
        "centroid_abs_ratio_mean": float(np.mean(centroid_abs_ratio)),
        "centroid_abs_ratio_std": float(np.std(centroid_abs_ratio)),
        "peak_abs_ratio_mean": float(np.mean(peak_abs_ratio)),
        "peak_abs_ratio_std": float(np.std(peak_abs_ratio)),
        "phase_abs_diff_mean": float(np.mean(arr("abs_diff_phase"))),
        "centroid_abs_diff_mean": float(np.mean(arr("abs_diff_centroid"))),
        "peak_abs_diff_mean": float(np.mean(arr("abs_diff_peak"))),
        "exc_width_mean": float(np.mean(arr("exc_width"))),
        "exc_skew_mean": float(np.mean(arr("exc_skew"))),
        "i_out_width_mean": float(np.mean(arr("i_out_width"))),
        "te_current_peak_mean": float(np.mean(arr("te_current_peak"))),
        "te_memory_peak_mean": float(np.mean(arr("te_memory_peak"))),
    }


def print_single_sweep(results: List[Dict[str, float]]) -> None:
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


def sweep_memory_params(base: SimParams, decays: List[float], gains: List[float], detunings: np.ndarray) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for md in decays:
        for mg in gains:
            p = SimParams(
                gamma=base.gamma,
                alpha=base.alpha,
                delta0=base.delta0,
                sigma=base.sigma,
                n_freq=base.n_freq,
                w_max=base.w_max,
                z=base.z,
                kappa=base.kappa,
                memory_decay=md,
                memory_gain=mg,
                te_delta_feedback=base.te_delta_feedback,
                te_kappa_feedback=base.te_kappa_feedback,
                observer_sigma=base.observer_sigma,
            )
            _, results = sweep_detuning(detunings, p)
            summary = summarize_results(results)
            rows.append({
                "memory_decay": md,
                "memory_gain": mg,
                **summary,
            })
    return rows


def print_memory_sweep(rows: List[Dict[str, float]]) -> None:
    headers = [
        "memory_decay", "memory_gain",
        "phase_abs_ratio_mean", "centroid_abs_ratio_mean", "peak_abs_ratio_mean",
        "phase_abs_ratio_std", "centroid_abs_ratio_std", "peak_abs_ratio_std",
        "phase_abs_diff_mean", "centroid_abs_diff_mean", "peak_abs_diff_mean",
        "te_current_peak_mean", "te_memory_peak_mean"
    ]
    print("\nTE memory sweep:")
    print(",".join(headers))
    for r in rows:
        print(
            f"{r['memory_decay']:.3f},{r['memory_gain']:.3f},"
            f"{r['phase_abs_ratio_mean']:.6f},{r['centroid_abs_ratio_mean']:.6f},{r['peak_abs_ratio_mean']:.6f},"
            f"{r['phase_abs_ratio_std']:.6f},{r['centroid_abs_ratio_std']:.6f},{r['peak_abs_ratio_std']:.6f},"
            f"{r['phase_abs_diff_mean']:.6f},{r['centroid_abs_diff_mean']:.6f},{r['peak_abs_diff_mean']:.6f},"
            f"{r['te_current_peak_mean']:.6e},{r['te_memory_peak_mean']:.6e}"
        )




def make_plots(results: List[Dict[str, float]], base: SimParams, rows: List[Dict[str, float]], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Figure 1: delays and ratios vs detuning
    delta = np.array([r["delta0"] for r in results])
    tau_phase = np.array([r["tau_g_phase"] for r in results])
    tau_cent = np.array([r["tau_g_centroid"] for r in results])
    tau_peak = np.array([r["tau_g_peak"] for r in results])
    tau_exc = np.array([r["tau_exc"] for r in results])
    r_phase = np.abs(np.array([r["R_phase"] for r in results]))
    r_cent = np.abs(np.array([r["R_centroid"] for r in results]))
    r_peak = np.abs(np.array([r["R_peak"] for r in results]))

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax=axes[0]
    ax.plot(delta, tau_phase, label='tau_phase')
    ax.plot(delta, tau_cent, label='tau_centroid')
    ax.plot(delta, tau_peak, label='tau_peak')
    ax.plot(delta, tau_exc, label='tau_exc')
    ax.set_ylabel('Delay')
    ax.set_title('Figure 1a: Delay measures vs detuning')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax=axes[1]
    ax.plot(delta, r_phase, label='|R_phase|')
    ax.plot(delta, r_cent, label='|R_centroid|')
    ax.plot(delta, r_peak, label='|R_peak|')
    ax.axhline(1.0, linestyle='--', linewidth=1)
    ax.set_xlabel('Detuning')
    ax.set_ylabel('|tau_g / tau_exc|')
    ax.set_title('Figure 1b: Ratio vs detuning')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path1 = out_dir / 'figure1_delay_and_ratio.png'
    fig.savefig(path1, dpi=180)
    plt.close(fig)
    paths.append(str(path1))

    # Figure 2: memory sweep
    selected_decays = [0.00, 0.10, 0.15, 0.20, 0.40]
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for md in selected_decays:
        subset = [r for r in rows if abs(r['memory_decay'] - md) < 1e-12]
        gains = np.array([r['memory_gain'] for r in subset])
        phase_ratio = np.array([r['phase_abs_ratio_mean'] for r in subset])
        te_mem = np.array([r['te_memory_peak_mean'] for r in subset])
        axes[0].plot(gains, phase_ratio, marker='o', label=f'decay={md:.2f}')
        axes[1].plot(gains, te_mem, marker='o', label=f'decay={md:.2f}')
    axes[0].axhline(1.0, linestyle='--', linewidth=1)
    axes[0].set_ylabel('Mean |R_phase|')
    axes[0].set_title('Figure 2a: Phase ratio vs memory gain')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel('memory_gain')
    axes[1].set_ylabel('TE memory peak mean')
    axes[1].set_title('Figure 2b: TE memory vs memory gain')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    path2 = out_dir / 'figure2_memory_sweep.png'
    fig.savefig(path2, dpi=180)
    plt.close(fig)
    paths.append(str(path2))

    # Figure 3: representative time-domain traces
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for ax, d in zip(axes, [-2.0, 0.0, 2.0]):
        p = SimParams(
            gamma=base.gamma, alpha=base.alpha, delta0=d, sigma=base.sigma,
            n_freq=base.n_freq, w_max=base.w_max, z=base.z, kappa=base.kappa,
            memory_decay=base.memory_decay, memory_gain=base.memory_gain,
            te_delta_feedback=base.te_delta_feedback, te_kappa_feedback=base.te_kappa_feedback,
            observer_sigma=base.observer_sigma,
        )
        w = np.linspace(-p.w_max, p.w_max, p.n_freq)
        t = time_grid_from_freq(w)
        e_in_w = gaussian_spectrum(w, p.delta0, p.sigma)
        e_in_t = ifft_shifted(e_in_w)
        H = transmission_function(w, p.gamma, p.alpha, p.z)
        e_out_w = e_in_w * H
        e_out_t = ifft_shifted(e_out_w)
        exc_t, te_current, te_memory = solve_atomic_excitation(
            t, e_in_t, p.gamma, p.delta0, p.kappa, p.memory_decay, p.memory_gain, p.te_delta_feedback, p.te_kappa_feedback
        )
        i_in = np.abs(e_in_t) ** 2
        i_out = np.abs(e_out_t) ** 2
        exc_scale = (np.max(i_in) / max(np.max(exc_t), 1e-15)) * 0.7
        te_scale = (np.max(i_in) / max(np.max(te_memory), 1e-15)) * 0.5
        ax.plot(t, i_in, label='input intensity')
        ax.plot(t, i_out, '--', label='output intensity')
        ax.plot(t, exc_t * exc_scale, ':', label='excitation (scaled)')
        ax.plot(t, te_memory * te_scale, '-.', label='TE memory (scaled)')
        ax.set_title(f'Figure 3 detuning={d:+.1f}')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time')
    axes[0].legend(loc='upper right')
    fig.tight_layout()
    path3 = out_dir / 'figure3_time_domain_traces.png'
    fig.savefig(path3, dpi=180)
    plt.close(fig)
    paths.append(str(path3))

    return paths

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
        memory_gain=1.0,
        te_delta_feedback=0.25,
        te_kappa_feedback=0.25,
        observer_sigma=1.0,
    )

    detunings = np.linspace(-4.0, 4.0, 25)
    _, results = sweep_detuning(detunings, base)

    print_single_sweep(results)
    print("\nsummary:")
    print(summarize_results(results))

    decays = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    gains = [0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
    rows = sweep_memory_params(base, decays, gains, detunings)
    print_memory_sweep(rows)

    out_dir = Path(__file__).resolve().parent / "group_delay_figures"
   paths = make_plots(results, base, rows, out_dir)

print("\nSaved figure files:")
for p in paths:
    print(p)


if __name__ == "__main__":
    main()
