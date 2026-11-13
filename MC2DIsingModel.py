import argparse
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

try:
    from scipy.special import ellipk
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

TC_ONSAGER = 2.0 / math.log(1.0 + math.sqrt(2.0))


@njit(cache=True)
def compute_total_energy(spins):
    size = spins.shape[0]
    energy = 0
    for i in range(size):
        for j in range(size):
            energy -= spins[i, j] * (spins[(i + 1) % size, j] + spins[i, (j + 1) % size])
    return energy


@njit(cache=True)
def run_metropolis(spins, temperature, n_burn, n_measure, seed):
    np.random.seed(seed)
    size = spins.shape[0]
    n_spins = size * size

    accept_prob = np.empty(2)
    accept_prob[0] = math.exp(-4.0 / temperature)
    accept_prob[1] = math.exp(-8.0 / temperature)

    energy = float(compute_total_energy(spins))
    magnetization = float(np.sum(spins))
    energies = np.empty(n_measure)
    magnetizations = np.empty(n_measure)

    for sweep in range(n_burn + n_measure):
        for _ in range(n_spins):
            i = np.random.randint(size)
            j = np.random.randint(size)
            neighbour_sum = (spins[(i + 1) % size, j] + spins[(i - 1) % size, j]
                             + spins[i, (j + 1) % size] + spins[i, (j - 1) % size])
            delta_e = 2 * spins[i, j] * neighbour_sum

            if delta_e <= 0 or np.random.random() < accept_prob[delta_e // 4 - 1]:
                spins[i, j] = -spins[i, j]
                energy += delta_e
                magnetization += 2 * spins[i, j]

        if sweep >= n_burn:
            energies[sweep - n_burn] = energy
            magnetizations[sweep - n_burn] = magnetization

    return energies, magnetizations


def run_simulation(spins, temperature, n_burn, n_measure, seed):
    energies, magnetizations = run_metropolis(spins, temperature, n_burn, n_measure, seed)
    return energies, magnetizations, float(spins.size)


def integrated_autocorr_time(series, window_factor=6.0):
    series = np.asarray(series, dtype=float)
    series = series - series.mean()
    n = len(series)
    if n < 10 or np.allclose(series, 0.0):
        return 0.5

    fourier = np.fft.rfft(series, 2 * n)
    acf = np.fft.irfft(fourier * np.conj(fourier))[:n]
    acf /= acf[0]
    running_tau = np.cumsum(acf) - 0.5
    lags = np.arange(n)
    converged = np.where(lags >= window_factor * running_tau)[0]
    window = converged[0] if len(converged) else n - 1
    return max(float(running_tau[window]), 0.5)


def compute_observables(mean_e, mean_e2, mean_abs_m, mean_m2, mean_m4, n_spins, temperature):
    energy_per_spin = mean_e / n_spins
    magnetization_per_spin = mean_abs_m / n_spins
    specific_heat = (mean_e2 - mean_e ** 2) / (n_spins * temperature ** 2)
    susceptibility = (mean_m2 - mean_abs_m ** 2) / (n_spins * temperature)
    binder = 1.0 - mean_m4 / (3.0 * mean_m2 ** 2) if mean_m2 > 0 else 0.0
    return np.array([energy_per_spin, magnetization_per_spin, specific_heat, susceptibility, binder])


def analyse_chain(energies, magnetizations, temperature, size):
    n_spins = size * size
    abs_mag = np.abs(magnetizations)
    tau_energy = integrated_autocorr_time(energies)
    tau_magnetization = integrated_autocorr_time(abs_mag)
    tau_max = max(tau_energy, tau_magnetization)
    n_samples = len(energies)

    block_size = max(1, min(int(math.ceil(5.0 * tau_max)), n_samples // 20))
    n_blocks = n_samples // block_size
    n_used = n_blocks * block_size

    moments = np.vstack([energies, energies ** 2, abs_mag, magnetizations ** 2, magnetizations ** 4])[:, :n_used]
    full_means = moments.mean(axis=1)
    block_means = moments.reshape(5, n_blocks, block_size).mean(axis=2)
    block_total = block_means.sum(axis=1)

    values = compute_observables(*full_means, n_spins, temperature)
    jackknife = np.empty((n_blocks, 5))
    for b in range(n_blocks):
        means_without_block = (block_total - block_means[:, b]) / (n_blocks - 1)
        jackknife[b] = compute_observables(*means_without_block, n_spins, temperature)

    errors = np.sqrt((n_blocks - 1) / n_blocks * np.sum((jackknife - jackknife.mean(axis=0)) ** 2, axis=0))

    return dict(
        e=values[0], e_err=errors[0],
        m=values[1], m_err=errors[1],
        C=values[2], C_err=errors[2],
        chi=values[3], chi_err=errors[3],
        U=values[4], U_err=errors[4],
        tau_E=tau_energy, tau_M=tau_magnetization,
        n_eff=n_samples / (2.0 * tau_max),
    )


def onsager_magnetization(temperature):
    temperature = np.asarray(temperature, dtype=float)
    result = np.zeros_like(temperature)
    ordered = temperature < TC_ONSAGER
    result[ordered] = (1.0 - np.sinh(2.0 / temperature[ordered]) ** -4) ** 0.125
    return result


def onsager_energy(temperature):
    beta = 1.0 / np.asarray(temperature, dtype=float)
    modulus = 2.0 * np.sinh(2 * beta) / np.cosh(2 * beta) ** 2
    elliptic_k = ellipk(modulus ** 2)
    coth = 1.0 / np.tanh(2 * beta)
    return -coth * (1.0 + (2.0 / np.pi) * (2.0 * np.tanh(2 * beta) ** 2 - 1.0) * elliptic_k)


def exact_observables_4x4(temperature):
    size = 4
    n_spins = size * size
    state_ids = np.arange(2 ** n_spins, dtype=np.int64)
    bits = (state_ids[:, None] >> np.arange(n_spins)) & 1
    spins = (2 * bits - 1).reshape(-1, size, size)

    energies = -np.sum(spins * np.roll(spins, 1, axis=1) + spins * np.roll(spins, 1, axis=2), axis=(1, 2)).astype(float)
    abs_mags = np.abs(spins.sum(axis=(1, 2))).astype(float)

    weights = np.exp(-(energies - energies.min()) / temperature)
    partition = weights.sum()

    mean_e = (weights * energies).sum() / partition
    mean_e2 = (weights * energies ** 2).sum() / partition
    mean_abs_m = (weights * abs_mags).sum() / partition

    return (mean_e / n_spins,
            (mean_e2 - mean_e ** 2) / (n_spins * temperature ** 2),
            mean_abs_m / n_spins)


def make_hot_start(size, rng):
    return rng.choice(np.array([-1, 1], dtype=np.int64), size=(size, size))


def make_cold_start(size):
    return np.ones((size, size), dtype=np.int64)


def run_validation(log):
    log("\n=== VALIDATION: 4x4 lattice, Metropolis vs exact enumeration ===")
    log(f"{'T':>5} | {'e (MC)':>16} {'e (exact)':>10} | {'C (MC)':>16} {'C (exact)':>10} | {'m (MC)':>16} {'m (exact)':>10}")

    passed = True
    for temperature in (1.5, 2.269, 3.0, 5.0):
        energies, magnetizations, _ = run_simulation(make_cold_start(4), temperature, 5000, 400000, 12345)
        mc = analyse_chain(energies, magnetizations, temperature, 4)
        e_exact, c_exact, m_exact = exact_observables_4x4(temperature)

        log(f"{temperature:5.3f} | {mc['e']:8.4f}+-{mc['e_err']:.4f} {e_exact:10.4f} | "
            f"{mc['C']:8.4f}+-{mc['C_err']:.4f} {c_exact:10.4f} | "
            f"{mc['m']:8.4f}+-{mc['m_err']:.4f} {m_exact:10.4f}")

        for estimate, error, exact in ((mc["e"], mc["e_err"], e_exact),
                                       (mc["C"], mc["C_err"], c_exact),
                                       (mc["m"], mc["m_err"], m_exact)):
            if abs(estimate - exact) > 4 * error + 1e-3:
                passed = False

    log("VALIDATION " + ("PASSED (all within ~4 sigma)" if passed else "FAILED - check the code!"))


def plot_fig1_equilibration(outdir, seed):
    size = 32
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rng = np.random.default_rng(seed)

    for ax, temperature in zip(axes, (2.0, TC_ONSAGER)):
        starts = (("hot start", make_hot_start(size, rng), "tab:red"),
                  ("cold start", make_cold_start(size), "tab:blue"))
        for label, spins, color in starts:
            energies, _, _ = run_simulation(spins, temperature, 0, 1500, seed + 1)
            ax.plot(energies / size ** 2, color=color, label=label, lw=1)

        ax.set_title(f"Equilibration at T = {temperature:.3f} (L={size}, Metropolis)")
        ax.set_xlabel("sweep")
        ax.set_ylabel("energy per spin")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig1_equilibration.png", dpi=150)
    plt.close(fig)


def plot_fig2_snapshots(outdir, size, seed):
    temperatures = (1.5, TC_ONSAGER, 3.5)
    n_burn = 3000
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    for ax, temperature in zip(axes, temperatures):
        spins = make_cold_start(size)
        run_simulation(spins, temperature, n_burn, 1, seed)
        ax.imshow(spins, cmap="gray", interpolation="nearest")
        ax.set_title(f"T = {temperature:.3f}  (m = {abs(spins.mean()):.2f})")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Spin configurations, L = {size}: ordered, critical, disordered")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig2_snapshots.png", dpi=150)
    plt.close(fig)


def build_temperature_grid(args):
    coarse = np.linspace(args.tmin, args.tmax, args.n_coarse)
    fine = np.linspace(2.15, 2.40, args.n_fine)
    return np.unique(np.round(np.concatenate([coarse, fine]), 4))


def run_temperature_scan(args, log):
    temperatures = build_temperature_grid(args)
    results = []

    for size in args.sizes:
        start_time = time.time()
        spins = make_cold_start(size)

        for k, temperature in enumerate(temperatures):
            seed = args.seed + 1000 * size + k
            energies, magnetizations, _ = run_simulation(
                spins, float(temperature), args.burn, args.sweeps, seed)

            result = analyse_chain(energies, magnetizations, float(temperature), size)
            result.update(L=size, T=float(temperature))

            result["tau_M_sweeps"] = result["tau_M"]
            results.append(result)

        log(f"  L={size:4d} done in {time.time() - start_time:6.1f} s")

    return results


def get_series(results, size, key):
    rows = sorted((r for r in results if r["L"] == size), key=lambda r: r["T"])
    return np.array([r[key] for r in rows])


def find_binder_crossings(results, sizes, window=(2.1, 2.45)):
    crossings = []
    for size_small, size_large in zip(sizes[:-1], sizes[1:]):
        temperatures = get_series(results, size_small, "T")
        difference = get_series(results, size_small, "U") - get_series(results, size_large, "U")
        best = None

        for i in range(len(temperatures) - 1):
            if not (window[0] <= temperatures[i] <= window[1]):
                continue

            if difference[i] == 0 or difference[i] * difference[i + 1] < 0:
                dT = temperatures[i + 1] - temperatures[i]
                dD = difference[i + 1] - difference[i]
                slope = abs(dD) / dT
                crossing_temperature = temperatures[i] - difference[i] * dT / dD

                if best is None or slope > best[1]:
                    best = (crossing_temperature, slope)

        if best is not None:
            crossings.append((size_small, size_large, best[0]))

    return crossings


def style_axis(ax, ylabel, tc_label="T_c = 2.269"):
    ax.axvline(TC_ONSAGER, color="r", ls=":", lw=1, label=tc_label)
    ax.set_xlabel("T  (J/k_B)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_errorbar_series(ax, results, sizes, colors, key):
    for size, color in zip(sizes, colors):
        ax.errorbar(get_series(results, size, "T"), get_series(results, size, key),
                    get_series(results, size, key + "_err"),
                    fmt="o-", ms=3, lw=0.8, color=color, label=f"L={size}")


def plot_fig3_energy_magnetization(results, sizes, colors, temperature_dense, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_errorbar_series(axes[0], results, sizes, colors, "e")
    plot_errorbar_series(axes[1], results, sizes, colors, "m")

    if HAVE_SCIPY:
        axes[0].plot(temperature_dense, onsager_energy(temperature_dense), "k--", lw=1.3, label="Onsager (L=inf)")
    axes[1].plot(temperature_dense, onsager_magnetization(temperature_dense), "k--", lw=1.3, label="Onsager (L=inf)")

    style_axis(axes[0], "energy per spin  e")
    style_axis(axes[1], "magnetization per spin  <|M|>/N")

    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig3_energy_magnetization.png", dpi=150)
    plt.close(fig)


def plot_fig4_specificheat_susceptibility(results, sizes, colors, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_errorbar_series(axes[0], results, sizes, colors, "C")
    plot_errorbar_series(axes[1], results, sizes, colors, "chi")

    style_axis(axes[0], "specific heat per spin  C")
    style_axis(axes[1], "susceptibility per spin  chi")

    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig4_specificheat_susceptibility.png", dpi=150)
    plt.close(fig)


def plot_fig5_binder_cumulant(results, sizes, colors, crossings, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, is_zoom in zip(axes, (False, True)):
        plot_errorbar_series(ax, results, sizes, colors, "U")
        style_axis(ax, "Binder cumulant  U_L", tc_label="T_c (Onsager)")

        if is_zoom:
            ax.set_xlim(2.1, 2.45)
            ax.set_ylim(0.35, 0.68)
            ax.set_title("zoom near T_c: curves cross")
            for _, _, crossing_temperature in crossings:
                ax.axvline(crossing_temperature, color="gray", lw=0.6, ls="--")

    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig5_binder_cumulant.png", dpi=150)
    plt.close(fig)


def plot_fig6_autocorrelation(results, sizes, colors, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    tau_at_tc = []

    for size, color in zip(sizes, colors):
        temperatures = get_series(results, size, "T")
        tau_sweeps = get_series(results, size, "tau_M_sweeps")
        axes[0].semilogy(temperatures, tau_sweeps, "o-", ms=3, lw=0.8, color=color, label=f"L={size}")
        tau_at_tc.append(tau_sweeps[np.argmin(np.abs(temperatures - TC_ONSAGER))])

    axes[0].axvline(TC_ONSAGER, color="r", ls=":", lw=1)
    axes[0].set_xlabel("T  (J/k_B)")
    axes[0].set_ylabel("tau_int of |M|  (sweep equivalents)")
    axes[0].set_title("Autocorrelation time (metropolis)")
    axes[0].legend(fontsize=8)

    sizes_array = np.array(sizes, dtype=float)
    tau_at_tc = np.array(tau_at_tc)
    axes[1].loglog(sizes_array, tau_at_tc, "o", color="tab:purple")

    z_exponent = None
    if len(sizes) >= 2:
        z_exponent, log_prefactor = np.polyfit(np.log(sizes_array), np.log(tau_at_tc), 1)
        axes[1].loglog(sizes_array, np.exp(log_prefactor) * sizes_array ** z_exponent, "-",
                       color="tab:purple", label=f"fit: tau ~ L^{z_exponent:.2f}")

    axes[1].set_xlabel("L")
    axes[1].set_ylabel("tau_int at T ~ T_c")
    axes[1].set_title("Critical slowing down")
    axes[1].legend()

    for ax in axes:
        ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig6_autocorrelation.png", dpi=150)
    plt.close(fig)
    return z_exponent


def make_result_plots(results, args, outdir):
    sizes = args.sizes
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(sizes)))
    temperature_dense = np.linspace(args.tmin, args.tmax, 600)
    temperature_dense = temperature_dense[np.abs(temperature_dense - TC_ONSAGER) > 0.004]

    plot_fig3_energy_magnetization(results, sizes, colors, temperature_dense, outdir)
    plot_fig4_specificheat_susceptibility(results, sizes, colors, outdir)

    crossings = find_binder_crossings(results, sizes)
    plot_fig5_binder_cumulant(results, sizes, colors, crossings, outdir)
    z_exponent = plot_fig6_autocorrelation(results, sizes, colors, outdir)

    summary = {"binder_crossings": crossings}
    if crossings:
        crossing_values = np.array([c[2] for c in crossings])
        std = crossing_values.std(ddof=1) if len(crossing_values) > 1 else float("nan")
        summary["tc_binder"] = (crossing_values.mean(), std)
    if z_exponent is not None:
        summary["z_exponent"] = z_exponent
    return summary


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--tmin", type=float, default=1.5)
    parser.add_argument("--tmax", type=float, default=3.5)
    parser.add_argument("--n-coarse", type=int, default=21)
    parser.add_argument("--n-fine", type=int, default=21)
    parser.add_argument("--burn", type=int, default=2000)
    parser.add_argument("--sweeps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--outdir", default="ising_output")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args, _ = parser.parse_known_args()

    if args.quick:
        args.sizes = [8, 16, 32]
        args.burn, args.sweeps = 1000, 5000
        args.n_coarse, args.n_fine = 12, 11

    args.sizes = sorted(args.sizes)
    return args


def main():
    args = parse_arguments()

    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    log_lines = []

    def log(message=""):
        print(message, flush=True)
        log_lines.append(message)

    start_time = time.time()
    log("2D Ising model - Monte Carlo study")
    log(f"algorithm=metropolis, sizes={args.sizes}, burn={args.burn}, measure={args.sweeps}, seed={args.seed}")
    log(f"Onsager T_c = {TC_ONSAGER:.5f}")

    if not args.skip_validation:
        run_validation(log)

    log("\nMaking equilibration and snapshot figures ...")
    plot_fig1_equilibration(args.outdir, args.seed)
    plot_fig2_snapshots(args.outdir, max(args.sizes), args.seed)

    log("\n=== Temperature scan ===")
    results = run_temperature_scan(args, log)

    log("\nMaking plots ...")
    summary = make_result_plots(results, args, args.outdir)

    log("\n\nFINAL RESULTS\n")
    log(f"Onsager exact T_c = {TC_ONSAGER:.4f}")

    for size_small, size_large, crossing_temperature in summary["binder_crossings"]:
        log(f"  Binder crossing (L={size_small}, L={size_large}):  T = {crossing_temperature:.4f}")

    if "tc_binder" in summary:
        mean_tc, std_tc = summary["tc_binder"]
        log(f"  Binder estimate:     T_c = {mean_tc:.4f} +- {std_tc:.4f}   "
            f"(deviation from Onsager: {mean_tc - TC_ONSAGER:+.4f})")

    if "z_exponent" in summary:
        literature = "literature: ~2.17 for Metropolis"
        log(f"Critical slowing down: tau ~ L^z with z = {summary['z_exponent']:.2f} ({literature})")

    log(f"\nMinimum effective sample size over all runs: {min(r['n_eff'] for r in results):.0f}")
    log(f"Total run time: {time.time() - start_time:.1f} s")
    log(f"Outputs written to: {os.path.abspath(args.outdir)}/")
    log("  summary.txt, fig1..fig6")

    with open(Path(args.outdir) / "summary.txt", "w") as summary_file:
        summary_file.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
