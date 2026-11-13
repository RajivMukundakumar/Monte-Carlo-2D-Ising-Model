import argparse
import math
import os
import time
import shutil
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
def total_energy(s):
    L = s.shape[0]
    e = 0
    for i in range(L):
        for j in range(L):
            e -= s[i, j] * (s[(i + 1) % L, j] + s[i, (j + 1) % L])
    return e

@njit(cache=True)
def metropolis_run(s, T, n_burn, n_meas, seed):
    np.random.seed(seed)
    L = s.shape[0]
    N = L * L
    
    w = np.empty(5)
    w[0] = 1.0
    w[1] = 1.0
    w[2] = 1.0
    w[3] = math.exp(-4.0 / T)
    w[4] = math.exp(-8.0 / T)

    E = float(total_energy(s))
    M = float(np.sum(s))
    Es = np.empty(n_meas)
    Ms = np.empty(n_meas)

    for sweep in range(n_burn + n_meas):
        for _ in range(N):
            i = np.random.randint(L)
            j = np.random.randint(L)
            nn = (s[(i + 1) % L, j] + s[(i - 1) % L, j]
                  + s[i, (j + 1) % L] + s[i, (j - 1) % L])
            dE = 2 * s[i, j] * nn
            
            if dE <= 0 or np.random.random() < w[dE // 4 + 2]:
                s[i, j] = -s[i, j]
                E += dE
                M += 2 * s[i, j]
                
        if sweep >= n_burn:
            Es[sweep - n_burn] = E
            Ms[sweep - n_burn] = M
            
    return Es, Ms

@njit(cache=True)
def wolff_run(s, T, n_burn, n_meas, seed):
    np.random.seed(seed)
    L = s.shape[0]
    N = L * L
    p_add = 1.0 - math.exp(-2.0 / T)
    
    stack_i = np.empty(N, dtype=np.int64)
    stack_j = np.empty(N, dtype=np.int64)
    Es = np.empty(n_meas)
    Ms = np.empty(n_meas)
    Cs = np.empty(n_meas)
    M = float(np.sum(s))

    for step in range(n_burn + n_meas):
        i0 = np.random.randint(L)
        j0 = np.random.randint(L)
        s0 = s[i0, j0]
        s[i0, j0] = -s0
        
        stack_i[0] = i0
        stack_j[0] = j0
        sp = 1
        size = 1
        
        while sp > 0:
            sp -= 1
            i = stack_i[sp]
            j = stack_j[sp]
            
            for d in range(4):
                if d == 0:
                    ni, nj = (i + 1) % L, j
                elif d == 1:
                    ni, nj = (i - 1) % L, j
                elif d == 2:
                    ni, nj = i, (j + 1) % L
                else:
                    ni, nj = i, (j - 1) % L
                    
                if s[ni, nj] == s0 and np.random.random() < p_add:
                    s[ni, nj] = -s0
                    stack_i[sp] = ni
                    stack_j[sp] = nj
                    sp += 1
                    size += 1
                    
        M += -2.0 * s0 * size
        
        if step >= n_burn:
            k = step - n_burn
            Es[k] = total_energy(s)
            Ms[k] = M
            Cs[k] = size
            
    return Es, Ms, Cs

def run_chain(s, T, n_burn, n_meas, seed, algo):
    if algo == "wolff":
        E, M, C = wolff_run(s, T, n_burn, n_meas, seed)
        return E, M, float(np.mean(C))
    E, M = metropolis_run(s, T, n_burn, n_meas, seed)
    return E, M, float(s.size)

def tau_int(x, c=6.0):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 10 or np.allclose(x, 0.0):
        return 0.5
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    taus = np.cumsum(acf) - 0.5
    t = np.arange(n)
    ok = np.where(t >= c * taus)[0]
    w = ok[0] if len(ok) else n - 1
    return max(float(taus[w]), 0.5)

def observables(E1, E2, A1, M2, M4, N, T):
    e = E1 / N
    m = A1 / N
    C = (E2 - E1 ** 2) / (N * T ** 2)
    chi = (M2 - A1 ** 2) / (N * T)
    U = 1.0 - M4 / (3.0 * M2 ** 2) if M2 > 0 else 0.0
    return np.array([e, m, C, chi, U])

def analyse(E, M, T, L, discard_nothing=True):
    N = L * L
    absM = np.abs(M)
    tE, tM = tau_int(E), tau_int(absM)
    tau = max(tE, tM)
    n = len(E)
    
    B = int(math.ceil(5.0 * tau))
    B = max(1, min(B, n // 20))
    nb = n // B
    use = nb * B

    mom = np.vstack([E, E ** 2, absM, M ** 2, M ** 4])[:, :use]
    full = mom.mean(axis=1)
    blocks = mom.reshape(5, nb, B).mean(axis=2)

    val = observables(*full, N, T)
    jk = np.empty((nb, 5))
    tot = blocks.sum(axis=1)
    
    for b in range(nb):
        mean_wo = (tot - blocks[:, b]) / (nb - 1)
        jk[b] = observables(*mean_wo, N, T)
        
    err = np.sqrt((nb - 1) / nb * np.sum((jk - jk.mean(axis=0)) ** 2, axis=0))

    return dict(
        e=val[0], e_err=err[0], m=val[1], m_err=err[1],
        C=val[2], C_err=err[2], chi=val[3], chi_err=err[3],
        U=val[4], U_err=err[4],
        tau_E=tE, tau_M=tM, n=n, n_eff=n / (2.0 * tau), nblocks=nb, block=B,
    )

def onsager_m(T):
    T = np.asarray(T, dtype=float)
    out = np.zeros_like(T)
    low = T < TC_ONSAGER
    out[low] = (1.0 - np.sinh(2.0 / T[low]) ** -4) ** 0.125
    return out

def onsager_e(T):
    T = np.asarray(T, dtype=float)
    b = 1.0 / T
    k = 2.0 * np.sinh(2 * b) / np.cosh(2 * b) ** 2
    K = ellipk(k ** 2)
    return -(1.0 / np.tanh(2 * b)) * (1.0 + (2.0 / np.pi) * (2.0 * np.tanh(2 * b) ** 2 - 1.0) * K)

def exact_enumeration_4x4(T):
    L = 4
    N = L * L
    states = np.arange(2 ** N, dtype=np.int64)
    bits = (states[:, None] >> np.arange(N)) & 1
    s = (2 * bits - 1).reshape(-1, L, L)
    
    E = -np.sum(s * np.roll(s, 1, axis=1) + s * np.roll(s, 1, axis=2), axis=(1, 2)).astype(float)
    Mabs = np.abs(s.sum(axis=(1, 2))).astype(float)
    
    wgt = np.exp(-(E - E.min()) / T)
    Z = wgt.sum()
    
    E1 = (wgt * E).sum() / Z
    E2 = (wgt * E ** 2).sum() / Z
    A1 = (wgt * Mabs).sum() / Z
    
    return E1 / N, (E2 - E1 ** 2) / (N * T ** 2), A1 / N

def hot_start(L, rng):
    return rng.choice(np.array([-1, 1], dtype=np.int64), size=(L, L))

def cold_start(L):
    return np.ones((L, L), dtype=np.int64)

def validation(outdir, log):
    log("\n=== VALIDATION: 4x4 lattice, Metropolis vs exact enumeration ===")
    log(f"{'T':>5} | {'e (MC)':>16} {'e (exact)':>10} | {'C (MC)':>16} {'C (exact)':>10} | {'m (MC)':>16} {'m (exact)':>10}")
    
    ok = True
    for T in (1.5, 2.269, 3.0, 5.0):
        s = cold_start(4)
        E, M, _ = run_chain(s, T, 5000, 400000, 12345, "metropolis")
        r = analyse(E, M, T, 4)
        e_ex, C_ex, m_ex = exact_enumeration_4x4(T)
        
        log(f"{T:5.3f} | {r['e']:8.4f}+-{r['e_err']:.4f} {e_ex:10.4f} | "
            f"{r['C']:8.4f}+-{r['C_err']:.4f} {C_ex:10.4f} | {r['m']:8.4f}+-{r['m_err']:.4f} {m_ex:10.4f}")
            
        for mc, err, ex in ((r["e"], r["e_err"], e_ex), (r["C"], r["C_err"], C_ex), (r["m"], r["m_err"], m_ex)):
            if abs(mc - ex) > 4 * err + 1e-3:
                ok = False
                
    log("VALIDATION " + ("PASSED (all within ~4 sigma)" if ok else "FAILED - check the code!"))
    return ok

def equilibration_figure(outdir, seed, algo):
    L = 32
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rng = np.random.default_rng(seed)
    
    for ax, T in zip(axes, (2.0, TC_ONSAGER)):
        for name, s0, col in (("hot start", hot_start(L, rng), "tab:red"), ("cold start", cold_start(L), "tab:blue")):
            E, M, _ = run_chain(s0.copy(), T, 0, 1500, seed + 1, "metropolis")
            ax.plot(E / L ** 2, color=col, label=name, lw=1)
            
        ax.set_title(f"Equilibration at T = {T:.3f} (L={L}, Metropolis)")
        ax.set_xlabel("sweep")
        ax.set_ylabel("energy per spin")
        ax.legend()
        ax.grid(alpha=0.3)
        
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig1_equilibration.png", dpi=150)
    plt.close(fig)

def snapshot_figure(outdir, L, seed, algo):
    temps = (1.5, TC_ONSAGER, 3.5)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    
    for ax, T in zip(axes, temps):
        s = cold_start(L)
        run_chain(s, T, 3000 if algo == "metropolis" else 1000, 1, seed, algo)
        ax.imshow(s, cmap="gray", interpolation="nearest")
        ax.set_title(f"T = {T:.3f}  (m = {abs(s.mean()):.2f})")
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig.suptitle(f"Spin configurations, L = {L}: ordered, critical, disordered")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig2_snapshots.png", dpi=150)
    plt.close(fig)

def temperature_grid(args):
    coarse = np.linspace(args.tmin, args.tmax, args.n_coarse)
    fine = np.linspace(2.15, 2.40, args.n_fine)
    return np.unique(np.round(np.concatenate([coarse, fine]), 4))

def scan(args, log):
    Ts = temperature_grid(args)
    results = []
    
    for L in args.sizes:
        t0 = time.time()
        s = cold_start(L)
        
        for k, T in enumerate(Ts):
            seed = args.seed + 1000 * L + k
            E, M, csz = run_chain(s, float(T), args.burn, args.sweeps, seed, args.algo)
            
            r = analyse(E, M, float(T), L)
            r.update(L=L, T=float(T), mean_cluster=csz)
            
            scale = csz / (L * L) if args.algo == "wolff" else 1.0
            r["tau_E_sweeps"] = r["tau_E"] * scale
            r["tau_M_sweeps"] = r["tau_M"] * scale
            
            results.append(r)
            
        log(f"  L={L:4d} done in {time.time() - t0:6.1f} s")
        
    return results

def arr(results, L, key):
    rows = sorted([r for r in results if r["L"] == L], key=lambda r: r["T"])
    return np.array([r[key] for r in rows])

def binder_crossings(results, sizes, window=(2.1, 2.45)):
    out = []
    for L1, L2 in zip(sizes[:-1], sizes[1:]):
        T = arr(results, L1, "T")
        d = arr(results, L1, "U") - arr(results, L2, "U")
        best = None
        
        for i in range(len(T) - 1):
            if not (window[0] <= T[i] <= window[1]):
                continue
                
            if d[i] == 0 or d[i] * d[i + 1] < 0:
                slope = abs(d[i + 1] - d[i]) / (T[i + 1] - T[i])
                tc = T[i] - d[i] * (T[i + 1] - T[i]) / (d[i + 1] - d[i])
                
                if best is None or slope > best[1]:
                    best = (tc, slope)
                    
        if best is not None:
            out.append((L1, L2, best[0]))
            
    return out

def make_plots(results, args, outdir, log):
    sizes = args.sizes
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(sizes)))
    Tdense = np.linspace(args.tmin, args.tmax, 600)
    Tdense = Tdense[np.abs(Tdense - TC_ONSAGER) > 0.004]
    summary = {}
    base_dir = Path(outdir)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for L, c in zip(sizes, colors):
        T = arr(results, L, "T")
        ax[0].errorbar(T, arr(results, L, "e"), arr(results, L, "e_err"), fmt="o-", ms=3, lw=0.8, color=c, label=f"L={L}")
        ax[1].errorbar(T, arr(results, L, "m"), arr(results, L, "m_err"), fmt="o-", ms=3, lw=0.8, color=c, label=f"L={L}")
        
    if HAVE_SCIPY:
        ax[0].plot(Tdense, onsager_e(Tdense), "k--", lw=1.3, label="Onsager (L=inf)")
        
    ax[1].plot(Tdense, onsager_m(Tdense), "k--", lw=1.3, label="Onsager (L=inf)")
    
    for a, yl in zip(ax, ("energy per spin  e", "magnetization per spin  <|M|>/N")):
        a.axvline(TC_ONSAGER, color="r", ls=":", lw=1, label="T_c = 2.269")
        a.set_xlabel("T  (J/k_B)")
        a.set_ylabel(yl)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        
    fig.tight_layout()
    fig.savefig(base_dir / "fig3_energy_magnetization.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for L, c in zip(sizes, colors):
        T = arr(results, L, "T")
        ax[0].errorbar(T, arr(results, L, "C"), arr(results, L, "C_err"), fmt="o-", ms=3, lw=0.8, color=c, label=f"L={L}")
        ax[1].errorbar(T, arr(results, L, "chi"), arr(results, L, "chi_err"), fmt="o-", ms=3, lw=0.8, color=c, label=f"L={L}")
        
    for a, yl in zip(ax, ("specific heat per spin  C", "susceptibility per spin  chi")):
        a.axvline(TC_ONSAGER, color="r", ls=":", lw=1, label="T_c = 2.269")
        a.set_xlabel("T  (J/k_B)")
        a.set_ylabel(yl)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        
    fig.tight_layout()
    fig.savefig(base_dir / "fig4_specificheat_susceptibility.png", dpi=150)
    plt.close(fig)

    crossings = binder_crossings(results, sizes)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, zoom in zip(ax, (False, True)):
        for L, c in zip(sizes, colors):
            a.errorbar(arr(results, L, "T"), arr(results, L, "U"), arr(results, L, "U_err"),
                       fmt="o-", ms=3, lw=0.8, color=c, label=f"L={L}")
        a.axvline(TC_ONSAGER, color="r", ls=":", lw=1, label="T_c (Onsager)")
        a.set_xlabel("T  (J/k_B)")
        a.set_ylabel("Binder cumulant  U_L")
        a.grid(alpha=0.3)
        
        if zoom:
            a.set_xlim(2.1, 2.45)
            a.set_ylim(0.35, 0.68)
            a.set_title("zoom near T_c: curves cross")
            for L1, L2, tc in crossings:
                a.axvline(tc, color="gray", lw=0.6, ls="--")
        a.legend(fontsize=8)
        
    fig.tight_layout()
    fig.savefig(base_dir / "fig5_binder_cumulant.png", dpi=150)
    plt.close(fig)
    
    cross_vals = np.array([c[2] for c in crossings])
    summary["binder"] = crossings
    if len(cross_vals):
        summary["Tc_binder"] = (cross_vals.mean(), cross_vals.std(ddof=1) if len(cross_vals) > 1 else float("nan"))

    Lf = np.array(sizes, dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    tau_tc = []
    
    for L, c in zip(sizes, colors):
        T = arr(results, L, "T")
        tm = arr(results, L, "tau_M_sweeps")
        ax[0].semilogy(T, tm, "o-", ms=3, lw=0.8, color=c, label=f"L={L}")
        i = int(np.argmin(np.abs(T - TC_ONSAGER)))
        tau_tc.append(tm[i])
        
    ax[0].axvline(TC_ONSAGER, color="r", ls=":", lw=1)
    ax[0].set_xlabel("T  (J/k_B)")
    ax[0].set_ylabel("tau_int of |M|  (sweep equivalents)")
    ax[0].set_title(f"Autocorrelation time ({args.algo})")
    ax[0].legend(fontsize=8)
    
    tau_tc = np.array(tau_tc)
    ax[1].loglog(Lf, tau_tc, "o", color="tab:purple")
    
    if len(sizes) >= 2:
        z, icz = np.polyfit(np.log(Lf), np.log(tau_tc), 1)
        summary["z_fit"] = z
        ax[1].loglog(Lf, np.exp(icz) * Lf ** z, "-", color="tab:purple", label=f"fit: tau ~ L^{z:.2f}")
        
    ax[1].set_xlabel("L")
    ax[1].set_ylabel("tau_int at T ~ T_c")
    ax[1].set_title("Critical slowing down")
    ax[1].legend()
    
    for a in ax:
        a.grid(alpha=0.3, which="both")
        
    fig.tight_layout()
    fig.savefig(base_dir / "fig8_autocorrelation.png", dpi=150)
    plt.close(fig)
    
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--tmin", type=float, default=1.5)
    ap.add_argument("--tmax", type=float, default=3.5)
    ap.add_argument("--n-coarse", type=int, default=21)
    ap.add_argument("--n-fine", type=int, default=21)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--sweeps", type=int, default=20000)
    ap.add_argument("--algo", choices=["metropolis", "wolff"], default="metropolis")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--outdir", default="ising_output")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-validation", action="store_true")
    
    args, _ = ap.parse_known_args()

    if args.quick:
        args.sizes = [8, 16, 32]
        args.burn, args.sweeps = 1000, 5000
        args.n_coarse, args.n_fine = 12, 11
        
    args.sizes = sorted(args.sizes)
    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    lines = []
    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    if not HAVE_NUMBA:
        log("WARNING: numba not found. Pure Python will be extremely slow.")
    if not HAVE_SCIPY:
        log("NOTE: scipy not found; Onsager energy curve will be skipped.")

    t_start = time.time()
    log("2D Ising model - Monte Carlo study")
    log(f"algorithm={args.algo}, sizes={args.sizes}, burn={args.burn}, measure={args.sweeps}, seed={args.seed}")
    log(f"Onsager T_c = {TC_ONSAGER:.5f}")

    if not args.skip_validation:
        validation(args.outdir, log)

    log("\nMaking equilibration and snapshot figures ...")
    equilibration_figure(args.outdir, args.seed, args.algo)
    snapshot_figure(args.outdir, max(args.sizes), args.seed, args.algo)

    log("\n=== Temperature scan ===")
    results = scan(args, log)

    log("\nMaking plots ...")
    summary = make_plots(results, args, args.outdir, log)

    log("\n\nFINAL RESULTS\n")
    log(f"Onsager exact T_c = {TC_ONSAGER:.4f}")
    
    if "binder" in summary:
        for L1, L2, tc in summary["binder"]:
            log(f"  Binder crossing (L={L1}, L={L2}):  T = {tc:.4f}")
            
    if "Tc_binder" in summary:
        m, s = summary["Tc_binder"]
        log(f"  Binder estimate:     T_c = {m:.4f} +- {s:.4f}   (deviation from Onsager: {m - TC_ONSAGER:+.4f})")
        
    if "z_fit" in summary:
        log(f"Critical slowing down: tau ~ L^z with z = {summary['z_fit']:.2f} "
            f"({'literature: ~2.17 for Metropolis' if args.algo == 'metropolis' else 'literature: ~0.25 for Wolff'})")
            
    log(f"\nMinimum effective sample size over all runs: {min(r['n_eff'] for r in results):.0f}")
    log(f"Total run time: {time.time() - t_start:.1f} s")
    log(f"Outputs written to: {os.path.abspath(args.outdir)}/")
    log("  summary.txt, fig1..fig5, fig8")

    with open(Path(args.outdir) / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()