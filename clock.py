"""Clock chains and the Blume--Capel control: Table 2, Figure 3 and the section 5.4 numbers,
plus the replica-subgroup check of eq. (3.5).

    python3 clock.py all --output results/records.json > results/clock.txt     (~3 min)
    python3 clock.py finite-size >> results/clock.txt                         (about 1 h)

Subcommands:
  replica       eq. (3.5): replica subgroup K_alpha and its annihilator, d = 2..5
  fits          Table 2 and Figure 3: charged slopes, window sensitivities and intercepts
  fractions     section 5.4: neutral / all-charged fractions and sector exponents
  finite-size   section 5.4: size sequences to L = 14, 12, 10 and same-ring ratios
  blume-capel   section 5.4: the non-Weyl control at lambda = 1.326, L = 12

The Fateev--Zamolodchikov chains, eq. (5.17), are solved by exact diagonalization on periodic
rings with a matrix-free Hamiltonian.  For a d-state site, X|j> = |j+1>, Z|j> = w^j|j>,
W_u = prod X^a Z^b, the charge under prod X is q = sum b mod d, and Xi(u) = |Tr rho W_u|^2 / D.
Single sites at separation r on a ring of L sites have eta = [l_c(1)/l_c(r)]^2 with chord length
l_c(x) = (L/pi) sin(pi x/L).  The primary fit uses the full separation window; the parenthesized
sensitivities are the largest shifts when the one or two shortest separations are removed,
keeping at least three points.  They are not statistical errors.
"""

import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def save_record(path, key, rows):
    record = json.load(open(path, encoding="utf-8")) if Path(path).exists() else {}
    record[key] = rows
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=1, sort_keys=True)


# ----------------------------------------------------------------------------------------
# Weyl basis, reduced density matrices and ground states
# ----------------------------------------------------------------------------------------

def weyl(d):
    """Weyl operators X^a Z^b, their clock charges b, and the vectorized overlap matrix."""
    w = np.exp(2j * np.pi / d); Z = np.diag([w ** j for j in range(d)])
    X = np.zeros((d, d), complex)
    for j in range(d):
        X[(j + 1) % d, j] = 1.0
    W = [np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b) for a in range(d) for b in range(d)]
    charges = [b for a in range(d) for b in range(d)]
    M = np.zeros((d * d, d * d), complex)
    for u, P in enumerate(W):
        for i in range(d):
            for j in range(d):
                M[u, d * i + j] = P[j, i]
    return W, charges, M


def rdm(psi, L, d, sites):
    n = len(sites); rest = [k for k in range(L) if k not in sites]
    T = np.transpose(psi.reshape((d,) * L), list(sites) + rest).reshape(d ** n, -1)
    return T @ T.conj().T


def chis(rho, n, d, M):
    """All Weyl expectations Tr(rho W_u) of an n-site density matrix."""
    T = rho.reshape((d,) * (2 * n)); perm = []
    for k in range(n):
        perm += [k, n + k]
    T = np.transpose(T, perm).reshape((d * d,) * n)
    for ax in range(n):
        T = np.moveaxis(np.tensordot(M, T, axes=([1], [ax])), 0, ax)
    return T.reshape(-1)


def fz_ground(d, L, tol=1e-9):
    """Matrix-free ground state of H = -sum_j sum_n a_n [(Z_j^dag Z_{j+1})^n + X_j^n], a_n = 1/sin(pi n/d)."""
    w = np.exp(2j * np.pi / d); a = np.array([0.0] + [1.0 / np.sin(np.pi * n / d) for n in range(1, d)])
    f = np.array([sum(a[n] * w ** (n * k) for n in range(1, d)) for k in range(d)]).real   # bond energy
    shape = (d,) * L
    diag = np.zeros(shape)
    for j in range(L):
        k = (j + 1) % L
        ci = np.arange(d).reshape([d if t == j else 1 for t in range(L)])
        ck = np.arange(d).reshape([d if t == k else 1 for t in range(L)])
        diag = diag - f[(ck - ci) % d]

    def matvec(v):
        T = v.reshape(shape); out = diag * T
        for j in range(L):
            for n in range(1, d):
                out = out - a[n] * np.roll(T, n, axis=j)
        return out.ravel()

    E, V = eigsh(LinearOperator((d ** L, d ** L), matvec=matvec, dtype=np.float64),
                 k=1, which="SA", tol=tol, maxiter=20000)
    return np.asarray(V[:, 0]).ravel(), E[0]


def chord(L, x):
    return (L / np.pi) * np.sin(np.pi * x / L)


def fundamental_ratios(psi, d, L, alphas):
    """Fundamental-charge (q = +-1) ratios for single sites at separations s with eta <= 1/2."""
    _, charges, M = weyl(d)
    xA = np.abs(chis(rdm(psi, L, d, [0]), 1, d, M)) ** 2 / d
    N = d * d; labels = np.arange(N * N)
    qA = np.array([charges[i % N] for i in labels]); qB = np.array([charges[i // N] for i in labels])
    mask = (((qA + qB) % d) == 0) & ((qA == 1) | (qA == d - 1))
    eta, ratios, separations = [], {a: [] for a in alphas}, []
    for s in range(2, L // 2 + 1):
        value = (chord(L, 1) / chord(L, s)) ** 2
        if value > 0.5:
            continue
        xB = np.abs(chis(rdm(psi, L, d, [s]), 1, d, M)) ** 2 / d
        xAB = np.abs(chis(rdm(psi, L, d, [0, s]), 2, d, M)) ** 2 / d ** 2
        eta.append(value); separations.append(s)
        for a in alphas:
            ratios[a].append((xAB[mask] ** a).sum() / ((xA ** a).sum() * (xB ** a).sum()))
    return np.array(eta), ratios, separations


def windows(n, minimum=3):
    return [(i, n) for i in (0, 1, 2) if n - i >= minimum]


def window_fits(eta, ratios, alphas):
    """Per-alpha exponents and the slope/intercept of h_ch(alpha) for each predetermined window."""
    x = np.log(eta); fits = []
    for i, j in windows(len(eta)):
        p = [np.polyfit(x[i:j], np.log(np.abs(np.array(ratios[a])[i:j])), 1)[0] for a in alphas]
        slope, intercept = np.polyfit(list(alphas), p, 1)
        fits.append((slope, intercept, p))
    return fits


# ----------------------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------------------

def cmd_replica(args):
    for d in range(2, 6):
        for alpha in (2, 3):
            v = np.array([1] * alpha + [-1] * alpha, dtype=int) % d
            vectors = np.array(list(product(range(d), repeat=2 * alpha)), dtype=int)
            vdot = vectors @ v % d
            invariant = np.all(np.outer(vdot, np.arange(d)) % d == 0, axis=1)   # every charge q occurs
            kernel = vdot == 0
            assert np.array_equal(invariant, kernel), f"tensor stabilizer failed for d={d}, alpha={alpha}"
            trivial_on_kernel = np.all((vectors @ vectors[kernel].T) % d == 0, axis=1)
            span = {tuple((q * v) % d) for q in range(d)}
            assert np.array_equal(trivial_on_kernel, np.array([tuple(r) in span for r in vectors])), \
                f"annihilator failed for d={d}, alpha={alpha}"
            print(f"d={d}, alpha={alpha}: |K|={int(kernel.sum())}, |K_perp|={int(trivial_on_kernel.sum())}")


def cmd_fits(args):
    print("charged-sector primary fit and predetermined window sensitivity\n")
    from ising import length_two_ratios
    _, eta, _, charged = length_two_ratios((2, 3, 4, 5))
    models = [("Ising", eta, charged, (2, 3, 4, 5), 0.25)]
    for name, d, L, pred in (("Potts", 3, 12, 4 / 15), ("Z4 FZ", 4, 10, 0.25), ("Z5 FZ", 5, 8, 8 / 35)):
        psi, _ = fz_ground(d, L, tol=1e-10)
        e, q, _ = fundamental_ratios(psi, d, L, (2, 3, 4))
        models.append((name, e, q, (2, 3, 4), pred))
    for name, e, q, alphas, pred in models:
        fits = window_fits(e, q, alphas)
        slope0, intercept0, p0 = fits[0]
        slope_sens = max(abs(f[0] - slope0) for f in fits)
        intercept_sens = max(abs(f[1] - intercept0) for f in fits)
        p_sens = [max(abs(f[2][k] - p0[k]) for f in fits) for k in range(len(alphas))]
        print(f"  {name:<12} slope = {slope0:.4f} +- {slope_sens:.4f}; "
              f"intercept = {intercept0:+.4f} +- {intercept_sens:.4f}   "
              f"[pred slope {pred:.4f}, intercept 0]  windows={windows(len(e))}  pts={len(e)}")
        print(" " * 16 + "individual h_ch: " + ", ".join(
            f"a={a}: {value:.4f} +- {sens:.4f}" for a, value, sens in zip(alphas, p0, p_sens)), flush=True)


def sector_fractions(d, L, alphas=(2, 3, 4)):
    psi, energy = fz_ground(d, L, tol=1e-10)
    _, charges, matrix = weyl(d)
    charges = np.asarray(charges)
    xi_a = np.abs(chis(rdm(psi, L, d, [0]), 1, d, matrix)) ** 2 / d
    q_a = np.repeat(charges, d * d); q_b = np.tile(charges, d * d)
    conserved = (q_a + q_b) % d == 0
    masks = {"fundamental": conserved & ((q_a == 1) | (q_a == d - 1)), "all_charged": conserved & (q_a != 0),
             "neutral": (q_a == 0) & (q_b == 0)}
    eta, leakage = [], []
    series = {k: {a: [] for a in alphas} for k in ("fundamental", "all_charged", "neutral", "full")}
    for s in range(2, L // 2 + 1):
        value = (chord(L, 1) / chord(L, s)) ** 2
        if value > 0.5:
            continue
        xi_b = np.abs(chis(rdm(psi, L, d, [s]), 1, d, matrix)) ** 2 / d
        xi_ab = np.abs(chis(rdm(psi, L, d, [0, s]), 2, d, matrix)) ** 2 / d ** 2
        eta.append(value); leakage.append(float(xi_ab[~conserved].sum()))
        for a in alphas:
            denominator = np.sum(xi_a ** a) * np.sum(xi_b ** a)
            series["fundamental"][a].append(np.sum(xi_ab[masks["fundamental"]] ** a) / denominator)
            series["all_charged"][a].append(np.sum(xi_ab[masks["all_charged"]] ** a) / denominator)
            series["neutral"][a].append(np.sum(xi_ab[masks["neutral"]] ** a) / denominator - 1.0)
            series["full"][a].append(np.sum(xi_ab ** a) / denominator - 1.0)
    log_eta = np.log(eta)
    exponent = lambda values: float(np.polyfit(log_eta, np.log(np.abs(values)), 1)[0])
    fractions = {a: np.array(series["neutral"][a]) / np.array(series["all_charged"][a]) for a in alphas}
    return {"d": d, "L": L, "energy_density": float(energy / L), "eta": eta, "max_charge_leakage": max(leakage),
            **{f"{k}_exponents": {str(a): exponent(series[k][a]) for a in alphas} for k in series},
            "neutral_to_all_charged": {str(a): [float(fractions[a].min()), float(fractions[a].max())] for a in alphas}}


def cmd_fractions(args):
    rows = []
    for item in args.models.split(","):
        d, L = (int(v) for v in item.split(":"))
        rows.append(sector_fractions(d, L))
        r = rows[-1]
        print(f"d={d}, L={L}: neutral/all-charged at alpha=2 between {100 * r['neutral_to_all_charged']['2'][0]:.0f}% "
              f"and {100 * r['neutral_to_all_charged']['2'][1]:.0f}%;  max charge leakage {r['max_charge_leakage']:.1e}")
        print("  exponents (alpha=2,3,4): " + "; ".join(
            f"{k} " + ", ".join(f"{r[k + '_exponents'][str(a)]:.4f}" for a in (2, 3, 4))
            for k in ("fundamental", "all_charged", "neutral", "full")), flush=True)
    return rows


def cmd_finite_size(args):
    models = {3: ("Potts", 4 / 15), 4: ("Z4 FZ", 0.25), 5: ("Z5 FZ", 8 / 35)}
    print("\n===== finite-size: section 5.4 =====\n")
    print(f"  {'model':<8}{'L':>4}{'pts':>5}{'win':>5}{'slope':>12}{'ring dim.':>16}{'ratio':>14}")
    for block in args.sizes.split(";"):
        d, sizes = block.split(":"); d = int(d)
        name, pred = models[d]
        for L in (int(x) for x in sizes.split(",")):
            psi, _ = fz_ground(d, L)
            alphas = (2, 3, 4)
            eta, q, separations = fundamental_ratios(psi, d, L, alphas)
            ws = windows(len(eta))
            if not ws:
                print(f"  {name:<8}{L:>4}   too few separations"); continue
            # ring dimension 2 Delta from <Z_0^dag Z_s>, fitted on the same windows
            w = np.exp(2j * np.pi / d); phases = np.array([w ** j for j in range(d)])
            weight = np.abs(psi.reshape((d,) * L)) ** 2
            zz = [abs(complex((weight * phases.conj().reshape([d] + [1] * (L - 1))
                               * phases.reshape([d if t == s else 1 for t in range(L)])).sum())) for s in separations]
            x = np.log(eta)
            slopes = np.array([f[0] for f in window_fits(eta, q, alphas)])
            dims = np.array([2 * np.polyfit(x[i:j], np.log(np.array(zz)[i:j]), 1)[0] for i, j in ws])
            ratio = slopes / dims
            sens = lambda v: int(round(np.max(np.abs(v - v[0])) * 1e4))
            print(f"  {name:<8}{L:>4}{len(eta):>5}{len(ws):>5}   {slopes[0]:.4f}({sens(slopes):>2d})"
                  f"   d={dims[0]:.4f}({sens(dims):>2d})   {ratio[0]:.4f}({sens(ratio):>2d})", flush=True)
        print(f"           [2*Delta exact = {pred:.4f}]")


SZ = np.diag([0.0, 1.0, -1.0])
SX = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) / np.sqrt(2.0)


def spin1_ground(L, lam, D=0.0, nev=2, tol=1e-11):
    """Lowest eigenpairs of H = -sum S^z S^z - lam sum S^x + D sum (S^z)^2 on a periodic ring."""
    shape = (3,) * L; labels = np.array([0.0, 1.0, -1.0]); diag = np.zeros(shape)
    for j in range(L):
        k = (j + 1) % L
        mj = labels.reshape([3 if a == j else 1 for a in range(L)])
        mk = labels.reshape([3 if a == k else 1 for a in range(L)])
        diag -= mj * mk
        if D:
            diag += D * mj * mj

    def matvec(vector):
        tensor = vector.reshape(shape); out = diag * tensor
        for j in range(L):
            out -= lam * np.moveaxis(np.tensordot(SX, tensor, axes=([1], [j])), 0, j)
        return out.ravel()

    energy, vectors = eigsh(LinearOperator((3 ** L, 3 ** L), matvec=matvec, dtype=np.float64),
                            k=nev, which="SA", tol=tol, maxiter=30000)
    order = np.argsort(energy)
    return energy[order], np.asarray(vectors[:, order])


def contiguous_window_slopes(x, y, minimum=3):
    return np.asarray([np.polyfit(x[i:j], np.log(np.abs(y[i:j])), 1)[0]
                       for i in range(len(x)) for j in range(i + minimum, len(x) + 1)])


def local_expectation(psi, L, sites, operators):
    op = np.array([[1.0]])
    for matrix in operators:
        op = np.kron(op, matrix)
    return complex(np.trace(rdm(psi, L, 3, sites) @ op))


def blume_capel(L, lam):
    energies, vectors = spin1_ground(L, lam)
    psi = vectors[:, 0].real; psi /= np.linalg.norm(psi)
    lengths = range(2, min(5, L // 2) + 1); entropy = []
    for ell in lengths:
        p = np.linalg.eigvalsh(rdm(psi, L, 3, list(range(ell)))); p = p[p > 1e-14]
        entropy.append(-np.sum(p * np.log(p)))
    result = {"L": L, "lambda": lam, "energy_density": float(energies[0] / L),
              "scaled_gap": float(L * (energies[1] - energies[0])),
              "c_eff_vn": float(3.0 * np.polyfit([np.log(chord(L, ell)) for ell in lengths], entropy, 1)[0])}

    distances = np.arange(2, L // 2 + 1)
    log_eta = np.log(np.array([(chord(L, 1) / chord(L, int(s))) ** 2 for s in distances]))
    mean_sz2 = local_expectation(psi, L, [0], [SZ @ SZ]).real
    mean_sx = local_expectation(psi, L, [0], [SX]).real
    spin = [abs(local_expectation(psi, L, [0, int(s)], [SZ, SZ])) for s in distances]
    energy_1 = [abs(local_expectation(psi, L, [0, int(s)], [SZ @ SZ, SZ @ SZ]).real - mean_sz2 ** 2) for s in distances]
    energy_2 = [abs(local_expectation(psi, L, [0, int(s)], [SX, SX]).real - mean_sx ** 2) for s in distances]
    spin_slopes = -contiguous_window_slopes(np.log([chord(L, int(s)) for s in distances]), spin)
    eps1 = contiguous_window_slopes(log_eta, energy_1); eps2 = contiguous_window_slopes(log_eta, energy_2)
    result.update({"two_delta_sigma_mean": float(spin_slopes.mean()), "two_delta_sigma_sd": float(spin_slopes.std()),
                   "delta_epsilon_sz2_mean": float(eps1.mean()), "delta_epsilon_sz2_sd": float(eps1.std()),
                   "delta_epsilon_sx_mean": float(eps2.mean()), "delta_epsilon_sx_sd": float(eps2.std())})

    _, _, matrix = weyl(3)
    conjugate = np.array([(3 - (u // 3)) % 3 * 3 + (3 - (u % 3)) % 3 for u in range(9)])
    alphas = (2, 3, 4)
    full, even, quadratic, linear = ({a: [] for a in alphas} for _ in range(4))
    leakage = []
    c_a = chis(rdm(psi, L, 3, [0]), 1, 3, matrix); xi_a = np.abs(c_a) ** 2 / 3
    charges = np.array([u % 3 for u in range(9)])
    q_a = np.repeat(charges, 9); q_b = np.tile(charges, 9)
    for s in distances:
        c_b = chis(rdm(psi, L, 3, [int(s)]), 1, 3, matrix); xi_b = np.abs(c_b) ** 2 / 3
        c_ab = chis(rdm(psi, L, 3, [0, int(s)]), 2, 3, matrix).reshape(9, 9); xi_ab = np.abs(c_ab) ** 2 / 9
        e_part = 0.5 * (c_ab + c_ab[conjugate, :]); o_part = 0.5 * (c_ab - c_ab[conjugate, :])
        disconnected = np.outer(c_a, c_b)
        leakage.append(float(xi_ab.ravel()[((q_a + q_b) % 3) != 0].sum()))
        for a in alphas:
            denominator = np.sum(xi_a ** a) * np.sum(xi_b ** a)
            full[a].append(np.sum(xi_ab ** a) / denominator - 1.0)
            even[a].append(np.sum((np.abs(e_part) ** 2 / 9) ** a) / denominator - 1.0)
            abs_e = np.abs(e_part); abs_o2 = np.abs(o_part) ** 2; re_eo = np.real(np.conj(e_part) * o_part)
            nz = abs_e > 1e-13
            quadratic[a].append((a * abs_e[nz] ** (2 * a - 2) * abs_o2[nz]
                                 + 2 * a * (a - 1) * abs_e[nz] ** (2 * a - 4) * re_eo[nz] ** 2).sum() / (9 ** a * denominator))
            linear[a].append((2 * a * np.abs(disconnected) ** (2 * a - 2)
                              * np.real(np.conj(disconnected) * o_part)).sum() / (9 ** a * denominator))
    result["max_forbidden_charge_weight"] = float(max(leakage))
    result["max_parity_odd_linear"] = float(max(abs(x) for a in alphas for x in linear[a]))
    for key, series in (("full_ratio_exponents", full), ("even_sector_exponents", even),
                        ("quadratic_weighted_exponents", quadratic)):
        result[key] = {str(a): float(contiguous_window_slopes(log_eta, series[a]).mean()) for a in alphas}
    result["full_ratio_exponent_sd"] = {str(a): float(contiguous_window_slopes(log_eta, full[a]).std()) for a in alphas}
    return result


def cmd_blume_capel(args):
    rows = []
    for lam in (float(v) for v in args.lambdas.split(",")):
        for L in (int(v) for v in args.sizes.split(",")):
            rows.append(blume_capel(L, lam)); r = rows[-1]
            print(f"lambda={lam}, L={L}: forbidden (qutrit phase-charge) weight {r['max_forbidden_charge_weight']:.4f}, "
                  f"max parity-odd linear term {r['max_parity_odd_linear']:.1e}")
            print("  mean full-ratio powers (alpha=2,3,4): "
                  + ", ".join(f"{r['full_ratio_exponents'][str(a)]:.3f}" for a in (2, 3, 4))
                  + f";  scaled gap {r['scaled_gap']:.3f}, c_eff {r['c_eff_vn']:.3f}", flush=True)
    return rows


def cmd_all(args):
    record = {}
    for title, command, key in (("replica: eq. (3.5)", cmd_replica, None),
                                ("fits: Table 2 and Figure 3", cmd_fits, None),
                                ("fractions: section 5.4", cmd_fractions, "sector_fractions"),
                                ("blume-capel: section 5.4", cmd_blume_capel, "blume_capel")):
        print(f"\n===== {title} =====\n", flush=True)
        rows = command(args)
        if key:
            record[key] = rows
    if args.output:
        for key, rows in record.items():
            save_record(args.output, key, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {"all": cmd_all, "replica": cmd_replica, "fits": cmd_fits, "fractions": cmd_fractions,
                "finite-size": cmd_finite_size, "blume-capel": cmd_blume_capel}
    parsers = {name: sub.add_parser(name) for name in commands}
    for name in ("all", "fractions"):
        parsers[name].add_argument("--models", default="3:12,4:10,5:8")
    for name in ("all", "blume-capel"):
        parsers[name].add_argument("--sizes", default="12")
        parsers[name].add_argument("--lambdas", default="1.326")
    for name in ("all", "fractions", "blume-capel"):
        parsers[name].add_argument("--output", help="write the rows into this JSON record")
    parsers["finite-size"].add_argument("--sizes", default="3:8,10,12,14;4:8,10,12;5:8,10",
                                        help="ring sizes per local dimension, e.g. '3:8,10;5:8'")
    args = parser.parse_args()
    rows = commands[args.command](args)
    if args.command in ("fractions", "blume-capel") and args.output:
        save_record(args.output, {"fractions": "sector_fractions", "blume-capel": "blume_capel"}[args.command], rows)


if __name__ == "__main__":
    main()
