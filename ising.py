"""Critical Ising chain: every Ising number in the paper except the index-two Pfaffians,
which are in pfaffian.py.

    python3 ising.py all > results/ising.txt                    fast checks (~10 s)
    python3 ising.py charged-coefficients --output results/records.json   Table 1 (~7 min)
    python3 ising.py neutral --output results/records.json        ell*A_1 (hours at ell = 16)
    python3 ising.py fixed-cross-ratio --output results/records.json

Subcommands, in the order of the paper:
  rotated            section 3.3, appendix A.4: the rotated two-site moment
  charged-coefficients  Table 1: exact charged coefficients C_alpha(ell), eq. (5.2)
  diagnostics        section 5.3 numbers derived from the Table 1 record
  neutral            linear neutral coefficient A_1(ell), eq. (A.9), by direct enumeration
  identities         appendices A.1-A.2: Cauchy-ensemble identities, inverse sums, the
                     weighted involution, and a two-block check of eqs. (A.8)-(A.9)
  length-two         Figure 2(a): length-two charged ratios (also the Ising row of Table 2)
  sectors            Figure 2(b): one-site charged, neutral and full ratios
  single-interval    section 5.5 and Figure 4
  full-system        section 5.5: periodic-chain constants, eq. (4.16)
  fixed-cross-ratio  section 6: L_LR at eta = 1/4, eq. (6.1)
  junction           appendix B: linear algebra of the index-two orbifold junction

The free-fermion calculations use the d = 2 clock convention H = -sum(Z Z + X), with Majorana
covariance <i a_m b_n> = -2/[pi(2(n-m)+1)].  The paper writes H = -sum(X X + Z), its
local-Hadamard image, which leaves every moment unchanged; `rotated` uses the paper's
convention directly.  Pauli strings keep their Jordan--Wigner tails across gaps.
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # multithreaded BLAS is pathologically slow on many small determinants

import argparse
import itertools
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from numpy.linalg import slogdet
from scipy.special import gammaln

HERE = Path(__file__).resolve().parent
RECORD = HERE / "results" / "records.json"
M_Z = 2 / math.pi
A_SIGMA = float(mp.power(2, mp.mpf(1) / 12) * mp.e ** (3 * mp.diff(mp.zeta, -1)))


# ----------------------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------------------

def load_record(path=RECORD):
    return json.load(open(path, encoding="utf-8")) if Path(path).exists() else {}


def save_record(path, key, rows, ident):
    """Merge rows into record[key], matching rows by ident(row) and updating their fields."""
    record = load_record(path)
    merged = {ident(row): row for row in record.get(key, [])}
    for row in rows:
        merged[ident(row)] = {**merged.get(ident(row), {}), **row}
    record[key] = [merged[k] for k in sorted(merged)]
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=1, sort_keys=True)


# ----------------------------------------------------------------------------------------
# Infinite chain: Majorana covariance, Pauli masks with Jordan--Wigner tails, Weyl weights
# ----------------------------------------------------------------------------------------

def covariance(N):
    """Exact critical Majorana covariance of the infinite chain, restricted to N sites."""
    G = np.zeros((2 * N, 2 * N)); n = np.arange(N)
    K = lambda d: -2.0 / (np.pi * (2 * d + 1))
    for i in range(N):
        G[2 * i, 1::2] = K(n - i); G[1::2, 2 * i] = -K(n - i)
    return G


def pauli_masks(N):
    """Majorana bit masks of 1, X, Y, Z on each site, including the Jordan--Wigner tail."""
    tail = [0] * (N + 1)
    for i in range(N):
        tail[i + 1] = tail[i] | (1 << (2 * i)) | (1 << (2 * i + 1))
    return [[0, (1 << (2 * i)) | (1 << (2 * i + 1)), tail[i] | (1 << (2 * i + 1)), tail[i] | (1 << (2 * i))]
            for i in range(N)]


def xi(G, masks, sites):
    """Weights Xi(u) = |<W_u>|^2 / 2^n of every Pauli string on the given sites."""
    n = len(sites); out = np.zeros(4 ** n)
    for idx in range(4 ** n):
        m = 0
        for k, s in enumerate(sites):
            m ^= masks[s][(idx // 4 ** k) % 4]
        S = np.flatnonzero(np.array([(m >> j) & 1 for j in range(G.shape[0])]))
        if len(S) % 2:
            continue
        out[idx] = 1.0 if len(S) == 0 else np.exp(slogdet(G[np.ix_(S, S)])[1])
    return out / 2 ** n


def odd(idx, n):
    """Z2 charge of a Pauli label: the number of X or Y factors mod 2 (clock convention)."""
    return sum(1 for k in range(n) if ((idx // 4 ** k) % 4) in (2, 3)) % 2


# ----------------------------------------------------------------------------------------
# One-interval Cauchy sums (appendix A.1)
# ----------------------------------------------------------------------------------------

def subsets(ell, size):
    return np.asarray(list(itertools.combinations(range(ell), size)), dtype=int)


def log_vandermonde(rows):
    out = np.zeros(len(rows))
    for left in range(rows.shape[1]):
        for right in range(left + 1, rows.shape[1]):
            out += np.log(rows[:, right] - rows[:, left])
    return out


def charged_coefficient(ell, alphas=(2, 3, 4), batch_size=50000):
    """T_alpha^C, Sigma_alpha^C and C_alpha(ell) from eqs. (A.1), (A.2) and (5.2)."""
    totals = {alpha: 1.0 for alpha in alphas}
    overlaps = {alpha: 0.0 for alpha in alphas}

    # Equal-cardinality Cauchy minors for T_alpha^C.
    for size in range(1, ell + 1):
        left_sets = subsets(ell, size); right_sets = subsets(ell, size)
        right_v = log_vandermonde(right_sets)
        for left in left_sets:
            left_v = sum(np.log(left[j] - left[i]) for i in range(size) for j in range(i + 1, size))
            for start in range(0, len(right_sets), batch_size):
                right = right_sets[start:start + batch_size]
                denominator = np.log(np.abs(right[:, None, :] - left[None, :, None] + 0.5)).sum(axis=(1, 2))
                log_abs_det = -size * np.log(np.pi) + left_v + right_v[start:start + batch_size] - denominator
                for alpha in alphas:
                    totals[alpha] += np.exp(2 * alpha * log_abs_det).sum()

    # Order-field overlaps: |I| = k and |J| = k + 1.
    for size in range(0, ell):
        left_sets = subsets(ell, size); right_sets = subsets(ell, size + 1)
        right_v = log_vandermonde(right_sets)
        right_gamma = (gammaln(ell - right_sets - 0.5) - gammaln(ell - right_sets)).sum(axis=1)
        for left in left_sets:
            left_v = sum(np.log(left[j] - left[i]) for i in range(size) for j in range(i + 1, size))
            left_gamma = (gammaln(ell - left + 0.5) - gammaln(ell - left)).sum() if size else 0.0
            for start in range(0, len(right_sets), batch_size):
                right = right_sets[start:start + batch_size]
                if size:
                    denominator = np.log(np.abs(right[:, None, :] - left[None, :, None] + 0.5)).sum(axis=(1, 2))
                else:
                    denominator = np.zeros(len(right))
                log_abs_overlap = (-size * np.log(np.pi) - 0.5 * np.log(np.pi)
                                   + left_v + right_v[start:start + batch_size] - denominator
                                   + left_gamma + right_gamma[start:start + batch_size])
                for alpha in alphas:
                    overlaps[alpha] += np.exp(2 * alpha * log_abs_overlap).sum()

    result = {"ell": ell, "A_sigma": A_SIGMA}
    for alpha in alphas:
        result[f"T_{alpha}"] = float(totals[alpha])
        result[f"sum_C_{alpha}"] = float(overlaps[alpha])
        result[f"C_{alpha}"] = float(ell ** (-alpha / 2) * A_SIGMA ** (2 * alpha)
                                     * (overlaps[alpha] / totals[alpha]) ** 2)
    return result


def kernel(ell):
    i = np.arange(ell)[:, None]; j = np.arange(ell)[None, :]
    return -1.0 / (np.pi * (j - i + 0.5))


def neutral_coefficient(ell, alphas=(2, 3), batch_size=50000):
    """Linear neutral coefficient A_1(ell) of eq. (A.9), plus the isolated quadratic response.

    The Schur-complement expansion reduces both to one-interval sums T, K = sum |D|^{2a} v and
    J = sum |D|^{2a} v^2, with v_IJ from eq. (A.7); the cost is binomial(2 ell, ell).
    """
    matrix = kernel(ell)
    total = {alpha: 1.0 for alpha in alphas}
    weighted = {alpha: 0.0 for alpha in alphas}
    weighted2 = {alpha: 0.0 for alpha in alphas}
    for size in range(1, ell + 1):
        sets = subsets(ell, size)
        for left in sets:
            for start in range(0, len(sets), batch_size):
                right = sets[start:start + batch_size]
                sign, logdet = np.linalg.slogdet(matrix[left[None, :, None], right[:, None, :]])
                determinant = np.where(np.abs(sign) > 0.5, np.exp(logdet), 0.0)
                v = -np.pi * (right.sum(axis=1) - left.sum() + size / 2.0)
                for alpha in alphas:
                    weight = determinant ** (2 * alpha)
                    total[alpha] += weight.sum()
                    weighted[alpha] += np.sum(weight * v)
                    weighted2[alpha] += np.sum(weight * v ** 2)
    result = {"ell": ell}
    for alpha in alphas:
        amplitude = 2.0 * alpha / (np.pi ** 2 * ell ** 2) * (weighted[alpha] / total[alpha]) ** 2
        quadratic = alpha * (2 * alpha - 1) / (np.pi ** 4 * ell ** 4) * (weighted2[alpha] / total[alpha]) ** 2
        result[f"T_{alpha}"] = float(total[alpha])
        result[f"K_{alpha}"] = float(weighted[alpha])
        result[f"J_{alpha}"] = float(weighted2[alpha])
        result[f"A_neut_{alpha}"] = float(amplitude)
        result[f"ell_A_neut_{alpha}"] = float(ell * amplitude)
        result[f"B_neut_{alpha}"] = float(quadratic)
    return result


def parse_alpha(value):
    alpha = float(value)
    if alpha <= 0:
        raise argparse.ArgumentTypeError("moment indices must be positive")
    return int(alpha) if alpha.is_integer() else alpha


# ----------------------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------------------

def cmd_charged_coefficients(args):
    alphas = tuple(parse_alpha(a) for a in args.alphas.split(","))
    rows = []
    for ell in (int(x) for x in args.lengths.split(",")):
        rows.append(charged_coefficient(ell, alphas=alphas))
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    if args.output:
        save_record(args.output, "charged_coefficients", rows, lambda r: r["ell"])


def cmd_neutral(args):
    alphas = tuple(int(a) for a in args.alphas.split(","))
    rows = []
    for ell in (int(x) for x in args.lengths.split(",")):
        rows.append(neutral_coefficient(ell, alphas=alphas))
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    if args.output:
        save_record(args.output, "neutral", rows, lambda r: r["ell"])


def cmd_diagnostics(args):
    rows = {row["ell"]: row for row in load_record(args.record)["charged_coefficients"]}
    lengths = sorted(rows)
    alphas = sorted(int(key[2:]) for key in rows[lengths[-1]] if key.startswith("C_"))
    print("Table 1, C_alpha(ell):")
    print("  alpha " + "".join(f"{'ell=' + str(ell):>14}" for ell in lengths))
    for alpha in alphas:
        print(f"  {alpha:>5} " + "".join(f"{rows[ell][f'C_{alpha}']:>14.6g}" for ell in lengths))
    high = [alpha for alpha in (5, 6, 8, 10) if alpha in alphas]
    if 16 in rows:
        row = rows[16]
        print("\nell = 16:  alpha   U/ell   identity fraction 1/T   single-string fraction ell/Sigma")
        for alpha in high:
            total, charged = row[f"T_{alpha}"], row[f"sum_C_{alpha}"]
            print(f"           {alpha:>5}  {charged / total / 16:6.4f}  {1 / total:22.3f}  {16 / charged:32.3f}")
    print("\nspread of ell^(alpha/2-2) C_alpha over ell =", lengths)
    for alpha in high:
        values = [rows[ell][f"C_{alpha}"] * ell ** (alpha / 2 - 2) for ell in lengths]
        print(f"  alpha = {alpha:>2}: max/min - 1 = {100 * (max(values) / min(values) - 1):.2f}%")
    if 12 in rows and all(f"T_{alpha}" in rows[12] for alpha in (2, 5)):
        print(f"\ntwelve-site sums: T_2(12) = {rows[12]['T_2']:.3f},  T_5(12) = {rows[12]['T_5']:.4f}")
    print("\nlength at which the bound (5.16), (1 + m_z^(2 alpha))^ell, reaches e:")
    for alpha in (5, 10):
        print(f"  alpha = {alpha:>2}: {1 / math.log1p(M_Z ** (2 * alpha)):.1f}")


def _vandermonde(values):
    result = 1.0
    for left, value in enumerate(values):
        for other in values[left + 1:]:
            result *= other - value
    return result


def _ensemble_errors(ell):
    """Term-by-term check of eqs. (A.3) and (A.4) against the Cauchy forms (A.1), (A.2)."""
    x = [float(i) for i in range(ell)]; y = [i + 0.5 for i in range(ell)]
    endpoint = math.prod(math.gamma(n - 0.5) / math.gamma(n) for n in range(1, ell + 1))
    w0 = lambda u: 1.0 / (math.gamma(u + 0.5) * math.gamma(ell - u + 0.5))
    w1 = lambda u: 1.0 / (math.gamma(u + 0.5) * math.gamma(ell - u))
    neutral_error = charged_error = 0.0
    for size in range(ell + 1):
        for left in itertools.combinations(range(ell), size):
            for right in itertools.combinations(range(ell), size):
                denominator = math.prod(abs(j - i + 0.5) for i in left for j in right)
                direct = math.pi ** (-size) * _vandermonde(list(left)) * _vandermonde(list(right)) / denominator
                roots = sorted([x[i] for i in left] + [y[j] for j in range(ell) if j not in right])
                ensemble = _vandermonde(y) * _vandermonde(roots) * math.prod(w0(u) for u in roots)
                neutral_error = max(neutral_error, abs(direct - ensemble))
    for size in range(ell):
        for left in itertools.combinations(range(ell), size):
            for right in itertools.combinations(range(ell), size + 1):
                denominator = math.prod(abs(j - i + 0.5) for i in left for j in right)
                direct = (math.pi ** (-size) / math.sqrt(math.pi)
                          * _vandermonde(list(left)) * _vandermonde(list(right)) / denominator)
                direct *= math.prod(math.gamma(ell - i + 0.5) / math.gamma(ell - i) for i in left)
                direct *= math.prod(math.gamma(ell - j - 0.5) / math.gamma(ell - j) for j in right)
                roots = sorted([x[i] for i in left] + [y[j] for j in range(ell) if j not in right])
                ensemble = (_vandermonde(y) * endpoint / math.sqrt(math.pi)
                            * _vandermonde(roots) * math.prod(w1(u) for u in roots))
                charged_error = max(charged_error, abs(direct - ensemble))
    return neutral_error, charged_error


def _involution_sums(ell, alpha):
    """Full and endpoint-restricted sums of eq. (A.11), and the error of eq. (A.7)."""
    full = endpoint = 0.0; error = 0.0
    for size in range(1, ell + 1):
        for I in itertools.combinations(range(ell), size):
            for J in itertools.combinations(range(ell), size):
                g = kernel(ell)[np.ix_(I, J)]
                weight = abs(np.linalg.det(g)) ** (2 * alpha)
                v = np.ones(size) @ np.linalg.solve(g, np.ones(size))
                error = max(error, abs(v + np.pi * (sum(J) - sum(I) + size / 2)))
                full += weight * v
                if 0 in I:
                    endpoint += weight * v
    return full, endpoint, error


def _two_block_check(ell=2, alpha=2, separations=(400, 800)):
    """r^2 (neutral ratio - 1) for two blocks, against 2 alpha vbar^2 / pi^2 of eq. (A.9)."""
    mp.mp.dps = 40
    def det(I, J):
        if not I:
            return mp.mpf(1)
        return mp.det(mp.matrix([[-1 / (mp.pi * (j - i + mp.mpf(1) / 2)) for j in J] for i in I]))
    def labels(sites):
        return [(I, J) for k in range(len(sites) + 1)
                for I in itertools.combinations(sites, k) for J in itertools.combinations(sites, k)]
    block = labels(list(range(ell)))
    total = sum(abs(det(I, J)) ** (2 * alpha) for I, J in block)
    vbar = sum(abs(det(I, J)) ** (2 * alpha) * (-mp.pi * (sum(J) - sum(I) + mp.mpf(len(I)) / 2))
               for I, J in block if I) / total
    values = []
    for r in separations:
        s = sum(abs(det(Ia + Ib, Ja + Jb)) ** (2 * alpha)
                for Ia, Ja in block for Ib, Jb in labels(list(range(r, r + ell))))
        values.append((s / total ** 2 - 1) * r ** 2)
    return values, (4 * values[1] - values[0]) / 3, 2 * alpha * vbar ** 2 / mp.pi ** 2


def cmd_identities(args):
    print("appendix A.1: ensemble identities (A.3)-(A.4), maximum absolute error per term")
    print("  ell   neutral (A.3)   charged (A.4)")
    for ell in range(1, 7):
        n_err, c_err = _ensemble_errors(ell)
        print(f"  {ell:>3}   {n_err:13.3e}   {c_err:13.3e}")
    ell = 2
    overlaps = []
    for size in range(ell):
        for I in itertools.combinations(range(ell), size):
            for J in itertools.combinations(range(ell), size + 1):
                value = (math.pi ** (-size) / math.sqrt(math.pi) * _vandermonde(list(I)) * _vandermonde(list(J))
                         / math.prod(abs(j - i + 0.5) for i in I for j in J))
                value *= math.prod(math.gamma(ell - i + 0.5) / math.gamma(ell - i) for i in I)
                value *= math.prod(math.gamma(ell - j - 0.5) / math.gamma(ell - j) for j in J)
                overlaps.append(value)
    print("  nonzero leading overlaps |C_IJ| at ell = 2: " + ", ".join(f"{v:.6f}" for v in sorted(overlaps, reverse=True)))

    print("\nappendix A.2: inverse sums (A.7) and the endpoint identity (A.11)")
    print("  ell  alpha   max error (A.7)   full sum            endpoint-only sum   difference")
    for ell in (3, 4, 5):
        for alpha in (0.7, 2, 2.5, 3):
            full, endpoint, error = _involution_sums(ell, alpha)
            print(f"  {ell:>3}  {alpha:>5}   {error:15.1e}   {full:18.15f}  {endpoint:18.15f}  {full - endpoint:10.1e}")

    values, extrapolated, predicted = _two_block_check()
    print("\nappendix A.2: two-block check of eqs. (A.8)-(A.9) at ell = 2, alpha = 2")
    print(f"  r^2 (ratio - 1) at r = 400, 800: {mp.nstr(values[0], 10)}, {mp.nstr(values[1], 10)}")
    print(f"  extrapolated in 1/r^2: {mp.nstr(extrapolated, 10)};  2 alpha vbar^2/pi^2 = {mp.nstr(predicted, 10)}")


# rotated two-site moment ------------------------------------------------------------------

R_MAX = 400_000
_j = np.arange(1, R_MAX + 1, dtype=np.float64)
_l = np.log1p(-1 / (4 * _j * _j))
_S0 = np.concatenate([[0.0], np.cumsum(_l)])
_S1 = np.concatenate([[0.0], np.cumsum(_j * _l)])


def c_xx(r):
    """<X_0 X_r> = (2/pi)^r prod_{j<r} (1 - 1/(4 j^2))^(j-r), evaluated stably for large r."""
    return math.exp(r * math.log(2 / math.pi) + _S1[r - 1] - r * _S0[r - 1])


def c_zz(r):
    return M_Z ** 2 + 4 / (math.pi ** 2 * (4 * r * r - 1))


def c_yy(r):
    return -c_xx(r) / (4 * r * r - 1)


def rotated_minus_one(r, alpha):
    """Eq. (A.15)."""
    e, o, e0 = c_zz(r) / 2, c_xx(r) / 2, M_Z ** 2 / 2
    tp = 1 + 2 ** (1 - alpha) * M_Z ** (2 * alpha)
    return (2 * abs(e + o) ** (2 * alpha) + 2 * abs(e - o) ** (2 * alpha)
            - 4 * e0 ** (2 * alpha) + abs(c_yy(r)) ** (2 * alpha)) / tp ** 2


def unrotated_minus_one(r, alpha):
    """Eq. (A.18)."""
    return (abs(c_xx(r)) ** (2 * alpha) + abs(c_yy(r)) ** (2 * alpha)
            + abs(c_zz(r)) ** (2 * alpha) - M_Z ** (4 * alpha)) / (1 + M_Z ** (2 * alpha)) ** 2


def brute_force_minus_one(r, alpha):
    """Rotate the two-site density matrix by exp(-i pi Y/8) on each site and sum all 16 labels."""
    eye = np.eye(2); x = np.array([[0, 1], [1, 0]]); y = np.array([[0, -1j], [1j, 0]]); z = np.diag([1, -1])
    paulis = [eye, x, y, z]
    expectations = {(0, 0): 1, (0, 3): M_Z, (3, 0): M_Z, (3, 3): c_zz(r), (1, 1): c_xx(r), (2, 2): c_yy(r)}
    rho = sum(v * np.kron(paulis[a], paulis[b]) for (a, b), v in expectations.items()) / 4
    u = math.cos(math.pi / 8) * eye - 1j * math.sin(math.pi / 8) * y
    rho = np.kron(u, u) @ rho @ np.kron(u, u).conj().T
    one = np.trace(rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    joint = sum(abs(np.trace(rho @ np.kron(paulis[a], paulis[b]))) ** (2 * alpha) for a in range(4) for b in range(4))
    single = sum(abs(np.trace(one @ p)) ** (2 * alpha) for p in paulis)
    return joint / single ** 2 - 1


def local_exponent(r, alpha, h=0.01):
    r1, r2 = int(round(r * (1 - h))), int(round(r * (1 + h)))
    return -0.5 * (math.log(rotated_minus_one(r2, alpha)) - math.log(rotated_minus_one(r1, alpha))) / math.log(r2 / r1)


def cmd_rotated(args):
    a = args.alpha
    dev = max(abs(rotated_minus_one(r, al) / brute_force_minus_one(r, al) - 1) for r in (1, 2, 5, 10) for al in (2, 3, 4))
    print(f"eq. (A.15) vs direct rotation of the two-site state: max relative deviation {dev:.1e}")
    print(f"C_xx(r) r^(1/4) at r = {R_MAX}: {c_xx(R_MAX) * R_MAX ** 0.25:.6f}   (A_sigma = {A_SIGMA:.6f})")
    e0 = M_Z ** 2 / 2
    tp = 1 + 2 ** (1 - a) * M_Z ** (2 * a)
    coefficient = a * (2 * a - 1) * e0 ** (2 * a - 2) * A_SIGMA ** 2 / tp ** 2
    r = 300_000
    print(f"alpha = {a}: eq. (3.7) coefficient {coefficient:.6e};  (F'-1) eta^(-1/4) at r = {r}: {rotated_minus_one(r, a) * r ** 0.5:.6e}")
    rs = np.arange(32, 257)
    slope = np.polyfit(np.log(1.0 / rs ** 2), np.log([rotated_minus_one(int(s), a) for s in rs]), 1)[0]
    print(f"alpha = {a}: rotated exponent fitted over r = 32..256: {slope:.3f}   (asymptotic value 1/4)")
    for r in (100, 1_000, 10_000, 100_000, 300_000):
        print(f"  local exponent at r = {r:>7}: {local_exponent(r, a):.4f}")
    print("unrotated state, eq. (A.18): (F-1)(1+m_z^(2a))^2 = A_sigma^(2a) eta^(a/4) + (2a m_z^(4a-2)/pi^2) eta + ...")
    print("  alpha   charged coeff   neutral coeff   exact/two-term at r = 10^4   exponent min(alpha/4, 1)")
    r = 10_000; eta = 1.0 / r ** 2
    for al in (2, 3, 4, 5, 6):
        c_ch = A_SIGMA ** (2 * al); c_n = 2 * al * M_Z ** (4 * al - 2) / math.pi ** 2
        two_term = (c_ch * eta ** (al / 4) + c_n * eta) / (1 + M_Z ** (2 * al)) ** 2
        print(f"  {al:>5}   {c_ch:13.4e}   {c_n:13.4e}   {unrotated_minus_one(r, al) / two_term:26.4f}   {min(al / 4, 1):g}")


# separated intervals on the infinite chain ------------------------------------------------

LENGTH_TWO_SEPARATIONS = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def length_two_ratios(alphas=(2, 3, 4, 5), separations=LENGTH_TWO_SEPARATIONS):
    """Charged ratios Z^ch/(Z_A Z_B) and I_2(A:B) for two length-two intervals, eta = (2/s)^2."""
    l = 2
    N = 2 * (separations[-1] + l) + 4; G = covariance(N); masks = pauli_masks(N)
    xA = xi(G, masks, list(range(l)))
    eta, mutual, charged = [], [], {a: [] for a in alphas}
    for s in separations:
        B = list(range(s, s + l))
        xB = xi(G, masks, B); xAB = xi(G, masks, list(range(l)) + B)
        labels = np.arange(4 ** (2 * l))
        pA = np.array([odd(i % 4 ** l, l) for i in labels]); pB = np.array([odd(i // 4 ** l, l) for i in labels])
        eta.append((l / s) ** 2)
        mutual.append(-np.log(xA.sum()) - np.log(xB.sum()) + np.log(xAB.sum()))
        for a in alphas:
            charged[a].append((xAB[(pA == 1) & (pB == 1)] ** a).sum() / ((xA ** a).sum() * (xB ** a).sum()))
    return xA.sum(), np.array(eta), np.array(mutual), {a: np.array(v) for a, v in charged.items()}


def cmd_length_two(args):
    alphas = (2, 3, 4, 5)
    purity, E, R2, OO = length_two_ratios(alphas)
    print(f"infinite critical chain, l=2.  purity check sum(Xi_A) = {purity:.8f}")
    print(f"\n  {'r':>5} {'eta':>11} {'I_2(A:B)':>12} " + " ".join(f"{'oo/ZaZb a=' + str(a):>14}" for a in alphas))
    for k, s in enumerate(LENGTH_TWO_SEPARATIONS):
        print(f"  {s:>5} {E[k]:>11.3e} {R2[k]:>12.4e} " + "".join(f"{OO[a][k]:>14.4e} " for a in alphas))
    x = np.log(E); k = slice(-6, None)
    print(f"\n  exponent fits over the smallest six eta ({E[-6]:.1e} .. {E[-1]:.1e}):")
    p2 = np.polyfit(x[k], np.log(np.abs(R2[k])), 1)[0]
    print(f"    I_2(A:B)  : p = {p2:.5f}    [2*Delta_sigma = 0.25]        err {100 * (p2 / 0.25 - 1):+.2f}%")
    for a in alphas:
        p = np.polyfit(x[k], np.log(OO[a][k]), 1)[0]
        print(f"    a={a} odd-odd : p = {p:.5f}    [2*a*Delta_sigma = {a / 4:.2f}]   err {100 * (p / (a / 4) - 1):+.2f}%")


def cmd_sectors(args):
    l = 1; al = (2, 3, 4, 5, 6); seps = [8, 16, 32, 64, 128, 256, 512]
    N = 2 * (seps[-1] + l) + 4; G = covariance(N); masks = pauli_masks(N)
    xA = xi(G, masks, list(range(l)))
    print(f"infinite critical Ising, l={l}: charged / neutral-connected / total\n")
    print(f"  {'eta':>10} " + " ".join(f"{'a=' + str(a):>34}" for a in al))
    print(f"  {'':>10} " + " ".join(f"{'chg':>10}{'neut':>12}{'total':>12}" for a in al))
    E = []; C = {a: [] for a in al}; Nn = {a: [] for a in al}; T = {a: [] for a in al}
    for s in seps:
        B = list(range(s, s + l)); xB = xi(G, masks, B); xAB = xi(G, masks, list(range(l)) + B)
        NN = 4 ** l; ia = np.arange(NN * NN)
        pA = np.array([odd(i % NN, l) for i in ia]); pB = np.array([odd(i // NN, l) for i in ia])
        eta = (l / s) ** 2; E.append(eta); row = f"  {eta:>10.2e} "
        for a in al:
            ZZ = (xA ** a).sum() * (xB ** a).sum()
            chg = (xAB[(pA == 1) & (pB == 1)] ** a).sum() / ZZ
            neu = (xAB[(pA == 0) & (pB == 0)] ** a).sum() / ZZ - 1.0      # deviation from factorisation
            tot = (xAB ** a).sum() / ZZ - 1.0
            C[a].append(chg); Nn[a].append(neu); T[a].append(tot)
            row += f"{chg:>10.2e}{neu:>12.2e}{tot:>12.2e}"
        print(row)
    record = load_record()
    record["sectors"] = [dict(eta=e, **{f"{t}_a{a}": v[a][i] for a in al for t, v in (("chg", C), ("neut", Nn), ("tot", T))})
                         for i, e in enumerate(E)]
    with open(RECORD, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=1, sort_keys=True)
    print("  wrote the sectors record to results/records.json")
    x = np.log(E); k = slice(-4, None)
    print(f"\n  exponents fitted over the four smallest eta ({E[-4]:.1e}..{E[-1]:.1e}):")
    for a in al:
        pc = np.polyfit(x[k], np.log(np.abs(np.array(C[a])[k])), 1)[0]
        pn = np.polyfit(x[k], np.log(np.abs(np.array(Nn[a])[k])), 1)[0]
        pt = np.polyfit(x[k], np.log(np.abs(np.array(T[a])[k])), 1)[0]
        print(f"    a={a}:  charged {pc:6.3f} [2a*D_sigma={a / 4:.2f}]   neutral {pn:6.3f} [D_eps=1]"
              f"   TOTAL {pt:6.3f} [min = {min(a / 4, 1.0):.2f}]")


def cmd_single_interval(args):
    """J_M(ell) = 2 M(ell) - M(2 ell) for M = S_2 = ell log 2 - log det(1 + G G^T) and
    hat M_2 = ell log 2 - log T_2(ell); local slopes plotted at 1/(ell+1/2), ell = 2..5."""
    s2, m2 = {}, {}
    for ell in range(1, 13):
        g = kernel(ell)
        s2[ell] = ell * math.log(2) - math.log(np.linalg.det(np.eye(ell) + g @ g.T))
        m2[ell] = ell * math.log(2) - math.log(charged_coefficient(ell, alphas=(2,))["T_2"])
    j_of = lambda m, ell: 2 * m[ell] - m[2 * ell]
    ells = [2, 3, 4, 5]
    x = np.array([1 / (ell + 0.5) for ell in ells])
    slope = lambda m: np.array([(j_of(m, ell + 1) - j_of(m, ell)) / math.log((ell + 1) / ell) for ell in ells])
    ent, mom = slope(s2), slope(m2)
    print("ell   1/(ell+1/2)   local slope J_{S_2}   local slope J_{hat M_2}")
    for ell, xv, a, b in zip(ells, x, ent, mom):
        print(f"{ell:>3}   {xv:11.4f}   {a:19.5f}   {b:23.5f}")
    for name, y in (("J_{S_2} (Renyi-2 entanglement)", ent), ("J_{hat M_2} (stabilizer series)", mom)):
        drift, intercept = np.polyfit(x, y, 1)
        print(f"{name}: linear-fit intercept {intercept:.4f}, drift {drift:+.4f}   (c/4 = 0.125)")
    diff = mom - ent
    print(f"slope-difference intercept: all four points {np.polyfit(x, diff, 1)[1]:.2e}, "
          f"last three {np.polyfit(x[1:], diff[1:], 1)[1]:.2e}")
    print(f"T_2(12) = {math.exp(12 * math.log(2) - m2[12]):.3f}")


def cmd_full_system(args):
    """Fit hat M_alpha(full) = m_alpha L + b over L = 6..12 on the periodic chain
    H = -sum Z Z - sum X, and compare b with -log(g_1)/(alpha-1) of eq. (4.16)."""
    from scipy.sparse import csr_matrix, identity, kron
    from scipy.sparse.linalg import eigsh
    I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1]).astype(float)
    M = np.zeros((4, 4), complex)
    for u, P in enumerate([I2, X, Y, Z]):
        for i in range(2):
            for j in range(2):
                M[u, 2 * i + j] = P[j, i]

    def chis_full(rho, n):
        T = rho.reshape((2,) * (2 * n)); perm = []
        for k in range(n):
            perm += [k, n + k]
        T = np.transpose(T, perm).reshape((4,) * n)
        for ax in range(n):
            T = np.moveaxis(np.tensordot(M, T, axes=([1], [ax])), 0, ax)
        return T.reshape(-1)

    def ground_state(L):
        def op(ops):
            m = identity(1, format="csr")
            for k in range(L):
                m = kron(m, csr_matrix(ops.get(k, I2)), format="csr")
            return m
        H = sum(-op({i: Z, (i + 1) % L: Z}) - op({i: X}) for i in range(L))
        _, vectors = eigsh(H, k=1, which="SA")
        return np.asarray(vectors[:, 0]).ravel()

    alphas = (2, 3, 4, 5, 6); sizes = [6, 8, 10, 12]; data = {a: [] for a in alphas}
    for L in sizes:
        psi = ground_state(L); rho = np.outer(psi, psi.conj())
        weights = np.abs(chis_full(rho, L)) ** 2 / 2 ** L
        assert abs(weights.sum() - 1) < 1e-8, weights.sum()
        for a in alphas:
            data[a].append(np.log((weights ** a).sum()) / (1 - a) - L * np.log(2))
        print(f"  L={L:2d}  " + "  ".join(f"M_{a}={data[a][-1]:.6f}" for a in alphas), flush=True)
    print(f"\n  fit M_alpha = m L + b over L = {sizes}")
    print(f"  {'alpha':>5} {'fitted b':>10} {'predicted':>10} {'rel. dev.':>10}")
    for a in alphas:
        _, b = np.polyfit(sizes, data[a], 1)
        predicted = -np.log(np.sqrt(a) if a <= 4 else 2.0) / (a - 1)
        print(f"  {a:>5} {b:10.5f} {predicted:10.5f} {100 * abs(b / predicted - 1):9.1f}%")


def fixed_cross_ratio(ell, multiple, alphas):
    """Two intervals of length ell with gap multiple*ell, so eta = 1/(1+multiple)^2."""
    gap = multiple * ell
    sites_a = list(range(ell)); sites_b = list(range(ell + gap, 2 * ell + gap))
    G = covariance(2 * ell + gap + 2); masks = pauli_masks(2 * ell + gap + 2)
    xi_a = xi(G, masks, sites_a); xi_b = xi(G, masks, sites_b); xi_ab = xi(G, masks, sites_a + sites_b)
    dim_a = 4 ** ell; labels = np.arange(dim_a * dim_a)
    parity_a = np.array([odd(index % dim_a, ell) for index in labels])
    parity_b = np.array([odd(index // dim_a, ell) for index in labels])
    charged = (parity_a == 1) & (parity_b == 1); neutral = (parity_a == 0) & (parity_b == 0)
    result = {"ell": ell, "gap_multiple": multiple, "eta": 1.0 / (1.0 + multiple) ** 2,
              "purity_check_a": float(np.sum(xi_a)), "forbidden_weight": float(np.sum(xi_ab[~(charged | neutral)]))}
    for alpha in alphas:
        denominator = np.sum(xi_a ** alpha) * np.sum(xi_b ** alpha)
        result[f"charged_{alpha}"] = float(np.sum(xi_ab[charged] ** alpha) / denominator)
        result[f"neutral_connected_{alpha}"] = float(np.sum(xi_ab[neutral] ** alpha) / denominator - 1.0)
        result[f"full_connected_{alpha}"] = float(np.sum(xi_ab ** alpha) / denominator - 1.0)
    result["I2"] = float(np.log(np.sum(xi_ab) / (np.sum(xi_a) * np.sum(xi_b))))
    result["I2_stab"] = float(np.log(1.0 + result["full_connected_2"]))
    result["long_range_A2"] = result["I2"] - result["I2_stab"]
    return result


def cmd_fixed_cross_ratio(args):
    alphas = [int(v) for v in args.alphas.split(",")]
    rows = []
    for ell in (int(v) for v in args.lengths.split(",")):
        for multiple in (int(v) for v in args.multiples.split(",")):
            rows.append(fixed_cross_ratio(ell, multiple, alphas))
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    quarter = [r for r in rows if r["gap_multiple"] == 1]
    if quarter:
        print("L_LR at eta = 1/4, eq. (6.1): " + ", ".join(f"ell={r['ell']}: {r['long_range_A2']:.4f}" for r in quarter))
    if args.output:
        save_record(args.output, "fixed_cross_ratio", rows, lambda r: (r["ell"], r["gap_multiple"]))


def cmd_junction(args):
    """Appendix B: the index-two orbifold junction, eqs. (5.8) and (B.2)-(B.7)."""
    eye2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1])
    k4 = lambda A: np.kron(np.kron(A, A), np.kron(A, A))
    Q = (k4(eye2) + k4(X) + k4(Y) + k4(Z)) / 4
    print("eq. (5.8): Q_2 = (1 + X^4)(1 + Z^4)/4 error %.1e, Q^2 - Q error %.1e, rank %d"
          % (abs(Q - (k4(eye2) + k4(X)) @ (k4(eye2) + k4(Z)) / 4).max(), abs(Q @ Q - Q).max(), np.linalg.matrix_rank(Q)))

    P0 = 0.5 * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    v = 0.5 * np.ones(4); P1 = np.outer(v, v)
    R0 = 2 * P0 - np.eye(4); R1 = 2 * P1 - np.eye(4)
    signs = list(itertools.product([1, -1], repeat=4))
    H0 = [b for b in signs if np.allclose(np.diag(b) @ P0 @ np.diag(b), P0)]
    H1 = [b for b in signs if np.allclose(np.diag(b) @ P1 @ np.diag(b), P1)]
    print(f"eq. (B.2): stabilizers G_P0 = {H0}, G_P1 = {H1}")
    print("image sector (d1,d2)   eigenvalues of R_0 s R_1 s        oscillator ground weight")
    show = lambda z: (f"{z.real:+.0f}" if abs(z.imag) < 1e-9 else ("+i" if z.imag > 0 else "-i"))
    sectors = {}
    for s in signs:
        ev = np.linalg.eigvals(R0 @ np.diag(s) @ R1 @ np.diag(s))
        nu = np.mod(np.angle(ev) / (2 * np.pi), 1.0)
        weight = sum(n * (1 - n) / 4 for n in nu)       # e^{2 pi i nu} per real boson: nu(1-nu)/4
        key = (s[0] * s[2], s[1] * s[3])
        sectors.setdefault(key, set()).add((" ".join(sorted(show(z) for z in ev)), round(weight, 8)))
    for key in sorted(sectors, reverse=True):
        for ev, weight in sectors[key]:
            print(f"  {str(key):<20} {ev:<32} {weight:.6f}")

    rng = np.random.default_rng(1)
    c = rng.normal(size=2) + 1j * rng.normal(size=2); c /= np.linalg.norm(c)
    mirror = np.kron(np.kron(c, c), np.kron(c.conj(), c.conj()))
    t1 = np.zeros(16); t1[0] = t1[15] = 1 / np.sqrt(2)
    t0 = np.zeros(16)
    for a, b in itertools.product((0, 1), repeat=2):
        t0[8 * a + 4 * b + 2 * a + b] = 0.5
    print(f"mirror overlaps for a random unit twist state: <t_1|mu> = {(t1 @ mirror).real:.12f}"
          f" vs (|mu_0|^4+|mu_pi|^4)/sqrt2 = {(abs(c[0]) ** 4 + abs(c[1]) ** 4) / np.sqrt(2):.12f};"
          f" <t_0|mu> = {(t0 @ mirror).real:.12f} vs 1/2")

    ell = 1.0
    worst = 0.0
    for r in (1.5, 3.0, 10.0, 100.0):
        z = mp.mpf(r); w = (2 * z - ell + 2 * mp.sqrt(z * (z - ell))) / ell
        dw = mp.diff(lambda t: (2 * t - ell + 2 * mp.sqrt(t * (t - ell))) / ell, z)
        worst = max(worst, abs(dw / (w ** 2 - 1) / (ell / (4 * r * (r - ell))) - 1))
    print(f"eq. (B.4): w'/(w^2-1) = l/(4r(r-l)) on r > l, max relative error {float(worst):.1e}")

    lam = 0.7; phi = rng.uniform(0, 2 * np.pi, size=2)
    cc = np.cos(phi[0]) * np.cos(phi[1]); ss = np.sin(phi[0]) * np.sin(phi[1])     # c1 c2 and s1 s2
    average = np.mean([np.exp(lam * np.cos(s1 * phi[0] - s2 * phi[1])) for s1 in (1, -1) for s2 in (1, -1)])
    print(f"eq. (B.6): reflection average {average:.12f} = e^(lam c1 c2) cosh(lam s1 s2) {np.exp(lam * cc) * np.cosh(lam * ss):.12f}"
          f" != e^(lam c1 c2) {np.exp(lam * cc):.12f}")

    mp.mp.dps = 30
    dn = lambda eta: (1 - eta) ** (-mp.mpf(1) / 8) * (2 / mp.pi * mp.ellipk(eta)) ** (-mp.mpf(1) / 2)
    nome = lambda eta: mp.exp(-mp.pi * mp.ellipk(1 - eta) / mp.ellipk(eta))
    print("eq. (B.7): eta   (F_DN - 1 - eta^2/128)/eta^3   theta_3(0,q) F_DN (1-eta)^(1/8) - 1")
    for eta in (mp.mpf("1e-2"), mp.mpf("1e-3")):
        print(f"           {mp.nstr(eta, 2):<6}{mp.nstr((dn(eta) - 1 - eta ** 2 / 128) / eta ** 3, 8):>20}"
              f"{mp.nstr(mp.jtheta(3, 0, nome(eta)) * dn(eta) * (1 - eta) ** (mp.mpf(1) / 8) - 1, 3):>26}")


def cmd_all(args):
    for title, command in (("identities: appendices A.1-A.2", cmd_identities),
                           ("diagnostics: section 5.3 and Table 1", cmd_diagnostics),
                           ("rotated: section 3.3 and appendix A.4", cmd_rotated),
                           ("length-two: Figure 2(a)", cmd_length_two),
                           ("sectors: Figure 2(b)", cmd_sectors),
                           ("single-interval: section 5.5 and Figure 4", cmd_single_interval),
                           ("full-system: section 5.5, eq. (4.16)", cmd_full_system),
                           ("junction: appendix B", cmd_junction)):
        print(f"\n===== {title} =====\n", flush=True)
        command(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {"all": cmd_all, "identities": cmd_identities, "diagnostics": cmd_diagnostics,
                "rotated": cmd_rotated, "length-two": cmd_length_two, "sectors": cmd_sectors,
                "single-interval": cmd_single_interval, "full-system": cmd_full_system,
                "junction": cmd_junction, "charged-coefficients": cmd_charged_coefficients, "neutral": cmd_neutral,
                "fixed-cross-ratio": cmd_fixed_cross_ratio}
    parsers = {name: sub.add_parser(name) for name in commands}
    for name in ("all", "diagnostics"):
        parsers[name].add_argument("--record", default=RECORD)
    for name in ("all", "rotated"):
        parsers[name].add_argument("--alpha", type=int, default=4)
    parsers["charged-coefficients"].add_argument("--lengths", default="4,8,12,16")
    parsers["charged-coefficients"].add_argument("--alphas", default="2,3,4,5,6,8,10")
    parsers["neutral"].add_argument("--lengths", default=",".join(str(n) for n in range(1, 17)))
    parsers["neutral"].add_argument("--alphas", default="2,3")
    parsers["fixed-cross-ratio"].add_argument("--lengths", default="1,2,3,4")
    parsers["fixed-cross-ratio"].add_argument("--multiples", default="1,2,3")
    parsers["fixed-cross-ratio"].add_argument("--alphas", default="2,4,5,6")
    for name in ("charged-coefficients", "neutral", "fixed-cross-ratio"):
        parsers[name].add_argument("--output", help="merge the rows into this JSON record")
    args = parser.parse_args()
    commands[args.command](args)


if __name__ == "__main__":
    main()
