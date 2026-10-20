"""Index-two Ising coefficients for growing intervals from two Pfaffians: section 5.2,
Figure 1, eqs. (5.6), (5.7), (5.10), (5.13) and (A.12)-(A.14).

    python3 pfaffian.py --verify --check-precision > results/pfaffian.txt     (~15 min)
    python3 pfaffian.py --lengths 4,8,16                                       (seconds)

At alpha = 2 the finite ensemble (5.3) is a Pfaffian of skew moments,
    Z_N[w] = Pf M^(N)[w],   M_ij = sum_{u in Lambda_ell} w(u)^4 [p_i p_j' - p_i' p_j],
so that
    U_2(ell) = (G_ell/sqrt(pi))^4 Pf M^(ell-1)[w_1] / Pf M^(ell)[w_0],   C_2 = A_sigma^4 U_2^2 / ell.
The linear neutral coefficient is A_1 = 4 dbar^2 / ell^2, where dbar = d/dt log Pf M(t) at t = 0
for the tilted weight w_0^4 exp[t(u - ell/2)], eq. (A.14); it is evaluated here as a symmetric
difference with step 1e-30.

The moments use the centered, rescaled monomials q_i(u) = ((u-c)/s)^i with c = s = (ell-1/2)/2,
which changes each Pfaffian by the known factor s^(-N(2N-1)).  Raw monomials u^i lose many digits
from ell ~ 32 on.  Each size is computed at 80 + 2 ell decimal digits; --check-precision repeats it
with 40 more digits and reports how many leading digits agree.  --verify compares the Pfaffians
with direct subset sums for ell <= 5, and the coefficients with the enumerated values in
results/records.json through ell = 16.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent


def pfaffian(A):
    """Pfaffian of a real skew-symmetric matrix (list of lists), by pivoted skew elimination."""
    A = [row[:] for row in A]; n = len(A); value = mp.mpf(1)
    for k in range(0, n - 1, 2):
        p = max(range(k + 1, n), key=lambda j: abs(A[k][j]))
        if p != k + 1:                       # swap rows and columns k+1 and p
            A[k + 1], A[p] = A[p], A[k + 1]
            for row in A:
                row[k + 1], row[p] = row[p], row[k + 1]
            value = -value
        pivot = A[k][k + 1]
        if not pivot:
            return mp.mpf(0)
        value *= pivot
        rk = [x / pivot for x in A[k]]; rk1 = [x / pivot for x in A[k + 1]]
        for i in range(k + 2, n):            # Schur complement of the leading 2x2 block
            Ai = A[i]; aik = Ai[k]; aik1 = Ai[k + 1]
            for j in range(i + 1, n):
                Ai[j] += aik * rk1[j] - aik1 * rk[j]
                A[j][i] = -Ai[j]
    return value


def nodes(ell):
    """Lambda_ell = X_ell u Y_ell, the integers 0..ell-1 and half-integers 1/2..ell-1/2."""
    return [mp.mpf(i) for i in range(ell)] + [mp.mpf(j) + mp.mpf(1) / 2 for j in range(ell)]


def weight4(ell, charged):
    """w_0(u)^4 or w_1(u)^4 of eq. (5.4)."""
    half = mp.mpf(1) / 2
    if charged:
        return lambda u: (mp.rgamma(u + half) * mp.rgamma(ell - u)) ** 4
    return lambda u: (mp.rgamma(u + half) * mp.rgamma(ell - u + half)) ** 4


def skew_moments(ell, w4, N, tilt=0):
    """M^(N) of eq. (5.6) in the centered basis, with the weight multiplied by exp[tilt (u - ell/2)]."""
    n = 2 * N; c = s = (mp.mpf(ell) - mp.mpf(1) / 2) / 2
    M = [[mp.mpf(0)] * n for _ in range(n)]
    for u in nodes(ell):
        W = w4(u) * (mp.exp(tilt * (u - mp.mpf(ell) / 2)) if tilt else 1)
        x = (u - c) / s
        pw = [mp.mpf(1)]
        for _ in range(n - 1):
            pw.append(pw[-1] * x)
        dp = [mp.mpf(0)] + [i * pw[i - 1] / s for i in range(1, n)]
        for i in range(n):
            Wp = W * pw[i]; Wd = W * dp[i]; Mi = M[i]
            for j in range(i + 1, n):
                value = Wp * dp[j] - Wd * pw[j]
                Mi[j] += value; M[j][i] -= value
    return M, s


def pf_ensemble(ell, charged, tilt=0):
    """Z_N^(2)[w] for N = ell (neutral, w_0) or N = ell - 1 (charged, w_1), eq. (5.6)."""
    N = ell - 1 if charged else ell
    if N == 0:
        return mp.mpf(1)
    M, s = skew_moments(ell, weight4(ell, charged), N, tilt)
    return pfaffian(M) * s ** (N * (2 * N - 1))


def coefficients(ell, extra_digits=0, step=mp.mpf("1e-30")):
    mp.mp.dps = 80 + 2 * ell + extra_digits
    a_sigma = mp.power(2, mp.mpf(1) / 12) * mp.exp(3 * mp.zeta(-1, derivative=1))
    g_ell = mp.fprod(mp.gamma(n - mp.mpf(1) / 2) / mp.gamma(n) for n in range(1, ell + 1))
    neutral = pf_ensemble(ell, charged=False)
    u2 = (g_ell / mp.sqrt(mp.pi)) ** 4 * pf_ensemble(ell, charged=True) / neutral
    plus, minus = pf_ensemble(ell, False, step), pf_ensemble(ell, False, -step)
    dbar = (mp.log(plus) - mp.log(minus)) / (2 * step)
    v_y = mp.fprod(mp.mpf(k - j) for j in range(ell) for k in range(j + 1, ell))
    return {"ell": ell, "dps": mp.mp.dps, "U2": u2, "C2": a_sigma ** 4 * u2 ** 2 / ell,
            "b": a_sigma ** 2 * u2 / mp.sqrt(ell), "lA1": 4 * dbar ** 2 / ell,
            "T2": v_y ** 4 * neutral, "dbar": dbar, "a_sigma": a_sigma}


def agreeing_digits(x, y):
    if x == y:
        return mp.mp.dps
    return int(mp.floor(-mp.log10(abs(x - y) / abs(x))))


def verify_small():
    """Pf against direct subset sums of eq. (5.3) at alpha = 2, and dbar against its subset mean."""
    print("check of Pf M = Z_N^(2)[w], eq. (5.6), and of the tilted derivative, eq. (A.14)")
    print("  ell   rel. error neutral   rel. error charged   rel. error dbar")
    for ell in range(1, 6):
        mp.mp.dps = 50
        errors = []
        for charged in (False, True):
            N = ell - 1 if charged else ell; w4 = weight4(ell, charged)
            total = mean = mp.mpf(0)
            for U in itertools.combinations(nodes(ell), N):
                term = mp.fprod([U[b] - U[a] for a in range(N) for b in range(a + 1, N)]) ** 4 * mp.fprod(w4(u) for u in U)
                total += term; mean += term * (mp.fsum(U) - mp.mpf(ell) ** 2 / 2)
            errors.append(abs(pf_ensemble(ell, charged) / total - 1))
            if not charged:
                dbar = coefficients(ell)["dbar"]; mp.mp.dps = 50
                dbar_error = abs(dbar / (mean / total) - 1)
        print(f"  {ell:>3}   {mp.nstr(errors[0], 3):>18}   {mp.nstr(errors[1], 3):>18}   {mp.nstr(dbar_error, 3):>15}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", default="4,8,12,16,32,64,128")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--check-precision", action="store_true")
    parser.add_argument("--record", default=HERE / "results" / "records.json")
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    if args.verify:
        verify_small(); print()

    rows = []
    print("  ell   digits   C_2(ell)                         ell A_1(ell)                     "
          "b_ell     2 A_s^2 U_2/sqrt(ell)")
    for ell in lengths:
        start = time.time()
        row = coefficients(ell)
        if args.check_precision:
            again = coefficients(ell, extra_digits=40)
            row["agree"] = min(agreeing_digits(row[k], again[k]) for k in ("C2", "lA1"))
        rows.append(row)
        print(f"  {ell:>3}   {row['dps']:>6}   {mp.nstr(row['C2'], 30):<32} {mp.nstr(row['lA1'], 30):<32} "
              f"{mp.nstr(row['b'], 8):<9} {mp.nstr(2 * row['b'], 8)}   ({time.time() - start:.0f} s)", flush=True)
    if args.check_precision:
        print(f"\nrepeat with 40 more digits: C_2 and ell A_1 agree to at least {min(r['agree'] for r in rows)} "
              "significant digits at every size")

    if args.verify and Path(args.record).exists():
        record = json.load(open(args.record, encoding="utf-8"))
        charged_record = {r["ell"]: r for r in record.get("charged_coefficients", [])}
        neutral = {r["ell"]: r for r in record.get("neutral", [])}
        shared = [r for r in rows if r["ell"] in charged_record]
        if shared:
            print("\ndirect enumeration (results/records.json) minus Pfaffian, eq. (5.7):")
            for r in shared:
                line = (f"  ell = {r['ell']:>2}:  C_2 {charged_record[r['ell']]['C_2'] - float(r['C2']):+.1e}   "
                        f"T_2 {charged_record[r['ell']]['T_2'] / float(r['T2']) - 1:+.1e} (relative)")
                if r["ell"] in neutral:
                    line += f"   ell A_1 {neutral[r['ell']]['ell_A_neut_2'] - float(r['lA1']):+.1e}"
                print(line)
            print(f"  max |difference| in C_2: {max(abs(charged_record[r['ell']]['C_2'] - float(r['C2'])) for r in shared):.1e}")

    ordered = sorted(rows, key=lambda r: r["ell"])
    if len(ordered) >= 3:
        print("\nfits of C_2 to c0 + c1/sqrt(ell) + c2/ell over the largest sizes:")
        for m in range(3, min(5, len(ordered)) + 1):
            x = np.array([float(r["ell"]) for r in ordered[-m:]]); y = np.array([float(r["C2"]) for r in ordered[-m:]])
            c0 = np.linalg.lstsq(np.vstack([np.ones(m), x ** -0.5, 1 / x]).T, y, rcond=None)[0][0]
            print(f"  largest {m} (ell = {', '.join(str(r['ell']) for r in ordered[-m:])}): c0 = {c0:.5f}")
        print(f"C_2({ordered[-1]['ell']}) is {100 * (1 - 4 * float(ordered[-1]['C2'])):.1f}% below 1/4")

    print("\nFigure 1 coordinates (1/sqrt(ell), value), read by make_figures.py:")
    for key in ("C2", "lA1"):
        print(f"  {key}: " + " ".join(f"({1 / np.sqrt(r['ell']):.9f},{float(r[key]):.10f})".replace("(0.", "(.").replace(",0.", ",.")
                                      for r in rows))


if __name__ == "__main__":
    main()
