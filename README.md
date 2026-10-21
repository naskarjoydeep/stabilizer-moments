# Numerical implementation: *Stabilizer Rényi moments beyond entanglement*

V. Jejjala and J. Naskar, [arXiv:2610.27596](https://arxiv.org/abs/2610.27596) [quant-ph].

This package contains the code behind every number, table and figure in the paper, and the
saved output of each calculation. Equation, table, figure and section numbers refer to
[arXiv:2610.27596v1](https://arxiv.org/abs/2610.27596v1).

The calculations are exact finite computations: closed-form Cauchy sums, Pfaffians and Majorana
determinants for the infinite critical Ising chain, and exact diagonalization of finite rings
for the clock and Blume–Capel chains. Nothing is sampled. They test the finite-support and
finite-interval statements of the paper; they do not by themselves establish the
growing-interval continuum limit.

## Contents

| File | What it computes |
|---|---|
| `ising.py` | Every Ising result except the index-two Pfaffians: Cauchy sums, Majorana determinants, the rotated two-site moment, the periodic chain, and checks of the identities in appendices A and B |
| `pfaffian.py` | The index-two Pfaffian formula for growing intervals: Figure 1 and section 5.2 |
| `clock.py` | The Fateev–Zamolodchikov clock chains, the Blume–Capel control and the replica-subgroup check |
| `make_figures.py` | Figures 1–4, written to `figures/*.pdf` when run |
| `results/` | Saved outputs: the printed output of each script (`ising.txt`, `pfaffian.txt`, `clock.txt`) and the machine-readable records (`records.json`) |

`ising.py` and `clock.py` take a subcommand; `python3 ising.py -h` lists them.

## Requirements

Python 3.9 or later with NumPy, SciPy, mpmath and Matplotlib (`requirements.txt`).
The outputs in `results/` were produced on macOS (Apple Silicon) with Python 3.9.6,
NumPy 1.24.2, SciPy 1.10.1, mpmath 1.4.1 and Matplotlib 3.9.4.

```sh
python3 -m pip install -r requirements.txt
```

Run everything from this directory. `ising.py` sets `OMP_NUM_THREADS=1` (and the
Accelerate/OpenBLAS equivalents) itself, since a multithreaded BLAS is far slower on many
small determinants.

## Where each result comes from

| Result in the paper | Command | Saved output | Time |
|---|---|---|---|
| Eq. (3.5): replica subgroup $K_\alpha$ and its annihilator | `clock.py replica` | `clock.txt` | < 1 s |
| Section 3.3 and appendix A.4: rotated Ising, eqs. (3.7), (3.8), (A.15)–(A.18); fit 0.36 over r = 32–256 and local exponent 0.255 at r = 10⁵ (α = 4) | `ising.py rotated` | `ising.txt` | < 1 s |
| Table 1: charged coefficients $\mathcal C_\alpha(\ell)$, eq. (5.2) | `ising.py charged-coefficients` | `records.json` (`charged_coefficients`) | ~7 min |
| Section 5.3: the < 6% spread of $\ell^{\alpha/2-2}\mathcal C_\alpha$, $U_\alpha(16)/16$, $T_2(12)$ ≈ 60 and $T_5(12)$ ≈ 1.4, the bound (5.16) reaching e at ℓ ≈ 92 and 8364 | `ising.py diagnostics` | `ising.txt` | < 1 s |
| Figure 1 and section 5.2: $\mathcal C_2(\ell)$ and $\ell A_1(\ell)$ for ℓ = 4–128, eqs. (5.6), (5.7), (A.12)–(A.14); $\mathcal C_2(128)$ = 0.2357979, the fits 0.24995, 0.24991, 0.24988, $\ell A_1(128)$ = 0.0275335, agreement with direct sums within 2×10⁻¹⁴, the precision check; the coupling $b_\ell$ of eq. (5.10) | `pfaffian.py --verify --check-precision` | `pfaffian.txt` | ~15 min |
| Section 5.2: $\ell A_1$ by direct enumeration, eq. (A.9): 0.0270122 (α = 2) and 0.00773 (α = 3) at ℓ = 16 | `ising.py neutral` | `records.json` (`neutral`) | hours at ℓ = 16; see below |
| Appendices A.1–A.2: identities (A.3)–(A.4), the ℓ = 2 overlaps 1, 1, 1/2, 1/2, the inverse sums (A.7), the endpoint identity (A.11) at real α, and a direct two-block check of eqs. (A.8)–(A.9) | `ising.py identities` | `ising.txt` | ~2 s |
| Appendix B: the projector (5.8), the stabilizers, eigenvalues and junction weights 1/16, 5/32, 3/16, the mirror overlaps, the slit map (B.4), eq. (B.6) and the expansion (B.7) | `ising.py junction` | `ising.txt` | < 1 s |
| Figure 2(a): length-two charged ratios | `ising.py length-two` | `ising.txt` | ~1 s |
| Figure 2(b): one-site charged, neutral and full ratios | `ising.py sectors` | `records.json` (`sectors`), `ising.txt` | < 1 s |
| Table 2 ($s_{\rm ch}$ column) and Figure 3: charged slopes, window sensitivities, intercepts | `clock.py fits` | `clock.txt` | ~40 s |
| Section 5.4: neutral/all-charged fractions 12–21%, 24–32%, 38–45% | `clock.py fractions` | `clock.txt`, `records.json` (`sector_fractions`) | ~15 s |
| Table 2 ($2\Delta_{\rm ref}$ and ratio columns for the clock chains, and the daggered L = 8 → 10 changes) and section 5.4: size sequences to L = 14, 12, 10 and the same-ring ratios | `clock.py finite-size` | `clock.txt` | about 1 h, dominated by ℤ₄ at L = 12; see below |
| Section 5.4: Blume–Capel control (λ = 1.326, L = 12): forbidden weight 0.0597, powers 0.783, 1.067, 1.130 | `clock.py blume-capel` | `clock.txt`, `records.json` (`blume_capel`) | ~2 min |
| Section 5.5 and Figure 4: single-interval slopes, intercepts 0.1267/0.1264, difference intercepts −3.0×10⁻⁴/−6.1×10⁻⁵ | `ising.py single-interval` | `ising.txt` | ~2 s |
| Section 5.5: full-system constants, eq. (4.16), agreement within 0.7–2.9% | `ising.py full-system` | `ising.txt` | ~3 s |
| Section 6: $L_{\rm LR}$ = 0.0819, 0.1088, 0.1223, 0.1307 at η = 1/4, eq. (6.1) | `ising.py fixed-cross-ratio` | `records.json` (`fixed_cross_ratio`) | ~3 s |
| Figures 1–4 | `make_figures.py` | `figures/*.pdf` | ~5 s |

The file `fig0_growing.pdf` is Figure 1, and `fig1_powerlaw.pdf`, `fig2_exponent.pdf` and
`fig3_singleinterval.pdf` are Figures 2, 3 and 4 of the paper.

### Reproducing everything

```sh
python3 ising.py all > results/ising.txt
python3 ising.py charged-coefficients --output results/records.json
python3 ising.py neutral --output results/records.json
python3 ising.py fixed-cross-ratio --output results/records.json
python3 pfaffian.py --verify --check-precision > results/pfaffian.txt
python3 clock.py all --output results/records.json > results/clock.txt
python3 clock.py finite-size >> results/clock.txt
python3 make_figures.py
```

Run `make_figures.py` after `pfaffian.py`, since Figure 1 reads `results/pfaffian.txt`.

`ising.py all` runs `identities`, `diagnostics` (from the Table 1 record), `rotated`,
`length-two`, `sectors`, `single-interval`, `full-system` and `junction`; `clock.py all` runs
`replica`, `fits`, `fractions` and `blume-capel`. Records passed with `--output` are merged
into `results/records.json` by size, so a partial run updates only the sizes it computes. For a quick
check, `charged-coefficients`, `neutral` and `fixed-cross-ratio` take `--lengths` and `--alphas`,
`clock.py finite-size` takes `--sizes`, and `pfaffian.py` takes `--lengths`, for example
`python3 ising.py neutral --lengths 1,2,3,4 --output /tmp/check.json`,
`python3 clock.py finite-size --sizes "3:8,10,12;4:8,10;5:8"` (about 15 s) or
`python3 pfaffian.py --lengths 4,8,16 --verify`.

## Conventions

* **Weyl operators and charge.** For a $d$-state site, $X|j\rangle=|j+1\rangle$,
  $Z|j\rangle=\omega^j|j\rangle$, $W_{\bm u}=\prod_j X_j^{a_j}Z_j^{b_j}$, and the charge under
  $\prod_j X_j$ is $q=\sum_j b_j \bmod d$. The weights are
  $\Xi(\bm u)=|{\rm Tr}\,\rho W_{\bm u}|^2/D$, and every ratio is normalized by the product of
  single-region moments.
* **Ising.** The free-fermion calculations work in the $d=2$ clock convention
  $H=-\sum(Z_jZ_{j+1}+X_j)$, with Majorana covariance
  $\langle i a_m b_n\rangle=-2/[\pi(2(n-m)+1)]$. The paper writes Ising as
  $H=-\sum(X_jX_{j+1}+Z_j)$, its local-Hadamard image, which leaves every moment unchanged.
  `ising.py rotated` uses the paper's convention directly. Strings keep their Jordan–Wigner
  tails across gaps; dropping them loses the charged sector.
* **Pfaffians.** `pfaffian.py` builds the skew moments (5.6) from centered, rescaled monomials
  $((u-c)/s)^i$ with $c=s=(\ell-1/2)/2$ and corrects each Pfaffian by $s^{N(2N-1)}$; raw
  monomials $u^i$ lose many digits from ℓ ≈ 32 on. Each size is computed at $80+2\ell$ decimal
  digits. The neutral derivative in (A.14) is taken as a symmetric difference of
  $\log\operatorname{Pf}$ with step $10^{-30}$.
* **Cross-ratios.** Intervals of length ℓ starting at 0 and at $r$ on the infinite chain have
  $\eta=(\ell/r)^2$. On a ring of $L$ sites, single sites at separation $r$ have
  $\eta=[l_c(1)/l_c(r)]^2$ with chord length $l_c(x)=(L/\pi)\sin(\pi x/L)$.
* **Fits.** Table 2 uses the full separation window as the primary fit. The parenthesized
  sensitivities are the largest shifts when the one or two shortest separations are removed,
  keeping at least three points; they are not statistical errors. Figure 3's Ising points are
  fits to the six smallest cross-ratios (η ≤ 1/576), as printed by `ising.py length-two`.
  In Table 2, $2\Delta_{\rm ref}$ is the exact value for Ising and, for the clock chains, the
  order-field dimension fitted on the same ring (`ring dim.` in the finite-size table).

## Provenance of the saved outputs

Everything in `results/` was regenerated with this package on 17 October 2026, with two
exceptions that are too slow to rerun routinely:

* the ℓ = 13–16 rows of the `neutral` record in `results/records.json` (about 6×10⁸ small
  determinants at ℓ = 16), and
* the Potts L = 14, ℤ₄ L = 12 and ℤ₅ L = 10 rows of the finite-size table in
  `results/clock.txt` (the ℤ₄ ring at L = 12 has 1.7×10⁷ states).

These are the authors' earlier runs of the same calculations, marked `[earlier run]` in the
finite-size table. At the smaller sizes the package reproduces the finite-size rows exactly and
the neutral rows to 2×10⁻¹⁵ relative, and at α = 2 `pfaffian.py --verify` checks the ℓ = 16
neutral row independently. Regenerated eigensolver outputs agree with earlier runs except in
quantities that vanish to numerical precision (charge leakage ~10⁻²², the Blume–Capel
parity-odd linear term ~10⁻¹⁹) and at the 10⁻⁹ level set by the Lanczos tolerance.

`make_figures.py` draws Figure 1 from the coordinates printed in `results/pfaffian.txt`, and the
other figures from rounded arrays copied out of these outputs: Figure 2(a) and the
Ising points of Figure 3 from the `length-two` section of `results/ising.txt`, Figure 2(b) from
the `sectors` record in `results/records.json`, the clock points of Figure 3 from the `fits`
section of `results/clock.txt`, the Blume–Capel points from the `blume_capel` record, and Figure 4 from the
`single-interval` section of `results/ising.txt`.

## How to cite

If you use this code or its results, please cite the paper:

> V. Jejjala and J. Naskar, *Stabilizer Rényi moments beyond entanglement*,
> [arXiv:2610.27596](https://arxiv.org/abs/2610.27596) [quant-ph].

```bibtex
@article{Jejjala:2026lgm,
    author = "Jejjala, Vishnu and Naskar, Joydeep",
    title = "{Stabilizer R{\'e}nyi moments beyond entanglement}",
    eprint = "2610.27596",
    archivePrefix = "arXiv",
    primaryClass = "quant-ph",
    month = "10",
    year = "2026"
}
```
