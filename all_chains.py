#!/usr/bin/env python3
"""
================================================================
all_chains.py  -  W(n,k) for every n = 4m and every k = 1..n
================================================================

For each order n (multiple of 4) and each weight k = 1..n, the program tries
the following methods IN THIS ORDER and stores the first one that yields a
matrix passing the exact check:

  1. trivial   k = 1 (identity), k = 2 (direct sum of H2 blocks)
  2. dsum      direct sum  W(a,k) (+) W(n-a,k)               (stored solutions)
  3. kron      Kronecker   W(a,k1) x W(b,k2) = W(ab,k1*k2)   (stored solutions
               plus the atoms W(1,1), W(2,1), W(2,2))
  4. paley     Paley I : W(q+1,q) skew conference and W(q+1,q+1) = I+S,
                         q prime power, q = 3 (mod 4)
               Paley II: W(2(q+1),2(q+1)) Hadamard, q prime power, q = 1 (mod 4)
  5. gs        Goethals-Seidel, four ternary circulant generators
               - m <= --exhaustive-max-m: complete search, so a negative answer
                 is a PROOF that no GS solution exists (status none_gs)
               - otherwise: simulated annealing on the exact PAF cost

Why the PAF cost is exact
-------------------------
For circulant A,B,C,D the off-diagonal blocks of G G^T vanish identically and
    G G^T = (AA^T + BB^T + CC^T + DD^T) (x) I_4 ,
so ||G G^T - kI||_1 = 4m * sum_s |P(s) - k delta_s| with P the sum of the four
periodic autocorrelations. Cost zero <=> G G^T = kI. The cost is updated
incrementally in O(m) per changed coordinate: no n x n matrix and no O(n^3)
product inside the search.

Necessary conditions used to restrict the annealing (all follow from P = k delta):
    sum w_i = k,  sum s_i^2 = k,  w_i = s_i (mod 2),  |s_i| <= w_i <= m
(w_i = non-zeros of sequence i, s_i = its sum). Negating/permuting sequences
preserves P, so WLOG s_0>=s_1>=s_2>=s_3>=0. Moves (swap two different entries
of one sequence) preserve every (w_i, s_i): the walk stays in the right slice.
If no admissible profile exists the status is none_gs (proved for GS).

--seed-prev (OFF by default) implements the W(n,k-2) -> W(n,k) idea: warm start
from the stored GS generators of W(n,k-2), adapted to the closest admissible
profile. Run with and without it and compare the 'restarts' and 'seconds'
columns of all_chains_status.csv.

Every accepted matrix is rebuilt and checked exactly (shape, entries, row and
column weights, W W^T = kI, W^T W = kI). Products use float64 BLAS on integer
data; all partial sums are integers far below 2^53, hence exact (no tolerance).

Output (compatible with verify_weighing.py and query_weighing_matrices.py):
    solutions/W{n}_{k}/solution_001.json
    solutions/all_chains_status.csv
Non-GS solutions carry empty A,B,C,D/weights/sums plus "method" and "recipe";
verify_weighing.py reports their generator checks only as consistency WARNINGS
(the matrix verdict is unaffected).

Usage:
    python all_chains.py --nmax 48
    python all_chains.py --nmin 28 --nmax 60 --time-per-k 120 --seed-prev
    python all_chains.py --nmax 12 --disable dsum,kron,paley     # GS only
Existing solutions in --out are respected (resume); failed pairs are not
retried unless --retry-failed is given.
================================================================
"""
__author__ = "Adrian Macias-Quintero"
__version__ = "1.1.0"
__license__ = "Apache-2.0"
__copyright__ = "Copyright (c) 2025 Adrian Macias-Quintero"

import argparse
import csv
import itertools
import json
import math
import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

ATOMS = {(1, 1), (2, 1), (2, 2)}


# ---------------------------------------------------------------- matrices
def circulant(x):
    m = len(x)
    idx = (np.arange(m)[None, :] - np.arange(m)[:, None]) % m
    return np.asarray(x, dtype=np.int8)[idx]


def gs_matrix(A, B, C, D):
    """Goethals-Seidel array (same convention as chain_odd/chain_even/verify)."""
    m = len(A)
    a, b, c, d = (circulant(v) for v in (A, B, C, D))
    R = np.fliplr(np.eye(m, dtype=np.int8))
    return np.block([
        [a,       b @ R,    c @ R,     d @ R],
        [-b @ R,  a,        d.T @ R,  -c.T @ R],
        [-c @ R, -d.T @ R,  a,         b.T @ R],
        [-d @ R,  c.T @ R, -b.T @ R,   a],
    ]).astype(np.int8)


def exact_check(W, k):
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        return False
    n = W.shape[0]
    if np.abs(W.astype(np.int16)).max() > 1:
        return False
    nz = W != 0
    if not (nz.sum(axis=1) == k).all() or not (nz.sum(axis=0) == k).all():
        return False
    Wf = W.astype(np.float64)
    target = k * np.eye(n)
    return bool(np.array_equal(Wf @ Wf.T, target) and np.array_equal(Wf.T @ Wf, target))


# ---------------------------------------------------------------- Paley
def prime_power(q):
    if q < 2:
        return None
    p = next((d for d in range(2, math.isqrt(q) + 1) if q % d == 0), q)
    e, t = 0, q
    while t % p == 0:
        t //= p
        e += 1
    return (p, e) if t == 1 else None


@lru_cache(maxsize=8)
def quad_char_matrix(q):
    """Q[i,j] = chi(j-i) over GF(q) (int8, q x q)."""
    p, e = prime_power(q)
    if e == 1:
        chi = np.full(q, -1, dtype=np.int8)
        chi[0] = 0
        chi[(np.arange(1, q, dtype=np.int64) ** 2) % q] = 1
        idx = (np.arange(q)[None, :] - np.arange(q)[:, None]) % q
        return chi[idx]
    pw = [p ** t for t in range(e)]
    digits = np.array([[(v // pw[t]) % p for t in range(e)] for v in range(q)], dtype=np.int64)
    logs = None
    for coeffs in itertools.product(range(p), repeat=e):       # f = x^e + sum c_t x^t
        if coeffs[0] == 0:
            continue
        cur = [1] + [0] * (e - 1)
        lg = np.full(q, -1, dtype=np.int64)
        ok = True
        for t in range(q - 1):                                 # powers of x
            v = sum(cur[j] * pw[j] for j in range(e))
            if lg[v] != -1:
                ok = False
                break
            lg[v] = t
            top = cur[-1]
            cur = [0] + cur[:-1]
            if top:
                cur = [(cur[j] - top * coeffs[j]) % p for j in range(e)]
        if ok and cur == [1] + [0] * (e - 1):                  # x primitive
            logs = lg
            break
    chi = np.zeros(q, dtype=np.int8)
    nzm = logs >= 0
    chi[nzm] = np.where(logs[nzm] % 2 == 0, 1, -1)
    diff = (((digits[None, :, :] - digits[:, None, :]) % p) @ np.array(pw)).astype(np.int64)
    return chi[diff]


def paley_skew(q):                       # q = 3 mod 4 -> W(q+1, q), skew
    S = np.zeros((q + 1, q + 1), dtype=np.int8)
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = quad_char_matrix(q)
    return S


def paley_sym(q):                        # q = 1 mod 4 -> symmetric conference matrix
    C = np.zeros((q + 1, q + 1), dtype=np.int8)
    C[0, 1:] = 1
    C[1:, 0] = 1
    C[1:, 1:] = quad_char_matrix(q)
    return C


# ---------------------------------------------------------------- store / recipes
class Store:
    def __init__(self, base):
        self.base = Path(base)
        self.recipes = {}
        self.solved = set()
        for d in self.base.glob("W*_*"):
            try:
                n, k = map(int, d.name[1:].split("_"))
            except ValueError:
                continue
            if (d / "solution_001.json").exists():
                self.solved.add((n, k))

    def path(self, n, k):
        return self.base / f"W{n}_{k}" / "solution_001.json"

    def available(self, n, k):
        return (n, k) in ATOMS or (n, k) in self.solved

    def recipe(self, n, k):
        if (n, k) in self.recipes:
            return self.recipes[(n, k)]
        if (n, k) in ATOMS:
            r = {"type": "atom"}
        else:
            with open(self.path(n, k), "r", encoding="utf-8") as f:
                d = json.load(f)
            if "recipe" in d:
                r = d["recipe"]
            elif d.get("A"):
                r = {"type": "gs", "A": d["A"], "B": d["B"], "C": d["C"], "D": d["D"]}
            else:
                r = {"type": "file", "path": str(self.path(n, k))}
        self.recipes[(n, k)] = r
        return r

    def matrix(self, n, k):
        return self.build(self.recipe(n, k), n, k)

    def build(self, r, n, k):
        t = r["type"]
        if t == "atom":
            if n == 1:
                return np.array([[1]], np.int8)
            return np.eye(2, dtype=np.int8) if k == 1 else np.array([[1, 1], [1, -1]], np.int8)
        if t == "identity":
            return np.eye(n, dtype=np.int8)
        if t == "h2blocks":
            return np.kron(np.eye(n // 2, dtype=np.int8), np.array([[1, 1], [1, -1]], np.int8))
        if t == "gs":
            return gs_matrix(*[np.array(r[c], dtype=np.int8) for c in "ABCD"])
        if t == "dsum":
            (a, ka), (b, kb) = r["parts"]
            W = np.zeros((n, n), dtype=np.int8)
            W[:a, :a] = self.matrix(a, ka)
            W[a:, a:] = self.matrix(b, kb)
            return W
        if t == "kron":
            (a, k1), (b, k2) = r["left"], r["right"]
            return np.kron(self.matrix(a, k1), self.matrix(b, k2)).astype(np.int8)
        if t == "paley1":
            return paley_skew(r["q"])
        if t == "paley1_h":
            return (np.eye(r["q"] + 1, dtype=np.int8) + paley_skew(r["q"])).astype(np.int8)
        if t == "paley2":
            C = paley_sym(r["q"])
            I = np.eye(C.shape[0], dtype=np.int8)
            return np.block([[C + I, C - I], [C - I, -C - I]]).astype(np.int8)
        if t == "file":
            with open(r["path"], "r", encoding="utf-8") as f:
                return np.array(json.load(f)["matrix"], dtype=np.int8)
        raise ValueError(f"unknown recipe type {t}")

    def save(self, n, k, recipe, W):
        d = {"n": n, "k": k, "m": n // 4, "A": [], "B": [], "C": [], "D": [],
             "weights": [], "sums": []}
        if recipe["type"] == "gs":
            seqs = [recipe[c] for c in "ABCD"]
            d.update(A=seqs[0], B=seqs[1], C=seqs[2], D=seqs[3],
                     weights=[int(np.count_nonzero(s)) for s in seqs],
                     sums=[int(sum(s)) for s in seqs])
        chars = np.array(["-", ".", "+"])
        Wi = W.astype(int)
        d["matrix"] = Wi.tolist()
        d["matrix_symbols"] = ["".join(row) for row in chars[Wi + 1]]
        d["method"] = recipe["type"]
        d["recipe"] = recipe
        p = self.path(n, k)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, separators=(",", ":"))
        self.solved.add((n, k))
        self.recipes[(n, k)] = recipe


# ---------------------------------------------------------------- cheap methods
def try_trivial(st, n, k):
    if k == 1:
        return {"type": "identity"}
    if k == 2 and n % 2 == 0:
        return {"type": "h2blocks"}
    return None


def try_dsum(st, n, k):
    a = (n // 2) // 4 * 4                      # most balanced split first
    while a >= 4:
        b = n - a
        if k <= a and st.available(a, k) and st.available(b, k):
            return {"type": "dsum", "parts": [[a, k], [b, k]]}
        a -= 4
    return None


def divisors(x):
    return [d for d in range(1, x + 1) if x % d == 0]


def try_kron(st, n, k):
    for a in divisors(n):
        b = n // a
        if a == 1 or b == 1 or a > b:
            continue
        for k1 in divisors(k):
            k2 = k // k1
            if k1 <= a and k2 <= b and st.available(a, k1) and st.available(b, k2):
                return {"type": "kron", "left": [a, k1], "right": [b, k2]}
    return None


def try_paley(st, n, k):
    q = n - 1
    if prime_power(q) and q % 4 == 3:
        if k == n - 1:
            return {"type": "paley1", "q": q}
        if k == n:
            return {"type": "paley1_h", "q": q}
    if k == n and n % 2 == 0:
        q = n // 2 - 1
        if q >= 3 and prime_power(q) and q % 4 == 1:
            return {"type": "paley2", "q": q}
    return None


# ---------------------------------------------------------------- GS: exhaustive
@lru_cache(maxsize=4)
def _pair_table(m):
    vals = np.array(list(itertools.product((-1, 0, 1), repeat=m)), dtype=np.int64)
    P = np.stack([(vals * np.roll(vals, -s, axis=1)).sum(axis=1) for s in range(m)], axis=1)
    ia, ib = np.triu_indices(len(vals))
    PS = P[ia] + P[ib]
    mult = (4 * m + 1) ** np.arange(m, dtype=np.int64)
    keys = ((PS + 2 * m) * mult).sum(axis=1)
    order = np.argsort(keys, kind="stable")
    return vals, ia[order], ib[order], PS[order], keys[order], mult


def exhaustive_gs(m, k):
    """Complete search -> (A,B,C,D) lists, or None (PROOF: no GS solution)."""
    vals, ia, ib, PS, keys, mult = _pair_table(m)
    T = np.zeros(m, dtype=np.int64)
    T[0] = k
    need = T[None, :] - PS
    valid = (np.abs(need) <= 2 * m).all(axis=1)
    if not valid.any():
        return None
    vidx = np.flatnonzero(valid)
    nk = ((need[vidx] + 2 * m) * mult).sum(axis=1)
    pos = np.minimum(np.searchsorted(keys, nk), len(keys) - 1)
    hit = np.flatnonzero(keys[pos] == nk)
    if hit.size == 0:
        return None
    f, s = vidx[hit[0]], pos[hit[0]]
    return tuple(vals[i].tolist() for i in (ia[f], ib[f], ia[s], ib[s]))


# ---------------------------------------------------------------- GS: annealing
@lru_cache(maxsize=None)
def sum_tuples(k, m):
    """Sorted (s0>=s1>=s2>=s3>=0), sum s^2 = k, admitting weights w_i <= m with sum w_i = k."""
    out = []
    for s0 in range(min(math.isqrt(k), m), -1, -1):
        r0 = k - s0 * s0
        for s1 in range(min(s0, math.isqrt(r0), m), -1, -1):
            r1 = r0 - s1 * s1
            for s2 in range(min(s1, math.isqrt(r1), m), -1, -1):
                r2 = r1 - s2 * s2
                s3 = math.isqrt(r2)
                if s3 * s3 != r2 or s3 > s2 or s3 > m:
                    continue
                s = (s0, s1, s2, s3)
                extra = k - sum(s)
                if extra < 0 or extra % 2:
                    continue
                if sum((m - si) // 2 for si in s) * 2 >= extra:
                    out.append(s)
    return tuple(out)


def sample_profile(k, m, rng):
    tups = sum_tuples(k, m)
    if not tups:
        return None
    s = rng.choice(tups)
    caps = [(m - si) // 2 for si in s]
    r = (k - sum(s)) // 2
    j = [0] * 4
    avail = [i for i in range(4) if caps[i] > 0]
    while r > 0:
        i = rng.choice(avail)
        j[i] += 1
        r -= 1
        if j[i] == caps[i]:
            avail.remove(i)
    prof = [(s[i] + 2 * j[i], s[i]) for i in range(4)]
    rng.shuffle(prof)
    return prof


def closest_arrangement(prof, prev_ws):
    best = None
    for perm in itertools.permutations(range(4)):
        arr, d = [], 0
        for i in range(4):
            w, s = prof[perm[i]]
            s = s if prev_ws[i][1] >= 0 else -s
            arr.append((w, s))
            d += abs(w - prev_ws[i][0]) + abs(s - prev_ws[i][1])
        if best is None or d < best[0]:
            best = (d, arr)
    return best


def random_state(m, prof, rng):
    X = np.zeros((4, m), dtype=np.int64)
    for i, (w, s) in enumerate(prof):
        pos = rng.sample(range(m), w)
        nplus = (w + s) // 2
        vals = [1] * nplus + [-1] * (w - nplus)
        rng.shuffle(vals)
        X[i, pos] = vals
    return X


def warm_start(prev, targets, rng):
    X = prev.copy()
    for i, (w, s) in enumerate(targets):
        x = X[i]
        cur = int(np.count_nonzero(x))
        while cur < w:
            p = int(rng.choice(np.flatnonzero(x == 0)))
            x[p] = rng.choice((-1, 1))
            cur += 1
        while cur > w:
            p = int(rng.choice(np.flatnonzero(x)))
            x[p] = 0
            cur -= 1
        sm = int(x.sum())
        while sm != s:
            if sm < s:
                x[int(rng.choice(np.flatnonzero(x == -1)))] = 1
                sm += 2
            else:
                x[int(rng.choice(np.flatnonzero(x == 1)))] = -1
                sm -= 2
    return X


def paf_total(X, m):
    F = np.fft.rfft(X, axis=1)
    return np.rint(np.fft.irfft((F * np.conj(F)).sum(axis=0), n=m)).astype(np.int64)


def anneal(X, m, iters, T0, T1, rng, deadline):
    """Anneal inside the (w_i, s_i) slice. Returns the solution state or None."""
    P = paf_total(X, m)
    E = int(np.abs(P[1:]).sum())
    if E == 0:
        return X
    movable = [i for i in range(4) if len(np.unique(X[i])) > 1]
    if not movable:
        return None
    ar = np.arange(m)
    PL = (ar[:, None] + ar[None, :]) % m
    MI = (ar[:, None] - ar[None, :]) % m
    alpha = (T1 / T0) ** (1.0 / iters)
    T = T0

    def apply(i, p, v):
        x = X[i]
        d = v - int(x[p])
        if d:
            P[:] += d * (x[PL[p]] + x[MI[p]])
            P[0] += d * d
            x[p] = v

    for it in range(iters):
        if (it & 255) == 0 and time.time() > deadline:
            return None
        Pold = P.copy()
        undo = []
        for _ in range(1 if rng.random() < 0.8 else 2):
            i = rng.choice(movable)
            x = X[i]
            for _t in range(8):
                p, q = rng.randrange(m), rng.randrange(m)
                if x[p] != x[q]:
                    break
            else:
                continue
            vp, vq = int(x[p]), int(x[q])
            apply(i, p, vq)
            apply(i, q, vp)
            undo.append((i, p, vp, q, vq))
        if not undo:
            continue
        En = int(np.abs(P[1:]).sum())
        dE = En - E
        if dE <= 0 or rng.random() < math.exp(-dE / T):
            E = En
            if E == 0:
                return X.copy()
        else:
            for (i, p, vp, q, vq) in reversed(undo):
                X[i, p] = vp
                X[i, q] = vq
            P[:] = Pold
        T *= alpha
    return None


def search_gs(st, n, k, opt, rng):
    """Returns (status, recipe|None, info)."""
    m = n // 4
    if m <= opt.exhaustive_max_m:
        res = exhaustive_gs(m, k)
        if res is None:
            return "none_gs", None, {}
        return "found", {"type": "gs", **{c: res[i] for i, c in enumerate("ABCD")}}, {}
    if not sum_tuples(k, m):
        return "none_gs", None, {}
    prev = None
    if opt.seed_prev and (n, k - 2) in st.solved:
        r = st.recipe(n, k - 2)
        if r["type"] == "gs":
            prev = np.array([r[c] for c in "ABCD"], dtype=np.int64)
    iters = opt.iters or max(20000, 300 * m)
    deadline = time.time() + opt.time_per_k
    restarts = 0
    while time.time() < deadline and restarts < opt.max_restarts:
        restarts += 1
        seeded = prev is not None and restarts % 2 == 1
        if seeded:
            prev_ws = [(int(np.count_nonzero(prev[i])), int(prev[i].sum())) for i in range(4)]
            best = min((closest_arrangement(sample_profile(k, m, rng), prev_ws) for _ in range(60)),
                       key=lambda t: t[0])
            X = warm_start(prev, best[1], rng)
        else:
            X = random_state(m, sample_profile(k, m, rng), rng)
        sol = anneal(X, m, iters, opt.T0, opt.T1, rng, deadline)
        if sol is not None:
            return "found", {"type": "gs", **{c: sol[i].tolist() for i, c in enumerate("ABCD")}}, \
                {"restarts": restarts, "seeded": seeded}
    return "notfound", None, {"restarts": restarts}


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description="W(n,k) for every n=4m and k=1..n (cascade of methods).")
    ap.add_argument("--out", default="solutions")
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--nmax", type=int, default=108)
    ap.add_argument("--time-per-k", type=float, default=6000000.0, help="annealing seconds per (n,k)")
    ap.add_argument("--iters", type=int, default=0, help="iterations per restart (0 = auto)")
    ap.add_argument("--max-restarts", type=int, default=10 ** 9)
    ap.add_argument("--T0", type=float, default=3.0)
    ap.add_argument("--T1", type=float, default=0.25)
    ap.add_argument("--exhaustive-max-m", type=int, default=6,
                    help="complete GS search for m <= this (m=6 -> n<=24)")
    ap.add_argument("--seed-prev", action="store_true", help="warm start from W(n,k-2) when possible")
    ap.add_argument("--disable", default="", help="comma list among: trivial,dsum,kron,paley")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.nmin % 4 or args.nmax % 4 or args.nmin < 4:
        ap.error("--nmin and --nmax must be positive multiples of 4")

    disabled = {s.strip() for s in args.disable.split(",") if s.strip()}
    rng = random.Random(args.seed)
    st = Store(args.out)
    st.base.mkdir(parents=True, exist_ok=True)
    status_path = st.base / "all_chains_status.csv"
    failed, new_file = set(), not status_path.exists()
    if not new_file and not args.retry_failed:
        with open(status_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] in ("none_gs", "notfound"):
                    failed.add((int(row["n"]), int(row["k"])))
    log = open(status_path, "a", newline="")
    writer = csv.writer(log)
    if new_file:
        writer.writerow(["n", "k", "status", "method", "seconds", "restarts", "seeded"])

    cheap = [("trivial", try_trivial), ("dsum", try_dsum), ("kron", try_kron), ("paley", try_paley)]
    for n in range(args.nmin, args.nmax + 1, 4):
        counts, pending = {}, []
        for k in range(1, n + 1):
            if (n, k) in st.solved:
                continue
            if (n, k) in failed:
                pending.append(k)
                continue
            t0 = time.time()
            recipe, status, info, W = None, "found", {}, None
            for name, fn in cheap:
                if name in disabled:
                    continue
                r = fn(st, n, k)
                if r is None:
                    continue
                W = st.build(r, n, k)
                if exact_check(W, k):
                    recipe = r
                    break
                print(f"  ! {name} failed the exact check for W({n},{k}); ignored")
            if recipe is None:
                status, recipe, info = search_gs(st, n, k, args, rng)
                if recipe is not None:
                    W = st.build(recipe, n, k)
                    if not exact_check(W, k):              # should never happen
                        print(f"  ! GS candidate for W({n},{k}) failed the exact check; discarded")
                        status, recipe = "notfound", None
            dt = time.time() - t0
            if recipe is not None:
                st.save(n, k, recipe, W)
                counts[recipe["type"]] = counts.get(recipe["type"], 0) + 1
                print(f"W({n},{k}): {recipe['type']} [{dt:.2f}s]")
                writer.writerow([n, k, "found", recipe["type"], f"{dt:.3f}",
                                 info.get("restarts", ""), info.get("seeded", "")])
            else:
                pending.append(k)
                print(f"W({n},{k}): {status} [{dt:.2f}s]")
                writer.writerow([n, k, status, "", f"{dt:.3f}", info.get("restarts", ""), ""])
            log.flush()
        total = sum(1 for (a, _b) in st.solved if a == n)
        print(f"== n={n}: {total}/{n} stored | new by method: {counts} | unresolved: {pending}")
    log.close()


if __name__ == "__main__":
    main()
