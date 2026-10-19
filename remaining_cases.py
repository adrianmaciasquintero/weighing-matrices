#!/usr/bin/env python3
"""
remaining_cases.py - W(n, n-1) via skew Hadamard (Paley I + doubling).

Produces JSONs with the SAME schema as all_chains.py:
    {n, k, m, A, B, C, D, weights, sums, matrix, matrix_symbols, method, recipe}

The recipe is {"type": "skew_hadamard", "n": n, "base": base, "q": q}
where `base` is the first order in the doubling chain {n, n/2, n/4, ...}
such that base - 1 is a prime power ≡ 3 (mod 4), and q = base - 1.

Console output matches all_chains.py:
    W(n,k): method [seconds]
and a final summary line.
"""

__author__ = "Adrian Macias-Quintero"
__version__ = "1.1.0"
__license__ = "Apache-2.0"
__copyright__ = "Copyright (c) 2025 Adrian Macias-Quintero"

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

DEFAULT_NS = (16, 40, 56, 64, 88, 96)


# ------------------------------------------------------------------ Paley / GF
def prime_power(q):
    if q < 2:
        return None
    p = next((d for d in range(2, q + 1) if q % d == 0), None)
    e, r = 0, q
    while r % p == 0:
        r //= p
        e += 1
    return (p, e) if r == 1 else None


def paley_character(q):
    p, e = prime_power(q)
    if e == 1:
        sub = (np.arange(q)[:, None] - np.arange(q)[None, :]) % q
        squares = {x * x % q for x in range(1, q)}
    else:
        def digits(a):
            return [(a // p ** i) % p for i in range(e)]

        def enc(v):
            return sum(c * p ** i for i, c in enumerate(v))

        coeffs = None
        for cand in itertools.product(range(p), repeat=e):
            def mul(a, b):
                da, db = digits(a), digits(b)
                prod = [0] * (2 * e - 1)
                for i, x in enumerate(da):
                    for j, y in enumerate(db):
                        prod[i + j] += x * y
                for k in range(2 * e - 2, e - 1, -1):
                    c, prod[k] = prod[k], 0
                    for i in range(e):
                        prod[k - e + i] += c * cand[i]
                return enc([c % p for c in prod[:e]])

            if all(mul(a, b) for a in range(1, q) for b in range(1, q)):
                coeffs = cand
                break

        def mul(a, b):
            da, db = digits(a), digits(b)
            prod = [0] * (2 * e - 1)
            for i, x in enumerate(da):
                for j, y in enumerate(db):
                    prod[i + j] += x * y
            for k in range(2 * e - 2, e - 1, -1):
                c, prod[k] = prod[k], 0
                for i in range(e):
                    prod[k - e + i] += c * coeffs[i]
            return enc([c % p for c in prod[:e]])

        sub = np.array([
            [enc([(x - y) % p for x, y in zip(digits(a), digits(b))])
             for b in range(q)]
            for a in range(q)
        ])
        squares = {mul(a, a) for a in range(1, q)}

    chi = np.array([0] + [1 if d in squares else -1 for d in range(1, q)])
    return chi[sub]


# ------------------------------------------------------------------ skew Hadamard
def skew_hadamard(n):
    """Skew Hadamard of order n, or None if this recursion cannot build it."""
    if n == 1:
        return np.ones((1, 1), dtype=int)

    q = n - 1
    if q >= 3 and q % 4 == 3 and prime_power(q):
        Q = paley_character(q)
        S = np.zeros((n, n), dtype=int)
        S[0, 1:] = 1
        S[1:, 0] = -1
        S[1:, 1:] = Q
        return np.eye(n, dtype=int) + S

    if n % 2 == 0:
        H = skew_hadamard(n // 2)
        if H is not None:
            return np.block([[H, H], [-H.T, H.T]])
    return None


def weighing_n_minus_1(n):
    H = skew_hadamard(n)
    if H is None:
        return None
    return (H - np.eye(n, dtype=int)).astype(np.int8)


# ------------------------------------------------------------------ recipe
def get_recipe(n):
    """Locate the Paley-I base of the doubling chain, or None if none exists."""
    nn = n
    while nn > 1:
        q = nn - 1
        if q >= 3 and q % 4 == 3 and prime_power(q):
            return {"type": "skew_hadamard", "n": n, "base": nn, "q": q}
        if nn % 2 == 1:
            break
        nn //= 2
    return None


# ------------------------------------------------------------------ exact check
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


# ------------------------------------------------------------------ save (same schema)
def save_solution(base_dir, n, k, recipe, W):
    d = {"n": n, "k": k, "m": n // 4,
         "A": [], "B": [], "C": [], "D": [],
         "weights": [], "sums": []}
    chars = np.array(["-", ".", "+"])
    Wi = W.astype(int)
    d["matrix"] = Wi.tolist()
    d["matrix_symbols"] = ["".join(row) for row in chars[Wi + 1]]
    d["method"] = recipe["type"]
    d["recipe"] = recipe
    p = Path(base_dir) / f"W{n}_{k}" / "solution_001.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, separators=(",", ":"))
    return p


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser(description="W(n, n-1) via skew Hadamard.")
    ap.add_argument("--out", default="solutions")
    ap.add_argument("--n", type=int, nargs="+", default=list(DEFAULT_NS),
                    help="orders n (multiples of 4) to process")
    args = ap.parse_args()

    base_dir = Path(args.out)
    base_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    notfound = []
    for n in args.n:
        k = n - 1
        path = base_dir / f"W{n}_{k}" / "solution_001.json"

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            method = data.get("method", "?")
            counts[method] = counts.get(method, 0) + 1
            print(f"W({n},{k}): already stored as {method} [0.00s]")
            continue

        t0 = time.time()
        W = weighing_n_minus_1(n)
        if W is None:
            dt = time.time() - t0
            print(f"W({n},{k}): notfound [{dt:.2f}s]")
            notfound.append(k)
            continue

        if not exact_check(W, k):
            dt = time.time() - t0
            print(f"  ! skew candidate for W({n},{k}) failed the exact check; discarded")
            print(f"W({n},{k}): notfound [{dt:.2f}s]")
            notfound.append(k)
            continue

        recipe = get_recipe(n)
        if recipe is None:                       # should not happen if W exists
            dt = time.time() - t0
            print(f"W({n},{k}): notfound [{dt:.2f}s]")
            notfound.append(k)
            continue

        save_solution(base_dir, n, k, recipe, W)
        dt = time.time() - t0
        counts[recipe["type"]] = counts.get(recipe["type"], 0) + 1
        print(f"W({n},{k}): {recipe['type']} [{dt:.2f}s]")

    print(f"== skew-Hadamard pass: {sum(counts.values())}/{len(args.n)} stored "
          f"| new by method: {counts} | unresolved: {notfound}")


if __name__ == "__main__":
    main()