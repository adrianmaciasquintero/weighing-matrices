
# Weighing Matrices W(n,k) for n ≡ 0 (mod 4), n ≤ 108: Constructions and Exact Verification

Computational construction and exact verification of weighing matrices

W(n,k),    n = 0 mod 4,    1 <= k <= n,

for all orders

n in {4, 8, 12, ..., 108}.

This repository contains **one verified weighing matrix for every pair (n,k)** in this range, for a total of

4 + 8 + 12 + ... + 100 + 104 + 108 = 1512

cases.

---

## Repository contents

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── all_chains.py
├── remaining_cases.py
├── verify_weighing.py
├── report.csv
└── solutions/
    ├── W4_1/
    │   └── solution_001.json
    ├── W4_2/
    │   └── solution_001.json
    ├── ...
    │── W108_108/
    │   └── solution_001.json
    └── all_chains_status.csv
```

The `solutions/` directory contains one `solution_001.json` file for every pair (n,k).

Each JSON file stores:

* the order n;
* the weight k;
* the matrix W;
* a symbolic representation of the matrix;
* the construction method;
* the construction recipe;
* the Goethals–Seidel generators when applicable.

---

## What is a weighing matrix?

A weighing matrix W(n,k) is an n x n matrix with entries in

{-1, 0, +1}

such that

W * W^T = W^T * W = k * I_n.

Equivalently:

1. W is an n x n matrix.
2. Every entry belongs to {-1, 0, +1}.
3. Every row contains exactly k nonzero entries.
4. Every column contains exactly k nonzero entries.
5. The rows are mutually orthogonal.
6. The columns are mutually orthogonal.

---

## Range covered

The database covers every order

```text
n = 4, 8, 12, ..., 108

```

and every weight

```text
k = 1, 2, ..., n.

```

Therefore the total number of pairs is

Sum_{j=1..27} (4j) = 4 * (27 * 28 / 2) = 1512.

Thus, for every

n ≤ 108,    n ≡ 0 mod 4,

there is at least one stored matrix W(n,k) for every 1 ≤ k ≤ n.

---

## Construction methods

The main program `all_chains.py` searches for weighing matrices using the following methods, in this order.

### 1. Identity

For k = 1, the identity matrix is used:

W(n,1) = I_n.

The JSON method is stored as:

```text
identity

```

### 2. Hadamard 2 x 2 blocks

For k = 2, the program uses direct sums of

H_2 =
( 1   1 )
( 1  -1 )

The corresponding method is:

```text
h2blocks

```

### 3. Direct sum

If two previously stored weighing matrices are available,

W(a,k) and W(b,k),

the program can construct

W(a,k) (+) W(b,k) = W(a+b,k).

The JSON method is:

```text
dsum

```

with a recipe of the form

```json
{
  "type": "dsum",
  "parts": [[a,k],[b,k]]
}

```

### 4. Kronecker product

The program also uses

W(a,k1) (x) W(b,k2) = W(a*b, k1*k2).

The corresponding method is:

```text
kron

```

with recipes such as:

```json
{
  "type": "kron",
  "left": [a,k1],
  "right": [b,k2]
}

```

### 5. Paley constructions

The program implements Paley constructions for appropriate prime powers.

For q ≡ 3 mod 4, Paley type I provides

W(q+1, q).

The methods stored are:

```text
paley1

```

and, after adding the identity,

```text
paley1_h

```

for the corresponding Hadamard matrix.

For suitable q ≡ 1 mod 4, Paley type II gives Hadamard matrices of order

2(q+1).

The corresponding method is:

```text
paley2

```

### 6. Goethals–Seidel

For cases not solved by the previous constructions, the program searches for a Goethals–Seidel array constructed from four ternary circulant sequences

A, B, C, D.

The corresponding method is:

```text
gs

```

The search can be exhaustive for small values of

m = n / 4

or use simulated annealing for larger values.

---

## Remaining k = n - 1 cases

The script

```text
remaining_cases.py

```

handles remaining cases of the form

W(n, n-1)

using skew-Hadamard matrices obtained from Paley type I constructions and doubling.

The corresponding JSON method is:

```text
skew_hadamard

```

with recipes of the form:

```json
{
  "type": "skew_hadamard",
  "n": n,
  "base": base,
  "q": q
}

```

This construction was used for the remaining cases not obtained during the first `all_chains.py` pass.

---

## Exact verification

The file

```text
verify_weighing.py

```

checks every stored matrix exactly.

To verify all `solution_*.json` files:

```bash
python verify_weighing.py solutions

```

To verify only the first solution in every folder:

```bash
python verify_weighing.py solutions --first-only

```

To obtain a verbose report:

```bash
python verify_weighing.py solutions --first-only --verbose

```

To export the verification results:

```bash
python verify_weighing.py solutions --first-only --export report.csv

```

An Excel report can also be generated:

```bash
python verify_weighing.py solutions --first-only --export report.xlsx

```

The matrix verification is exact and does not use floating-point tolerances.

---

## Exact verification conditions

For each matrix W, the verifier checks:

```text
W.shape == (n,n)

```

```text
entries(W) subset of {-1, 0, +1}

```

```text
every row has exactly k nonzero entries

```

```text
every column has exactly k nonzero entries

```

```text
W @ W.T == k * I

```

```text
W.T @ W == k * I

```

The matrix products are evaluated with integer arithmetic and compared exactly.

---

## Reproducing the constructions

To reproduce the full search:

```bash
python all_chains.py --nmin 4 --nmax 100 --out solutions

```

Then run the remaining skew-Hadamard constructions:

```bash
python remaining_cases.py --out solutions

```

Finally verify the complete dataset:

```bash
python verify_weighing.py solutions --first-only

```

Existing solution files are respected, so the programs can be resumed without rebuilding already stored cases.

---

## JSON format

A typical JSON file has the following structure:

```json
{
  "n": 8,
  "k": 3,
  "m": 2,
  "A": [],
  "B": [],
  "C": [],
  "D": [],
  "weights": [],
  "sums": [],
  "matrix": [
    [0,1,1,1,0,0,0,0],
    [-1,0,1,-1,0,0,0,0],
    [-1,-1,0,1,0,0,0,0],
    [-1,1,-1,0,0,0,0,0],
    [0,0,0,0,0,1,1,1],
    [0,0,0,0,-1,0,1,-1],
    [0,0,0,0,-1,-1,0,1],
    [0,0,0,0,-1,1,-1,0]
  ],
  "matrix_symbols": [
    ".+++....",
    "-.+-....",
    "--.+....",
    "-+-.....",
    ".....+++",
    "....-.+-",
    "....--.+",
    "....-+-."
  ],
  "method": "dsum",
  "recipe": {
    "type": "dsum",
    "parts": [[4,3],[4,3]]
  }
}

```

The symbolic representation uses:

```text
+  ->  1
-  -> -1
.  ->  0

```

For `gs` solutions, the fields

```text
A
B
C
D
weights
sums

```

contain the Goethals–Seidel generators and their associated data.

For constructions that do not use Goethals–Seidel, these fields may be empty.

---

## Interpretation of the computational results

This repository provides **one constructed and exactly verified matrix for each pair (n,k)** in the specified range.

It does **not** classify all weighing matrices up to equivalence. In general, a fixed W(n,k) may have many inequivalent representatives.

The construction method stored in each JSON file indicates the method that produced the representative stored in that file.

---

## Exhaustive versus heuristic search

There is an important distinction between the different Goethals–Seidel search modes.

For small values of

m = n / 4,

the program can perform a complete exhaustive GS search.

In that situation, failure to find a solution means that **no Goethals–Seidel solution exists within the searched GS construction space**.

For larger values of m, the program can use simulated annealing.

In that case:

* finding a solution provides an explicit construction;
* failure to find a solution is **not** a proof that no solution exists.

In every case, however, a stored candidate is subjected to exact matrix verification before being accepted.

---

## Results, hardware, software environment and computation times

The file

```text
report.csv

```

contains the verification report.

The complete dataset contains **1512 stored cases**.

The computation log used to generate the table records the construction method and elapsed time for each (n,k).

Computation was run on an MSI GL65 9SDK laptop (Intel Core i7-9750H @ 2.60 GHz, 16 GB RAM, Windows 11 Home), Python 3.11.4, NumPy 1.26.4. Total Goethals–Seidel annealing time was ≈ 26.6 h for n ≤ 100 (of which ≈ 19.3 h for n = 100 alone), ≈ 86.2 h for n = 104, and ≈ 99.7 h for n = 108.

---

## License

This project is released under the

**Apache License 2.0**

---

## Author

**Adrián Macías-Quintero**

Copyright © 2026 Adrián Macías-Quintero


