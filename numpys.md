# 🔢 Complete NumPy for Data Analysis — From Basics to Advanced

> A complete, beginner-to-advanced reference guide for NumPy with examples, multiple approaches, and practice problems.

---

## 📋 Table of Contents

1. [What is NumPy?](#1-what-is-numpy)
2. [Installation & Import](#2-installation--import)
3. [NumPy Arrays — ndarray](#3-numpy-arrays--ndarray)
4. [Creating Arrays](#4-creating-arrays)
5. [Array Attributes](#5-array-attributes)
6. [Array Data Types (dtype)](#6-array-data-types-dtype)
7. [Indexing & Slicing](#7-indexing--slicing)
8. [Reshaping & Resizing](#8-reshaping--resizing)
9. [Array Operations & Math](#9-array-operations--math)
10. [Broadcasting](#10-broadcasting)
11. [Comparison & Boolean Operations](#11-comparison--boolean-operations)
12. [Aggregation & Statistics](#12-aggregation--statistics)
13. [Sorting & Searching](#13-sorting--searching)
14. [Linear Algebra](#14-linear-algebra)
15. [Random Number Generation](#15-random-number-generation)
16. [Stacking & Splitting Arrays](#16-stacking--splitting-arrays)
17. [Copying Arrays](#17-copying-arrays)
18. [Universal Functions (ufuncs)](#18-universal-functions-ufuncs)
19. [Fancy Indexing & Advanced Indexing](#19-fancy-indexing--advanced-indexing)
20. [Structured Arrays](#20-structured-arrays)
21. [File I/O with NumPy](#21-file-io-with-numpy)
22. [Performance & Memory Tips](#22-performance--memory-tips)
23. [NumPy + Pandas Integration](#23-numpy--pandas-integration)
24. [Practice Problems](#24-practice-problems)

---

## 1. What is NumPy?

**NumPy** (Numerical Python) is the foundational library for numerical computing in Python. It provides:

- **ndarray** — a fast, multi-dimensional array object
- **Mathematical functions** — element-wise math, linear algebra, statistics
- **Broadcasting** — operations between arrays of different shapes
- **C-level speed** — operations execute in compiled C code, not Python loops

```
Python List  →  slow, flexible, any types
NumPy Array  →  fast, fixed type, multi-dimensional, vectorized
```

### Why NumPy over Python lists?

```python
import numpy as np
import time

size = 1_000_000

# Python list
lst = list(range(size))
start = time.time()
result = [x * 2 for x in lst]
print(f"List: {time.time() - start:.4f}s")

# NumPy array
arr = np.arange(size)
start = time.time()
result = arr * 2
print(f"NumPy: {time.time() - start:.4f}s")

# NumPy is typically 10–100x faster
```

---

## 2. Installation & Import

```bash
pip install numpy
```

```python
import numpy as np        # standard alias — always use np

print(np.__version__)     # check version
```

---

## 3. NumPy Arrays — ndarray

The core object in NumPy is the **ndarray** (N-dimensional array).

Key properties:
- All elements must be the **same data type**
- Size is **fixed** after creation
- Supports **vectorized** operations (no loops needed)
- Memory is **contiguous** — fast to access

```
1D:  [1, 2, 3, 4]                         → shape (4,)
2D:  [[1, 2, 3],                           → shape (3, 3)
      [4, 5, 6],
      [7, 8, 9]]
3D:  [[[1,2],[3,4]],[[5,6],[7,8]]]         → shape (2, 2, 2)
```

---

## 4. Creating Arrays

### From Python Data

```python
# --- From a list (1D) ---
a = np.array([1, 2, 3, 4, 5])
print(a)           # [1 2 3 4 5]
print(type(a))     # <class 'numpy.ndarray'>

# --- From a list of lists (2D) ---
a = np.array([[1, 2, 3],
              [4, 5, 6]])
# shape: (2, 3)

# --- 3D array ---
a = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])
# shape: (2, 2, 2)

# --- With explicit dtype ---
a = np.array([1, 2, 3], dtype=float)
a = np.array([1, 2, 3], dtype=np.int32)
a = np.array([1, 2, 3], dtype=np.complex128)
```

### Built-in Array Creators

```python
# --- zeros: all zeros ---
np.zeros(5)                    # [0. 0. 0. 0. 0.]
np.zeros((3, 4))               # 3x4 matrix of zeros
np.zeros((2, 3, 4))            # 3D array of zeros

# --- ones: all ones ---
np.ones(5)
np.ones((3, 4))

# --- full: filled with a constant ---
np.full(5, 7)                  # [7 7 7 7 7]
np.full((3, 3), 9)             # 3x3 matrix of 9s

# --- empty: uninitialized (garbage values, fast) ---
np.empty((3, 3))

# --- eye: identity matrix ---
np.eye(3)                      # 3x3 identity matrix
np.eye(3, dtype=int)

# --- identity: same as eye ---
np.identity(4)

# --- diag: diagonal matrix ---
np.diag([1, 2, 3])             # [[1,0,0],[0,2,0],[0,0,3]]
np.diag(a)                     # extract diagonal from 2D array

# --- like functions: same shape as existing array ---
np.zeros_like(a)               # zeros with same shape and dtype as a
np.ones_like(a)
np.full_like(a, 99)
np.empty_like(a)
```

### Range-Based Array Creators

```python
# --- arange: like Python range but returns array ---
np.arange(5)                   # [0 1 2 3 4]
np.arange(1, 10)               # [1 2 3 4 5 6 7 8 9]
np.arange(0, 1, 0.2)           # [0.  0.2 0.4 0.6 0.8] (supports float step)
np.arange(10, 0, -2)           # [10  8  6  4  2]

# --- linspace: evenly spaced between start and stop (inclusive) ---
np.linspace(0, 1, 5)           # [0.   0.25 0.5  0.75 1.  ]
np.linspace(0, 10, 100)        # 100 points between 0 and 10
np.linspace(0, 1, 5, endpoint=False)  # exclude endpoint

# --- logspace: evenly spaced on a log scale ---
np.logspace(0, 3, 4)           # [1, 10, 100, 1000]

# --- geomspace: evenly spaced geometric progression ---
np.geomspace(1, 1000, 4)       # [1, 10, 100, 1000]
```

---

## 5. Array Attributes

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

a.ndim        # 2          → number of dimensions
a.shape       # (2, 3)     → (rows, columns)
a.size        # 6          → total number of elements
a.dtype       # dtype('int64') → element data type
a.itemsize    # 8          → bytes per element
a.nbytes      # 48         → total bytes (size * itemsize)

# Change shape (does NOT copy)
a.shape = (3, 2)   # reshape in-place (must keep same total size)

# Transpose
a.T            # swap axes → (3,2) becomes (2,3)
a.transpose()  # same as .T but can specify axis order
```

---

## 6. Array Data Types (dtype)

```python
# Integer types
np.int8     # -128 to 127
np.int16    # -32768 to 32767
np.int32    # -2B to 2B
np.int64    # default on most systems

# Unsigned integers
np.uint8    # 0 to 255
np.uint16
np.uint32
np.uint64

# Float types
np.float16  # half precision
np.float32  # single precision
np.float64  # double precision (default)

# Complex
np.complex64
np.complex128

# Boolean
np.bool_

# String / object
np.str_
np.object_

# Check and convert dtype
a.dtype                          # current dtype
a.astype(np.float32)             # convert to float32
a.astype('float64')              # string alias also works
a.astype(int)                    # Python int → int64

# dtype info
np.iinfo(np.int8)                # min/max for integer types
np.finfo(np.float32)             # min/max/eps for float types
```

---

## 7. Indexing & Slicing

### 1D Indexing

```python
a = np.array([10, 20, 30, 40, 50])

a[0]          # 10        → first element
a[-1]         # 50        → last element
a[-2]         # 40        → second to last

a[1:4]        # [20 30 40]  → slice (start:stop, exclusive stop)
a[:3]         # [10 20 30]
a[2:]         # [30 40 50]
a[::2]        # [10 30 50]  → every other element (step=2)
a[::-1]       # [50 40 30 20 10] → reverse
a[1:4:2]      # [20 40]    → start:stop:step
```

### 2D Indexing

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Single element
a[0, 0]       # 1    → row 0, col 0
a[1, 2]       # 6    → row 1, col 2
a[-1, -1]     # 9    → last row, last col

# Row selection
a[0]          # [1 2 3]    → first row
a[0, :]       # [1 2 3]    → same, explicit
a[[0, 2]]     # rows 0 and 2

# Column selection
a[:, 0]       # [1 4 7]    → first column
a[:, 1:3]     # columns 1 and 2

# Submatrix
a[0:2, 0:2]   # [[1,2],[4,5]] → top-left 2x2

# All rows, specific cols
a[:, [0, 2]]  # columns 0 and 2
```

### 3D Indexing

```python
a = np.zeros((2, 3, 4))   # 2 blocks, 3 rows, 4 cols

a[0]           # first block → shape (3, 4)
a[0, 1]        # first block, second row → shape (4,)
a[0, 1, 2]     # single element
a[:, :, 0]     # first column of every block
```

---

## 8. Reshaping & Resizing

```python
a = np.arange(12)      # [0 1 2 3 4 5 6 7 8 9 10 11]

# --- reshape: returns new view (if possible) ---
a.reshape(3, 4)        # 3 rows, 4 cols
a.reshape(2, 6)
a.reshape(2, 2, 3)     # 3D
a.reshape(-1, 4)       # -1 means "figure it out" → (3, 4)
a.reshape(3, -1)       # → (3, 4)

# --- np.reshape ---
np.reshape(a, (3, 4))

# --- ravel: flatten to 1D (returns view if possible) ---
a.ravel()              # [0 1 2 3 ...]

# --- flatten: flatten to 1D (always returns copy) ---
a.flatten()

# --- squeeze: remove dimensions of size 1 ---
a = np.array([[[1], [2], [3]]])   # shape (1, 3, 1)
a.squeeze()                        # shape (3,)
np.squeeze(a, axis=0)              # remove only axis 0

# --- expand_dims: add a new dimension ---
a = np.array([1, 2, 3])           # shape (3,)
np.expand_dims(a, axis=0)         # shape (1, 3) → row vector
np.expand_dims(a, axis=1)         # shape (3, 1) → column vector
a[np.newaxis, :]                  # same as expand_dims(a, 0)
a[:, np.newaxis]                  # same as expand_dims(a, 1)

# --- resize: changes array in-place (repeats if needed) ---
a = np.arange(6)
a.resize(3, 4)         # fills extra with zeros or repeats values
```

---

## 9. Array Operations & Math

### Arithmetic (Element-wise)

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Basic arithmetic — all element-wise
a + b          # [11 22 33 44]
a - b          # [-9 -18 -27 -36]
a * b          # [10 40 90 160]
a / b          # [0.1 0.1 0.1 0.1]
a // b         # [0 0 0 0]    floor division
a % b          # [1 2 3 4]    modulo
a ** 2         # [1 4 9 16]   power
-a             # [-1 -2 -3 -4]

# Scalar operations (broadcasts automatically)
a + 10         # [11 12 13 14]
a * 3          # [3 6 9 12]
a ** 2         # [1 4 9 16]

# In-place operations (modify original)
a += 1
a *= 2
```

### Math Functions

```python
a = np.array([1.0, 4.0, 9.0, 16.0])

np.sqrt(a)         # [1. 2. 3. 4.]
np.cbrt(a)         # cube root
np.square(a)       # element-wise square
np.abs(a)          # absolute value
np.exp(a)          # e^x
np.exp2(a)         # 2^x
np.log(a)          # natural log
np.log2(a)         # log base 2
np.log10(a)        # log base 10
np.log1p(a)        # log(1 + x) — numerically stable for small x
np.sign(a)         # -1, 0, or 1
np.floor(a)        # round down
np.ceil(a)         # round up
np.round(a, 2)     # round to 2 decimal places
np.trunc(a)        # truncate toward zero

# Trigonometric
np.sin(a)
np.cos(a)
np.tan(a)
np.arcsin(a)
np.arccos(a)
np.arctan(a)
np.arctan2(y, x)   # angle from x-axis
np.degrees(a)      # radians to degrees
np.radians(a)      # degrees to radians
np.pi              # 3.14159...

# Hyperbolic
np.sinh(a)
np.cosh(a)
np.tanh(a)

# Special
np.clip(a, 2, 10)      # clip values to [2, 10]
np.where(a > 5, a, 0)  # conditional element selection
```

### Matrix Operations

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication
A * B

# Matrix multiplication (dot product)
A @ B                    # recommended syntax (Python 3.5+)
np.dot(A, B)             # same
np.matmul(A, B)          # same (no scalars)

# Dot product of 1D vectors
np.dot([1,2,3], [4,5,6])  # 1*4 + 2*5 + 3*6 = 32

# Cross product
np.cross([1,0,0], [0,1,0])   # [0 0 1]
```

---

## 10. Broadcasting

Broadcasting allows NumPy to perform operations on arrays of **different shapes** by "stretching" the smaller array.

### Rules
1. If arrays have different ndim, pad the smaller shape on the **left** with 1s
2. Arrays with size 1 along a dimension are stretched to match
3. Arrays must be compatible (same size or one of them is 1) in every dimension

```python
# Scalar + array
np.array([1, 2, 3]) + 10          # [11 12 13]

# 1D + 2D
a = np.array([[1, 2, 3],           # shape (3, 3)
              [4, 5, 6],
              [7, 8, 9]])
b = np.array([10, 20, 30])         # shape (3,) → broadcast to (3, 3)
a + b
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]

# Column vector broadcasting
col = np.array([[10], [20], [30]])  # shape (3, 1)
a + col
# [[11 12 13]
#  [24 25 26]
#  [37 38 39]]

# Outer product via broadcasting
row = np.array([1, 2, 3])         # shape (3,)
col = np.array([[1], [2], [3]])   # shape (3, 1)
row * col
# [[1 2 3]
#  [2 4 6]
#  [3 6 9]]
```

### Broadcasting Shape Compatibility

```
Shape A: (3, 4)    Shape A: (5, 1, 4)    Shape A: (3,)
Shape B: (   4)    Shape B: (   3, 1)    Shape B: (4,)
Result:  (3, 4)    Result:  (5, 3, 4)    Result:  ERROR (not compatible)
```

---

## 11. Comparison & Boolean Operations

### Comparison Operators

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

a == b         # [F F T F F]
a != b         # [T T F T T]
a >  b         # [F F F T T]
a <  b         # [T T F F F]
a >= b         # [F F T T T]
a <= b         # [T T T F F]

# Returns boolean arrays — useful for filtering
a[a > 3]       # [4 5]
a[a % 2 == 0]  # [2 4]
```

### Logical Functions

```python
a = np.array([True, False, True])
b = np.array([True, True, False])

np.logical_and(a, b)   # [T F F]
np.logical_or(a, b)    # [T T T]
np.logical_not(a)      # [F T F]
np.logical_xor(a, b)   # [F T T]

# Bitwise operators (same result for bool arrays)
a & b                  # logical AND
a | b                  # logical OR
~a                     # logical NOT
a ^ b                  # logical XOR
```

### any() and all()

```python
a = np.array([1, 2, 3, 4, 5])

np.any(a > 4)        # True  → at least one element > 4
np.all(a > 0)        # True  → all elements > 0
np.any(a > 10)       # False

# Along an axis
a = np.array([[1, 0], [1, 1]])
np.any(a, axis=0)    # [True True]    → any True per column
np.all(a, axis=1)    # [False True]   → all True per row
```

### np.where()

```python
a = np.array([1, -2, 3, -4, 5])

# Condition: if a > 0 keep value, else replace with 0
np.where(a > 0, a, 0)        # [1 0 3 0 5]

# Replace all negatives with -1, positives with 1
np.where(a > 0, 1, -1)       # [1 -1 1 -1 1]

# Get indices where condition is True
np.where(a > 0)               # (array([0, 2, 4]),)
np.nonzero(a)                  # same as np.where(a)

# np.select: multiple conditions
conditions = [a < 0, a == 0, a > 0]
choices = [-1, 0, 1]
np.select(conditions, choices) # [-1 would be replaced per condition]
```

---

## 12. Aggregation & Statistics

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# Basic aggregations
np.sum(a)             # 21           → sum of all elements
np.sum(a, axis=0)     # [5 7 9]      → column sums
np.sum(a, axis=1)     # [6 15]       → row sums

np.prod(a)            # 720          → product of all
np.cumsum(a)          # [1 3 6 10 15 21]
np.cumprod(a)         # [1 2 6 24 120 720]

# Statistics
np.min(a)             # 1
np.max(a)             # 6
np.min(a, axis=0)     # [1 2 3]
np.max(a, axis=1)     # [3 6]

np.mean(a)            # 3.5
np.median(a)          # 3.5
np.std(a)             # standard deviation
np.var(a)             # variance
np.std(a, ddof=1)     # sample std (ddof=1), population std (ddof=0, default)

# Index of min/max
np.argmin(a)          # 0    → flat index
np.argmax(a)          # 5    → flat index
np.argmin(a, axis=0)  # [0 0 0]  → row index of min per column
np.argmax(a, axis=1)  # [2 2]    → col index of max per row

# Percentile / Quantile
np.percentile(a, 50)              # 50th percentile (median)
np.percentile(a, [25, 50, 75])    # quartiles
np.quantile(a, 0.75)              # same as percentile but 0-1 scale

# Weighted average
np.average(a, weights=[[1,2,3],[1,2,3]])

# Correlation and covariance
np.corrcoef([1,2,3,4], [2,4,6,8])   # correlation matrix
np.cov([1,2,3,4], [2,4,6,8])        # covariance matrix

# Histogram counts
np.histogram([1,2,1,3,4,2,1], bins=4)  # (counts, bin_edges)
np.bincount([0,1,1,2,0,3])             # count of each integer
np.unique(a, return_counts=True)        # unique values with counts
```

---

## 13. Sorting & Searching

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# --- Sort ---
np.sort(a)                    # returns sorted copy [1 1 2 3 4 5 6 9]
a.sort()                      # sort IN-PLACE (modifies original)
np.sort(a)[::-1]              # descending sort (reverse after sorting)

# 2D sorting
a = np.array([[3,1,2],[6,4,5]])
np.sort(a, axis=1)            # sort each row
np.sort(a, axis=0)            # sort each column

# --- argsort: returns INDICES that would sort the array ---
a = np.array([30, 10, 40, 20])
np.argsort(a)                 # [1 3 0 2] → sort by these indices
a[np.argsort(a)]              # same as np.sort(a)

# Stable sort
np.argsort(a, kind='stable')

# --- Searching ---
a = np.array([10, 20, 30, 40, 50])

np.searchsorted(a, 25)        # 2 → index to insert 25 (binary search)
np.searchsorted(a, [15, 35])  # [1 3]

# Find indices of condition
np.where(a > 25)              # (array([2, 3, 4]),)
np.argwhere(a > 25)           # [[2],[3],[4]] → 2D version

# Unique values
np.unique(a)
np.unique(a, return_index=True)    # also return first occurrence index
np.unique(a, return_inverse=True)  # also return inverse mapping
np.unique(a, return_counts=True)   # also return counts
```

---

## 14. Linear Algebra

```python
# All linear algebra lives in np.linalg
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
A @ B
np.dot(A, B)
np.matmul(A, B)

# Transpose
A.T

# Determinant
np.linalg.det(A)           # -2.0

# Inverse
np.linalg.inv(A)           # [[-2. 1.][1.5 -0.5]]

# Rank
np.linalg.matrix_rank(A)   # 2

# Trace (sum of diagonal)
np.trace(A)                # 5

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(A)

# Solve linear equations: Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)    # solves A @ x = b

# Least squares solution (overdetermined systems)
x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

# Norms
np.linalg.norm(A)             # Frobenius norm (default)
np.linalg.norm(A, 'fro')      # Frobenius norm
np.linalg.norm(A, 1)          # L1 norm
np.linalg.norm(A, 2)          # L2 / spectral norm
np.linalg.norm([3, 4])        # vector norm → 5.0
```

---

## 15. Random Number Generation

```python
# Modern way: use np.random.default_rng() (recommended)
rng = np.random.default_rng(seed=42)   # seed for reproducibility

# --- Integer random ---
rng.integers(0, 10)                    # single int in [0, 10)
rng.integers(0, 10, size=5)            # array of 5 ints
rng.integers(0, 10, size=(3, 3))       # 3x3 matrix

# --- Float random [0, 1) ---
rng.random()                           # single float
rng.random(size=5)                     # array of 5 floats
rng.random(size=(3, 4))                # 3x4 matrix

# --- Normal distribution ---
rng.normal(loc=0, scale=1, size=100)   # mean=0, std=1
rng.normal(50, 10, size=(5, 5))        # mean=50, std=10

# --- Uniform distribution [low, high) ---
rng.uniform(0, 100, size=10)

# --- Other distributions ---
rng.exponential(scale=1.0, size=100)
rng.poisson(lam=5, size=100)
rng.binomial(n=10, p=0.5, size=100)
rng.choice([1, 2, 3, 4, 5], size=3)          # sample without replacement by default
rng.choice([1, 2, 3, 4, 5], size=3, replace=True)   # with replacement

# Shuffle
arr = np.array([1, 2, 3, 4, 5])
rng.shuffle(arr)         # in-place shuffle
rng.permutation(arr)     # returns shuffled copy (original unchanged)

# --- Legacy API (still common, avoid for new code) ---
np.random.seed(42)
np.random.rand(3, 4)           # uniform [0,1) → shape (3,4)
np.random.randn(3, 4)          # standard normal → shape (3,4)
np.random.randint(0, 10, 5)    # 5 random ints in [0,10)
np.random.choice(arr, size=3)
np.random.shuffle(arr)
```

---

## 16. Stacking & Splitting Arrays

### Stacking (Combining)

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# --- vstack: vertical stack (rows) ---
np.vstack([a, b])
# [[1 2 3]
#  [4 5 6]]

# --- hstack: horizontal stack (columns) ---
np.hstack([a, b])
# [1 2 3 4 5 6]

# --- column_stack: stack 1D as columns ---
np.column_stack([a, b])
# [[1 4]
#  [2 5]
#  [3 6]]

# --- concatenate: general purpose ---
np.concatenate([a, b])              # default axis=0 → [1 2 3 4 5 6]
np.concatenate([a, b], axis=0)

# 2D stacking
A = np.ones((2, 3))
B = np.ones((2, 3))
np.concatenate([A, B], axis=0)     # shape (4, 3) → stack rows
np.concatenate([A, B], axis=1)     # shape (2, 6) → stack columns

# --- stack: creates NEW axis ---
np.stack([a, b])           # shape (2, 3) — new axis at 0
np.stack([a, b], axis=1)   # shape (3, 2) — new axis at 1

# --- dstack: depth stack (along 3rd axis) ---
np.dstack([a, b])          # shape (1, 3, 2)
```

### Splitting

```python
a = np.array([1, 2, 3, 4, 5, 6])

# --- split: into equal parts ---
np.split(a, 2)             # [array([1,2,3]), array([4,5,6])]
np.split(a, 3)             # [array([1,2]), array([3,4]), array([5,6])]

# Split at specific indices
np.split(a, [2, 4])        # [array([1,2]), array([3,4]), array([5,6])]

# --- array_split: allows unequal splits ---
np.array_split(a, 4)       # [array([1,2]), array([3,4]), array([5]), array([6])]

# 2D splitting
A = np.arange(16).reshape(4, 4)
np.hsplit(A, 2)            # split horizontally into 2 (by column)
np.vsplit(A, 2)            # split vertically into 2 (by row)
np.dsplit(A, 2)            # split along depth (3D only)
```

---

## 17. Copying Arrays

```python
a = np.array([1, 2, 3])

# --- View (shallow copy) — shares memory ---
b = a.view()
b = a[:]           # slice is also a view
b = a.reshape(1,3) # reshape may return a view
b[0] = 99          # changes a too!
b.base is a        # True → b is a view of a

# --- Copy (deep copy) — independent ---
b = a.copy()
b = np.copy(a)
b[0] = 99          # does NOT change a
b.base is None     # True → b owns its data

# Check if array owns its data
a.flags['OWNDATA']         # True if owns data
np.shares_memory(a, b)     # True if a and b share memory

# When does slicing create a view vs copy?
# Slicing: always a VIEW (be careful!)
# Fancy indexing (using array of indices): always a COPY
# Boolean indexing: always a COPY
```

---

## 18. Universal Functions (ufuncs)

Universal functions (ufuncs) operate **element-wise** on arrays and are implemented in C for speed.

```python
a = np.array([1.0, 4.0, 9.0])

# All math functions we've seen are ufuncs
np.sqrt(a)           # element-wise sqrt
np.add(a, a)         # same as a + a
np.multiply(a, 2)    # same as a * 2

# ufunc methods
np.add.reduce(a)          # same as np.sum(a)
np.multiply.reduce(a)     # same as np.prod(a)
np.add.accumulate(a)      # same as np.cumsum(a)
np.add.outer([1,2], [3,4]) # outer product: [[4,5],[5,6]]

# Custom ufunc via np.frompyfunc
def double(x):
    return x * 2

double_ufunc = np.frompyfunc(double, 1, 1)
double_ufunc(a)         # [2. 8. 18.]

# np.vectorize (wraps a function to work on arrays)
vfunc = np.vectorize(lambda x: 'high' if x > 5 else 'low')
vfunc(np.array([3, 7, 2, 9]))   # ['low' 'high' 'low' 'high']
# Note: np.vectorize is NOT faster than a loop — just convenient
```

---

## 19. Fancy Indexing & Advanced Indexing

### Fancy Indexing (Integer Array Indexing)

```python
a = np.array([10, 20, 30, 40, 50])

# Select specific elements by index array
idx = [0, 2, 4]
a[idx]              # [10 30 50]
a[[1, 3]]           # [20 40]
a[np.array([0,0,1,2])]   # [10 10 20 30] — duplicates allowed

# 2D fancy indexing
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
rows = [0, 2]
cols = [1, 2]
A[rows, cols]       # [2 9]  → A[0,1] and A[2,2]

# Select multiple rows
A[[0, 2]]           # rows 0 and 2
A[[0, 2], :]        # same, explicit

# Select multiple columns
A[:, [0, 2]]        # columns 0 and 2

# Outer-product style: all combinations
A[np.ix_([0,2], [0,2])]  # 2x2 submatrix at intersections
```

### Boolean Indexing

```python
a = np.array([1, -2, 3, -4, 5, -6])

# Boolean mask
mask = a > 0
a[mask]             # [1 3 5]
a[a > 0]            # same in one line

# Modify elements matching condition
a[a < 0] = 0        # replace negatives with 0
a[a > 4] *= 2       # double all values > 4

# 2D boolean indexing
A = np.array([[1,2,3],[4,5,6]])
A[A > 3]            # [4 5 6] → flattened result
A[A > 3] = 0        # zero out all elements > 3

# np.where with boolean index
np.where(a > 0)
```

---

## 20. Structured Arrays

Structured arrays allow **mixed data types** (like a simple table) in a single NumPy array.

```python
# Define a structured dtype
dt = np.dtype([
    ('name', 'U20'),          # Unicode string, max 20 chars
    ('age', np.int32),
    ('salary', np.float64)
])

# Create structured array
data = np.array([
    ('Alice', 25, 75000.0),
    ('Bob',   30, 85000.0),
    ('Charlie',22, 65000.0)
], dtype=dt)

# Access by field name
data['name']          # ['Alice' 'Bob' 'Charlie']
data['age']           # [25 30 22]
data['salary']        # [75000. 85000. 65000.]

# Access a single record
data[0]               # ('Alice', 25, 75000.)
data[0]['name']       # 'Alice'

# Filtering on a field
data[data['age'] > 24]

# Sort by field
np.sort(data, order='salary')
np.sort(data, order=['age', 'salary'])  # sort by multiple fields
```

---

## 21. File I/O with NumPy

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

# --- Binary format (.npy) — single array ---
np.save('array.npy', a)
a_loaded = np.load('array.npy')

# --- Binary format (.npz) — multiple arrays ---
np.savez('arrays.npz', arr1=a, arr2=a*2)
loaded = np.load('arrays.npz')
loaded['arr1']
loaded['arr2']

# Compressed npz
np.savez_compressed('arrays_compressed.npz', arr1=a)

# --- Text format (.txt / .csv) ---
np.savetxt('array.txt', a)
np.savetxt('array.csv', a, delimiter=',', header='a,b,c', comments='')
np.savetxt('array.txt', a, fmt='%.4f')   # format specifier

a_loaded = np.loadtxt('array.txt')
a_loaded = np.loadtxt('array.csv', delimiter=',', skiprows=1)  # skip header

# --- genfromtxt: handles missing values ---
a = np.genfromtxt('data.csv', delimiter=',', names=True, dtype=None,
                   encoding='utf-8', filling_values=0)

# Note: For large tabular data with mixed types, prefer Pandas
```

---

## 22. Performance & Memory Tips

```python
# 1. Use appropriate dtypes — smaller = faster + less memory
a = np.array([1, 2, 3], dtype=np.int8)      # 1 byte each (vs 8 for int64)
a = np.array([1.0, 2.0], dtype=np.float32)  # 4 bytes (vs 8 for float64)

# 2. Avoid unnecessary copies
# Views are memory-efficient — slices, reshape, transpose are views
b = a[1:3]        # view — no copy
b = a.T           # view — no copy
b = a.copy()      # explicit copy when you need independence

# 3. Vectorize everything — avoid Python loops
# SLOW
total = 0
for x in a:
    total += x
# FAST
total = np.sum(a)

# 4. Use in-place operations to avoid allocating new arrays
a += 1            # in-place (no new array)
a = a + 1         # creates new array

# 5. Contiguous memory — C vs Fortran order
a = np.ascontiguousarray(a)    # ensure C-order (row-major)
a = np.asfortranarray(a)       # Fortran-order (column-major)

# 6. np.einsum for complex linear algebra (very fast)
# Dot product
np.einsum('i,i->', a, b)
# Matrix multiply
np.einsum('ij,jk->ik', A, B)
# Trace
np.einsum('ii->', A)

# 7. Memory mapping for huge arrays (larger than RAM)
fp = np.memmap('large.dat', dtype='float32', mode='w+', shape=(1000, 1000))
fp[0] = np.arange(1000)

# 8. Check memory usage
a.nbytes                         # total bytes
a.itemsize                       # bytes per element

# 9. Use np.empty instead of np.zeros when you'll fill it anyway
a = np.empty((1000, 1000))      # faster — no initialization
a = np.zeros((1000, 1000))      # slower — initializes to zero

# 10. Profile with timeit
import timeit
timeit.timeit(lambda: np.sum(a), number=1000)
```

---

## 23. NumPy + Pandas Integration

```python
import pandas as pd
import numpy as np

# NumPy → Pandas
arr = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
s = pd.Series(np.arange(5))

# Pandas → NumPy
arr = df.values                   # returns underlying NumPy array
arr = df.to_numpy()               # recommended (same result)
arr = df['A'].to_numpy()

# NumPy functions work on Pandas objects
np.mean(df['A'])
np.sqrt(df['A'])
np.where(df['A'] > 2, 'high', 'low')

# Use NumPy for fast math inside Pandas
df['log_A'] = np.log(df['A'])
df['clipped'] = np.clip(df['A'], 1, 5)

# Apply NumPy ufuncs directly on Pandas Series
df['A'].apply(np.sqrt)            # slower
np.sqrt(df['A'])                  # faster

# NumPy random for creating sample DataFrames
rng = np.random.default_rng(42)
df = pd.DataFrame({
    'age':    rng.integers(18, 65, size=100),
    'salary': rng.normal(50000, 15000, size=100),
    'score':  rng.uniform(0, 100, size=100)
})

# Shared memory between NumPy and Pandas
arr = np.array([1, 2, 3])
s = pd.Series(arr)
arr[0] = 99
print(s[0])     # also 99 — they share memory!
# Use .copy() to avoid this:
s = pd.Series(arr.copy())
```

---

## 24. Practice Problems

### 🟢 Beginner

**Problem 1:** Array Basics
```python
# Create the following:
# a) A 1D array of even numbers from 2 to 20
# b) A 3x3 matrix filled with 7s
# c) A 4x4 identity matrix
# d) 10 evenly spaced points between 0 and pi

# Your solution here...
```

**Problem 2:** Array Operations
```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
# a) Find sum, mean, min, max, and std
# b) Sort the array
# c) Find all elements greater than 4
# d) Replace all elements greater than 5 with 5 (clipping)
```

---

### 🟡 Intermediate

**Problem 3:** Matrix Operations
```python
A = np.array([[2, 1], [5, 3]])
b = np.array([4, 7])
# a) Compute A @ A (matrix squared)
# b) Solve the linear system Ax = b
# c) Find the determinant and inverse of A
# d) Verify: A @ A_inverse == identity matrix
```

**Problem 4:** Broadcasting Challenge
```python
# Without any loops:
# Create a multiplication table from 1 to 10 (10x10 matrix)
# Hint: use np.arange and broadcasting

# Expected output:
# [[ 1  2  3  4  5  6  7  8  9 10]
#  [ 2  4  6  8 10 12 14 16 18 20]
#  ...
#  [10 20 30 40 50 60 70 80 90 100]]
```

**Problem 5:** Statistics on Real Data
```python
rng = np.random.default_rng(42)
scores = rng.normal(70, 15, size=200)   # exam scores

# a) Find mean, median, std
# b) Count how many students scored above 85
# c) What percentage scored below 60?
# d) Find the 25th and 75th percentile
# e) Normalize scores to [0, 1] range
# f) Standardize scores (z-score normalization)
```

---

### 🔴 Advanced

**Problem 6:** Image as Array
```python
# A grayscale image is just a 2D NumPy array of pixel values (0–255)
rng = np.random.default_rng(0)
image = rng.integers(0, 256, size=(100, 100), dtype=np.uint8)

# a) Flip the image horizontally and vertically
# b) Crop the center 50x50 pixels
# c) Threshold: set pixels > 128 to 255, else to 0
# d) Compute a 3x3 rolling mean (manual, no libraries)
# e) Rotate 90 degrees clockwise
```

**Problem 7:** Performance Challenge
```python
# Implement the following WITHOUT any Python loops:
# Given a 2D array of shape (1000, 1000):
# a) Compute the row-wise dot product with a 1D vector of shape (1000,)
# b) Find the row with the maximum sum
# c) Normalize each row to have unit L2 norm
# d) Replace all NaN values with the column mean
```

**Problem 8:** Linear Algebra Application
```python
# Implement simple Linear Regression using only NumPy:
rng = np.random.default_rng(42)
X = rng.uniform(0, 10, size=100)
y = 3 * X + 7 + rng.normal(0, 2, size=100)   # y = 3x + 7 + noise

# a) Add a bias column to X (column of ones) → X_b
# b) Solve for weights using the Normal Equation:
#    w = (X_b.T @ X_b)^-1 @ X_b.T @ y
# c) Print the slope and intercept
# d) Compute MSE (Mean Squared Error) on the training data

# Expected: slope ≈ 3.0, intercept ≈ 7.0
```

---

## 📌 Quick Reference Cheat Sheet

```python
# ── IMPORT ──────────────────────────────────────
import numpy as np

# ── CREATE ──────────────────────────────────────
np.array([1, 2, 3])              # from list
np.zeros((3, 4))                 # zeros
np.ones((3, 4))                  # ones
np.full((3, 3), 7)               # filled with 7
np.eye(4)                        # identity matrix
np.arange(0, 10, 2)              # [0 2 4 6 8]
np.linspace(0, 1, 5)             # 5 points 0 to 1
np.random.default_rng(42).random((3, 4))  # random floats

# ── ATTRIBUTES ──────────────────────────────────
a.shape                          # dimensions tuple
a.ndim                           # number of dims
a.size                           # total elements
a.dtype                          # element type
a.nbytes                         # memory in bytes

# ── RESHAPE ─────────────────────────────────────
a.reshape(3, 4)                  # new shape
a.flatten()                      # to 1D (copy)
a.T                              # transpose
np.expand_dims(a, axis=0)        # add dimension
a.squeeze()                      # remove size-1 dims

# ── INDEX & SLICE ────────────────────────────────
a[0]                             # first element
a[-1]                            # last element
a[1:4]                           # slice
a[::2]                           # every other
a[a > 5]                         # boolean filter
a[[0, 2, 4]]                     # fancy indexing

# 2D
a[0, :]                          # first row
a[:, 0]                          # first column
a[0:2, 1:3]                      # submatrix

# ── MATH ────────────────────────────────────────
np.sum(a) / np.mean(a) / np.std(a)
np.min(a) / np.max(a)
np.argmin(a) / np.argmax(a)
np.sort(a) / np.argsort(a)
np.unique(a)
np.clip(a, 0, 10)
np.where(a > 5, a, 0)

# ── SHAPE OPS ────────────────────────────────────
np.concatenate([a, b], axis=0)
np.vstack([a, b])
np.hstack([a, b])
np.split(a, 3)

# ── LINALG ──────────────────────────────────────
A @ B                            # matrix multiply
np.linalg.inv(A)                 # inverse
np.linalg.det(A)                 # determinant
np.linalg.solve(A, b)            # solve Ax=b
np.linalg.eig(A)                 # eigenvalues
np.linalg.norm(a)                # vector/matrix norm

# ── RANDOM ──────────────────────────────────────
rng = np.random.default_rng(42)
rng.integers(0, 10, size=5)
rng.normal(0, 1, size=100)
rng.uniform(0, 1, size=100)
rng.choice(arr, size=5)
rng.shuffle(arr)                 # in-place
```

---

> 💡 **Tips for NumPy:**
> 1. Always check `.shape` and `.dtype` when debugging unexpected results
> 2. Prefer vectorized operations — avoid Python loops at all costs
> 3. Use `a.copy()` explicitly when you want independence from the original
> 4. Understand views vs copies — slicing creates views, fancy indexing creates copies
> 5. Use `np.random.default_rng(seed)` for reproducible randomness (modern API)
> 6. Use appropriate dtypes (`int32`, `float32`) for large arrays to save memory
> 7. Broadcasting is powerful but check shapes carefully to avoid silent bugs

---

*Made for Data Analysis learners. Covers NumPy 1.x and 2.x.*