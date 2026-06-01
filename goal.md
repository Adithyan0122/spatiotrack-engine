# Project 1 — Spatiotemporal Tracking & Network Analytics Engine

> **Goal:** Build geometric and numerical intuition for linear algebra in low-dimensional concrete spaces ($R^2$ to $R^n$). Every concept is motivated by a real computation you need to perform.

---

## Core Mathematical Mechanics

| Topic | What You Learn |
|---|---|
| Affine & linear transformations | Change of basis, coordinate frames |
| Normal Equations ($A^TA\hat{x} = A^Tb$) | Overdetermined systems, least squares |
| QR Factorization | Gram-Schmidt + Householder, stability comparison |
| LU Decomposition | Partial pivoting, direct solve |
| Null space, rank, column space | Fundamental subspaces, rank deficiency |
| Eigenvalues & Eigenvectors | Spectral structure, dominant directions |
| Power Iteration | Practical eigenvalue algorithm, Markov chains |
| Condition Numbers | Numerical stability of transformations |

---

## Architecture

### Module 1 — Ingestion & Transformation Pipeline

**What it does:**
Ingests continuous 2D or 3D coordinate trajectories (moving objects or agents on a field). Applies transformation matrices to normalize tracking data relative to different reference frames.

**What you implement:**
- Represent each coordinate frame as a basis matrix $B$
- Transform between frames using $B^{-1}$ and $B$ (change of basis)
- Affine transformations: $T(x) = Ax + b$ for rotation, scaling, translation
- Compose multiple transformations: $T_3 \circ T_2 \circ T_1$

**Linear algebra in action:**
- Change of basis: if $v$ is expressed in frame $A$, convert to frame $B$ via $[v]_B = P^{-1}[v]_A$ where $P$ is the change-of-basis matrix
- Affine maps as homogeneous matrices (embed $R^n$ into $R^{n+1}$):

$$\begin{bmatrix} A & b \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ 1 \end{bmatrix} = \begin{bmatrix} Ax + b \\ 1 \end{bmatrix}$$

**Numerical stability check:**
After each transformation, compute the condition number:

$$\kappa(A) = \|A\| \cdot \|A^{-1}\|$$

Flag frames where $\kappa(A) > 10^6$ as numerically unstable. Observe what happens to coordinates when you apply a poorly conditioned transformation — errors amplify by a factor of $\kappa(A)$.

**Key insight:** You will feel concretely why "inverting a matrix" is dangerous. A camera calibration matrix with $\kappa = 10^8$ means a 1-pixel measurement error becomes a 100-meter position error.

---

### Module 2 — Trajectory Optimization (OLS Solver)

**What it does:**
Fits predictive polynomial curves over noisy spatial data points to forecast future positions.

**What you implement:**
Given $m$ noisy observations of position $(t_i, y_i)$, fit a degree-$d$ polynomial:

$$y = c_0 + c_1 t + c_2 t^2 + \cdots + c_d t^d$$

This is an overdetermined system $Ax = b$ where $A$ is the Vandermonde matrix:

$$A = \begin{bmatrix} 1 & t_1 & t_1^2 & \cdots & t_1^d \\ 1 & t_2 & t_2^2 & \cdots & t_2^d \\ \vdots & & & & \vdots \\ 1 & t_m & t_m^2 & \cdots & t_m^d \end{bmatrix}$$

**Solve via Normal Equations:**

$$A^TA\hat{x} = A^Tb$$

Understand *why* this works: minimizing $\|Ax - b\|^2$ requires $A^T(Ax - b) = 0$, meaning the residual is orthogonal to the column space of $A$.

**Solve via QR (Gram-Schmidt):**
Decompose $A = QR$ where $Q$ has orthonormal columns and $R$ is upper triangular. Then:

$$A\hat{x} = b \Rightarrow QR\hat{x} = b \Rightarrow R\hat{x} = Q^Tb$$

Solve $R\hat{x} = Q^Tb$ by back-substitution. No matrix inverse needed.

**Implement Gram-Schmidt:**
```
for j = 1 to n:
    v_j = a_j
    for i = 1 to j-1:
        v_j = v_j - proj(v_j onto q_i)
    q_j = v_j / ||v_j||
```

**Then implement Householder reflections** as a second QR implementation:
- Householder reflector: $H = I - 2vv^T$ where $v = u / \|u\|$, $u = x - \|x\|e_1$
- Apply sequence of Householder reflectors to zero out subdiagonals

**Critical experiment:** Run both QR implementations on a nearly-singular Vandermonde matrix (high-degree polynomial, closely spaced $t_i$ values). Gram-Schmidt will visibly lose orthogonality. Householder will stay stable. Check with:

$$\|Q^TQ - I\|_F$$

This is the moment numerical stability stops being theoretical.

---

### Module 3 — Null Space Explorer

**What it does:**
For trajectory segments with fewer observations than unknowns (underdetermined system), explicitly compute and visualize the null space to understand what motion is unobservable.

**What you implement:**
Given $Ax = 0$ with $A \in R^{m \times n}$, $m < n$:
- Compute $\text{null}(A)$ via QR decomposition: columns of $Q$ corresponding to zero (or near-zero) diagonal entries of $R$
- Or via SVD: right singular vectors corresponding to zero singular values

**Key subspace relationships (Four Fundamental Subspaces):**

$$R^n = \text{col}(A^T) \oplus \text{null}(A)$$
$$R^m = \text{col}(A) \oplus \text{null}(A^T)$$

**Rank-nullity theorem (make it concrete):**

$$\text{rank}(A) + \text{nullity}(A) = n$$

**What this means physically:** If your trajectory has 3 degrees of freedom but only 2 observations, the null space is 1-dimensional. That one direction is the motion your sensor cannot detect. Visualize it as an arrow — that is the blind spot of your measurement system.

**Implement:**
- `compute_rank(A, tol)` — count singular values above tolerance
- `compute_null_basis(A)` — return orthonormal basis for null space
- Visualize null space basis vectors as arrows on the 2D/3D tracking field

---

### Module 4 — LU Solver

**What it does:**
Implements LU decomposition with partial pivoting as an alternative solve path. You compare it against QR to understand when each is appropriate.

**What you implement:**
Decompose $PA = LU$ where:
- $P$ is a permutation matrix (partial pivoting)
- $L$ is unit lower triangular
- $U$ is upper triangular

**Algorithm:**
```
for k = 1 to n-1:
    find pivot: row p = argmax |A[k:n, k]|
    swap rows k and p in A (and track in P)
    for i = k+1 to n:
        A[i,k] = A[i,k] / A[k,k]          # multiplier -> L
        A[i,k+1:n] -= A[i,k] * A[k,k+1:n] # row elimination -> U
```

**Solve $Ax = b$:**
1. Apply $P$: $Pb$
2. Forward substitution: solve $Ly = Pb$
3. Back substitution: solve $Ux = y$

**Critical experiment:** Solve the same trajectory optimization problem with both your QR solver and your LU solver. Compute the residual $\|Ax - b\|$ for both. Then:
- Try an ill-conditioned input (nearly repeated rows in $A$)
- Try a matrix with a zero on the diagonal before pivoting (LU without pivoting fails; with pivoting succeeds)

**Key insight:** LU is faster for square systems ($O(n^3/3)$ vs $O(2n^3/3)$ for QR) but QR is more stable for overdetermined systems. Now you know *when to use which* — not just that they exist.

---

### Module 5 — Influence Topology Graph

**What it does:**
Converts spatial interactions (proximity, passing networks) into an adjacency matrix and computes structural importance across the network.

**What you implement:**

**Build adjacency matrix:**
- Define interaction: agents within distance $d$ at time $t$ are "connected"
- $W_{ij}$ = number of interactions between agent $i$ and agent $j$
- Normalize to row-stochastic matrix: $P_{ij} = W_{ij} / \sum_k W_{ik}$

**Power Iteration from scratch:**
```
x_0 = uniform vector (1/n, ..., 1/n)
for k = 1 to max_iter:
    x_{k+1} = P^T x_k
    x_{k+1} = x_{k+1} / ||x_{k+1}||
    if ||x_{k+1} - x_k|| < tol: break
```

This converges to the **dominant eigenvector** of $P^T$ — the PageRank / stationary distribution.

**Connect back to eigenstructure:**
- The stationary distribution $\pi$ satisfies $P^T\pi = \pi$, i.e., $\pi$ is the eigenvector of $P^T$ with eigenvalue 1
- Convergence rate of Power Iteration depends on the ratio $|\lambda_2| / |\lambda_1|$ — the **spectral gap**
- A small spectral gap means slow mixing: parts of your network are weakly connected

**Implement eigenvalue estimation:**
After Power Iteration converges to $x^*$, estimate eigenvalue via Rayleigh quotient:

$$\lambda \approx \frac{x^{*T}Ax^*}{x^{*T}x^*}$$

**Extend to $k$ largest eigenvectors** using deflation: subtract the contribution of the found eigenvector and repeat. This gives you the top-$k$ most influential directions in the network.

---

## What You Know After Project 1

- You can construct, apply, and invert linear transformations and understand what condition number tells you about stability
- You can solve overdetermined systems two ways (QR, Normal Equations) and know why QR is more stable
- You can solve square systems via LU with pivoting and compare it to QR
- You understand the four fundamental subspaces and can compute null spaces explicitly
- You can implement Power Iteration and connect it to the eigenvalue problem
- You have *felt* numerical instability — not just read about it

---

## Recommended Implementation Order

1. Module 1 (Transformations) — establishes the data pipeline and matrix operations
2. Module 2 (OLS/QR) — core solver, implement Gram-Schmidt first then Householder
3. Module 4 (LU) — now that you have QR, compare against LU
4. Module 3 (Null Space) — requires QR or SVD, do after Module 2
5. Module 5 (Graph/Power Iteration) — standalone, do last

---

## Reference Formulas

| Operation | Formula |
|---|---|
| Change of basis | $[v]_B = P^{-1}[v]_A$ |
| Normal equations | $A^TA\hat{x} = A^Tb$ |
| QR solve | $R\hat{x} = Q^Tb$ |
| Condition number | $\kappa(A) = \sigma_{max} / \sigma_{min}$ |
| Householder reflector | $H = I - 2vv^T / v^Tv$ |
| Rank-nullity | $\text{rank}(A) + \text{nullity}(A) = n$ |
| Rayleigh quotient | $\lambda \approx x^TAx / x^Tx$ |
| Power Iteration | $x_{k+1} = Ax_k / \|Ax_k\|$ |
