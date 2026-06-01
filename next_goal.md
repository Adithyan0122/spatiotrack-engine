# Project 2 — High-Dimensional Latent Semantic & Embedding Engine

> **Goal:** Transition from geometric intuition (Project 1) to statistical and numerical reasoning in abstract high-dimensional spaces ($R^{1000+}$). Every concept is motivated by a retrieval system that must work at scale.

---

## Core Mathematical Mechanics

| Topic | What You Learn |
|---|---|
| SVD ($A = U\Sigma V^T$) | Implemented from scratch via bidiagonalization |
| PCA | Dimensionality reduction, variance preservation |
| Covariance matrices | Statistical structure, spectral theorem in action |
| Sparse matrix algebra | CSR/CSC formats, sparse-preserving operations |
| Cosine similarity | High-dimensional distance, inner product geometry |
| Frobenius norm | Matrix norm as a measurement tool |
| Condition numbers | Numerical rank, ill-conditioning diagnosis |
| Numerical stability | Householder bidiagonalization, shift strategies |

---

## Architecture

### Module 1 — Sparse Matrix Builder

**What it does:**
Constructs a massive document-term, item-feature, or token-co-occurrence matrix from raw relational data. This is your input to the entire pipeline.

**What you implement:**

**Term-document matrix:**
- Rows = documents, columns = terms
- Entry $A_{ij}$ = TF-IDF weight of term $j$ in document $i$:

$$\text{TF-IDF}(t, d) = \frac{f_{t,d}}{\sum_k f_{k,d}} \cdot \log\frac{N}{|\{d : t \in d\}|}$$

**Store as CSR (Compressed Sparse Row):**
- `data[]` — nonzero values
- `indices[]` — column index of each nonzero
- `indptr[]` — index into `data` where each row starts

Typical density: 0.1–1%. Storing as dense $10000 \times 50000$ float64 = 4GB. CSR = ~40MB.

**Implement sparse matrix-vector product manually:**
```
for i in range(n_rows):
    for j in range(indptr[i], indptr[i+1]):
        result[i] += data[j] * x[indices[j]]
```

Understanding this loop is understanding sparse computation.

**Condition number diagnostic:**
After building the matrix, estimate $\kappa(A)$ using the ratio of largest to smallest singular values:

$$\kappa(A) = \frac{\sigma_1}{\sigma_r}$$

where $r = \text{rank}(A)$. Deliberately construct an ill-conditioned version (add near-duplicate documents) and observe:
- What happens to singular value spectrum (many small $\sigma_i$ near zero)
- What happens to retrieval quality downstream

**Key insight:** Condition number is a diagnostic you run *before* decomposition. A $\kappa > 10^{10}$ tells you your data has redundancy that will pollute the embedding space.

---

### Module 2 — From-Scratch SVD via Bidiagonalization

**What it does:**
Implements SVD without calling any library decomposition. This is the mathematical core of the entire project.

**Why not just implement eigendecomposition of $A^TA$?**
$A^TA$ squares the condition number: $\kappa(A^TA) = \kappa(A)^2$. For $\kappa(A) = 10^6$, computing $A^TA$ introduces $10^{12}$ conditioning error. Bidiagonalization avoids forming $A^TA$ entirely.

**Phase 1 — Householder Bidiagonalization:**

Reduce $A \in R^{m \times n}$ to bidiagonal form $B$:

$$U^T A V = B = \begin{bmatrix} d_1 & f_1 & & \\ & d_2 & f_2 & \\ & & \ddots & f_{n-1} \\ & & & d_n \end{bmatrix}$$

Algorithm:
```
for k = 1 to n:
    # left Householder: zero out below A[k,k] in column k
    u = A[k:m, k]
    u[0] += sign(u[0]) * ||u||
    H_left = I - 2*u*u^T / (u^T*u)
    A = H_left * A   # updates A in-place, accumulate into U

    if k < n-1:
        # right Householder: zero out right of A[k,k+1] in row k
        v = A[k, k+1:n]
        v[0] += sign(v[0]) * ||v||
        H_right = I - 2*v*v^T / (v^T*v)
        A = A * H_right  # updates A in-place, accumulate into V
```

This directly reuses Householder reflectors from Project 1. The connection is intentional.

**Phase 2 — QR Iteration on Bidiagonal Matrix:**

Apply QR iteration with **Wilkinson shifts** to drive off-diagonal entries of $B$ to zero:

```
while not converged:
    choose shift mu (Wilkinson shift from bottom-right 2x2)
    apply implicit QR step to B (Givens rotations)
    check if any f_i < epsilon * (|d_i| + |d_{i+1}|)
    if yes: deflate (split problem, solve subproblems)
```

**Givens rotation** (used instead of Householder for bidiagonal — more efficient):

$$G(i,j,\theta) = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}$$

Applied to zero out a single off-diagonal entry while preserving bidiagonal structure.

**What debugging convergence failures teaches you:**
- Slow convergence = poor shift strategy — you feel why shifts matter
- NaN propagation = division by near-zero diagonal — you implement deflation as a fix
- Loss of orthogonality in $U$, $V$ — you add periodic reorthogonalization and understand why it's needed

You do not need production-grade performance. You need to implement it well enough to debug it.

**Verify your SVD:**
```
||A - U * diag(sigma) * V^T||_F / ||A||_F < 1e-10
||U^T*U - I||_F < 1e-10
||V^T*V - I||_F < 1e-10
```

---

### Module 3 — Truncated SVD with Principled Rank Selection

**What it does:**
Compresses your high-dimensional matrix into a dense low-rank latent space. Forces you to operationalize the spectral theorem: what does "significant structure" mean numerically?

**The rank-$k$ approximation:**

By Eckart-Young theorem, the best rank-$k$ approximation to $A$ in Frobenius norm is:

$$A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^k \sigma_i u_i v_i^T$$

where $U_k$, $\Sigma_k$, $V_k$ are the first $k$ columns/values of the full SVD.

**Implement two rank selection strategies:**

**Strategy 1 — Explained variance threshold:**

$$k^* = \min k \text{ such that } \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^n \sigma_i^2} \geq \tau$$

Typical $\tau = 0.95$. Plot the explained variance curve (scree plot) — the "elbow" is visible.

**Strategy 2 — Numerical rank:**

$$k^* = |\{i : \sigma_i / \sigma_1 > \epsilon_{machine} \cdot \max(m,n)\}|$$

where $\epsilon_{machine} \approx 2.2 \times 10^{-16}$ for float64. This is how numpy computes `matrix_rank` internally.

**Compare both strategies on your data:**
- They often agree on well-conditioned matrices
- They diverge on ill-conditioned matrices — Strategy 2 throws away more dimensions
- The gap between them is a measure of how much numerical noise is in your data

**What the spectral theorem says here:**
The covariance matrix $C = A^TA / (m-1)$ has eigendecomposition $C = V\Lambda V^T$ where $\Lambda = \Sigma^2 / (m-1)$. PCA principal components are exactly the right singular vectors $V$. SVD and PCA are the same computation — you now know *why*.

**Implement PCA framing explicitly:**
```
# center the data
A_centered = A - mean(A, axis=0)

# covariance matrix (never form it — use SVD of A_centered directly)
U, sigma, Vt = your_svd(A_centered)

# principal components = columns of V^T
# project data: Z = A_centered @ Vt[:k].T
```

Forming $A^TA$ explicitly to get eigenvalues is numerically inferior — SVD of $A$ directly is the right approach. You will have implemented both paths by this point.

---

### Module 4 — Sparse-Aware Projection

**What it does:**
Projects new query vectors into the latent space while maintaining sparsity as long as possible. Forces you to reason about matrix-vector products as algebra, not just as NumPy calls.

**The projection formula:**
Given a new query $q \in R^n$ (sparse), project to latent space:

$$\hat{q} = \Sigma_k^{-1} U_k^T q$$

Wait — this requires $U_k^T q$, a dense-times-sparse product. But $q$ is sparse (e.g., 0.1% nonzero).

**Implement sparse-dense matrix-vector product:**
```
# q is stored as (indices, values) — sparse format
# U_k is dense (m x k)

result = zeros(k)
for idx, val in zip(q.indices, q.values):
    result += val * U_k[idx, :]   # pick row idx of U_k, scale by val
```

This is $O(nnz(q) \cdot k)$ instead of $O(m \cdot k)$. For $nnz(q) = 50$, $m = 10000$, $k = 50$: 200x faster.

**When does densification become unavoidable?**
- After the first matrix multiply: $U_k^T q$ is always dense (size $k$)
- The sparsity lives in the *input space*, not the latent space
- Lesson: sparsity is a property of your data representation, not of the mathematical transformation

**Implement a query pipeline:**
```
precompute: V_k, Sigma_k from truncated SVD of A
query(q_sparse):
    q_latent = Sigma_k^{-1} * (U_k^T * q_sparse)  # sparse-aware
    return q_latent  # dense vector in R^k
```

---

### Module 5 — Dense Vector Retriever + Frobenius Drift Detector

**What it does:**
Builds a cosine similarity search engine over the latent space. Adds a matrix-norm-based drift detector that measures when the embedding space becomes stale.

**Index construction:**
Project all documents into latent space:

$$Z = A \cdot V_k \cdot \Sigma_k^{-1} \in R^{m \times k}$$

$Z$ is your embedding matrix. Each row is a document's latent representation.

**Cosine similarity search (pure matrix multiplication):**

Normalize all rows: $\hat{Z}_{i} = Z_i / \|Z_i\|$

Query:
```
q_latent_normalized = q_latent / ||q_latent||
scores = Z_normalized @ q_latent_normalized   # shape (m,)
top_k = argsort(scores)[-k:]
```

This is a single matrix-vector product. At scale (FAISS, Annoy, ScaNN) this is exactly what happens — now you understand the math.

**Why cosine, not Euclidean?**
In high dimensions, Euclidean distance concentrates — all points become approximately equidistant (curse of dimensionality). Cosine similarity measures *angle*, not magnitude, making it stable in high-dimensional spaces. Verify this empirically: plot the distribution of Euclidean distances vs cosine similarities for 1000 random pairs.

**Frobenius Norm Drift Detector:**

As new documents arrive, periodically recompute a fresh embedding $Z_{new}$ and measure:

$$\text{drift} = \frac{\|Z_{new} - Z_{old}\|_F}{\|Z_{old}\|_F}$$

Threshold (e.g., drift > 0.05) triggers a full SVD recomputation.

**What this teaches:**
- Frobenius norm measures the "total energy" of a matrix: $\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\sum_i \sigma_i^2}$
- Relative drift is dimensionless — it works regardless of embedding size
- Matrix norms are *measurements*, not just definitions from a textbook

**Implement norm relationships explicitly:**
```
# Verify: Frobenius norm = sqrt(sum of squared singular values)
sigma = your_svd(A)[1]
frobenius_via_svd = sqrt(sum(sigma**2))
frobenius_direct = sqrt(sum(A**2))
assert |frobenius_via_svd - frobenius_direct| < 1e-10
```

---

## What You Know After Project 2

- You can implement SVD from scratch and understand why bidiagonalization is numerically superior to eigendecomposition of $A^TA$
- You understand PCA as a special case of SVD and can implement both framings
- You can select rank principally (variance threshold vs numerical rank) and articulate why they differ
- You understand sparse matrix algebra at the operation level, not just the data structure level
- You can build a cosine similarity search engine from first principles
- You understand why cosine similarity is preferred over Euclidean distance in high dimensions
- You can use the Frobenius norm as an operational measurement tool
- Condition numbers and numerical stability are now part of your diagnostic vocabulary

---

## The Bridge from Project 1 to Project 2

| Project 1 | Project 2 | Connection |
|---|---|---|
| Householder QR | Householder bidiagonalization | Same reflectors, different target structure |
| Normal equations ($A^TA\hat{x} = A^Tb$) | Covariance matrix ($A^TA$) | Same matrix, different interpretation |
| Power Iteration (1 eigenvector) | SVD (all singular vectors) | SVD generalizes power iteration |
| Condition number as a warning | Condition number as a diagnostic | Deepened usage |
| LU/QR solve ($R^n$) | Projection into latent space ($R^k$) | Same mechanics, higher dimension |

---

## Recommended Implementation Order

1. Module 1 (Sparse Matrix Builder) — establishes the data and sparsity concepts
2. Module 2 (From-Scratch SVD) — the mathematical core; do this before anything else that uses SVD
3. Module 3 (Truncated SVD + Rank Selection) — direct extension of Module 2
4. Module 4 (Sparse-Aware Projection) — requires Module 3 output
5. Module 5 (Retriever + Drift Detector) — final integration, requires Modules 3 and 4

---

## Reference Formulas

| Operation | Formula |
|---|---|
| SVD | $A = U\Sigma V^T$ |
| Best rank-$k$ approx | $A_k = U_k\Sigma_k V_k^T$ |
| TF-IDF | $\text{tf}(t,d) \cdot \log(N / df(t))$ |
| Explained variance | $\sum_{i=1}^k \sigma_i^2 / \sum_{i=1}^n \sigma_i^2$ |
| Numerical rank threshold | $\sigma_i / \sigma_1 > \epsilon \cdot \max(m,n)$ |
| Cosine similarity | $\cos(\theta) = u^Tv / (\|u\|\|v\|)$ |
| Frobenius norm | $\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2} = \sqrt{\sum_i \sigma_i^2}$ |
| Frobenius drift | $\|Z_{new} - Z_{old}\|_F / \|Z_{old}\|_F$ |
| Query projection | $\hat{q} = \Sigma_k^{-1} U_k^T q$ |
| Householder reflector | $H = I - 2vv^T / v^Tv$ |
| Condition number | $\kappa(A) = \sigma_{max} / \sigma_{min}$ |
