#!/usr/bin/env python3
"""
Linear Algebra Explorer: Independence, Basis, and Dimension
=============================================================

Educational Python script with visualizations to teach core linear algebra concepts.
Best experienced in a Jupyter Notebook or Google Colab environment.

CONCEPTS TAUGHT:
----------------
1. Linear Independence vs Dependence
   - Definition, geometric meaning, how to check (rank, determinant, nontrivial relations)

2. Span of a set of vectors
   - Visualized by linear combinations (lattice points + parallelogram)

3. Basis of a Vector Space
   - Independent + Spanning set
   - Standard vs non-standard bases
   - Change of coordinates / change of basis visualization

4. Dimension
   - Number of vectors in a basis (same for all bases)
   - Cannot have more linearly independent vectors than the dimension
   - Subspaces have their own dimension (line=1, plane=2, space=3)

REQUIREMENTS:
-------------
pip install numpy matplotlib ipywidgets

RUN INSTRUCTIONS:
-----------------
- Jupyter Notebook / JupyterLab (recommended): Paste into a cell and run.
- Google Colab: Paste into a cell. It works with ipywidgets.
- Plain Python: The static demos will work; interactive part requires Jupyter.

Author: Grok (xAI) - Educational example
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon, FancyArrowPatch
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import warnings
warnings.filterwarnings('ignore')

# Nice matplotlib defaults
plt.rcParams['figure.figsize'] = (9, 7)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_linear_independence(vectors):
    """Check if a list of vectors is linearly independent using matrix rank."""
    if len(vectors) == 0:
        return True, 0, 0
    mat = np.column_stack([np.asarray(v, dtype=float) for v in vectors])
    rank = np.linalg.matrix_rank(mat, tol=1e-8)
    num_vecs = len(vectors)
    is_independent = (rank == num_vecs)
    return is_independent, rank, num_vecs


def find_dependence_relation(vectors):
    """
    Find a nontrivial linear dependence relation c1*v1 + c2*v2 + ... = 0
    using SVD (null space). Returns normalized coefficient vector or None.
    """
    if len(vectors) < 2:
        return None
    mat = np.column_stack([np.asarray(v, dtype=float) for v in vectors])
    rank = np.linalg.matrix_rank(mat, tol=1e-8)
    if rank == len(vectors):
        return None  # Independent
    
    # SVD to get approximate null space
    U, S, Vt = np.linalg.svd(mat, full_matrices=True)
    # Right singular vectors corresponding to \~0 singular values
    null_dim = Vt.shape[0] - rank
    if null_dim > 0:
        null_vec = Vt[-1, :]  # Last row of Vt
        norm = np.linalg.norm(null_vec)
        if norm > 1e-8:
            null_vec = null_vec / norm
            # Round for nicer display
            null_vec = np.round(null_vec, decimals=6)
            return null_vec
    return None


def plot_vectors_2d(vectors, title="Vectors in R²", ax=None, 
                    colors=None, labels=None, show_combinations=True):
    """
    Beautiful 2D visualization of vectors + their span via linear combinations.
    - Scatter points = many linear combinations (shows the span geometrically)
    - Green parallelogram = fundamental parallelogram spanned by the vectors
    - When vectors are dependent, everything collapses to a line (great visual!)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    ax.set_aspect('equal')
    ax.axhline(y=0, color='black', linewidth=0.6, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.6, alpha=0.7)
    
    # Dynamic limits based on vectors
    all_vecs = np.array(vectors)
    max_val = max(6, np.max(np.abs(all_vecs)) * 1.4)
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    if colors is None:
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'][:len(vectors)]
    if labels is None:
        labels = [f'v{i+1}' for i in range(len(vectors))]
    
    origin = np.zeros(2)
    
    # Draw vectors as arrows
    for i, v in enumerate(vectors):
        v = np.asarray(v)
        ax.annotate('', xy=v, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=2.8,
                                   connectionstyle='arc3,rad=0'))
        # Label slightly offset
        offset = v * 0.12
        ax.text(v[0] + offset[0], v[1] + offset[1], labels[i], 
                fontsize=12, fontweight='bold', color=colors[i])
    
    # Show span via linear combinations (the key visualization!)
    if show_combinations and len(vectors) >= 1:
        if len(vectors) == 2:
            v1, v2 = np.asarray(vectors[0]), np.asarray(vectors[1])
            
            # Grid of coefficients (integer-ish for clean lattice look)
            coeffs = np.linspace(-2.5, 2.5, 11)
            points = []
            for c1 in coeffs:
                for c2 in coeffs:
                    p = c1 * v1 + c2 * v2
                    points.append(p)
            points = np.array(points)
            
            # Color by "distance" from origin to show structure
            dists = np.linalg.norm(points, axis=1)
            ax.scatter(points[:, 0], points[:, 1], 
                      c=dists, cmap='plasma', alpha=0.35, s=18, 
                      label='Linear combinations (the span)')
            
            # Fundamental parallelogram (0, v1, v1+v2, v2)
            para_vertices = np.array([origin, v1, v1 + v2, v2])
            poly = Polygon(para_vertices, closed=True, 
                          alpha=0.18, facecolor='#27ae60', edgecolor='#27ae60', lw=1.5)
            ax.add_patch(poly)
            
            # Add small annotation
            ax.text(0.98, 0.02, "Green region = parallelogram\nPurple dots = span fills plane\n(when independent)",
                    transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
            
        elif len(vectors) == 1:
            v = np.asarray(vectors[0])
            t = np.linspace(-4, 4, 200)
            line_pts = t[:, np.newaxis] * v
            ax.plot(line_pts[:, 0], line_pts[:, 1], color='#9b59b6', 
                   linewidth=3.5, alpha=0.6, label='Span = 1D line')
            ax.text(0.02, 0.98, "Only 1D span (a line through origin)",
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='#ffeaa7', alpha=0.9))
    
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    return ax


def plot_change_of_basis(b1, b2, w, title="Change of Basis in R²"):
    """Visualize expressing a vector w in a new basis {b1, b2}."""
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_aspect('equal')
    
    max_val = max(7, np.max(np.abs([b1, b2, w])) * 1.3)
    ax.set_xlim(-1, max_val)
    ax.set_ylim(-1, max_val)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.6)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.6)
    
    # Faint standard basis (old coordinate system)
    ax.annotate('', xy=(max_val-0.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='lightgray', lw=1.2))
    ax.annotate('', xy=(0, max_val-0.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='lightgray', lw=1.2))
    ax.text(max_val-0.3, 0.3, 'x (standard)', color='gray', fontsize=10)
    ax.text(0.3, max_val-0.3, 'y (standard)', color='gray', fontsize=10)
    
    # New basis vectors (colored)
    ax.annotate('', xy=b1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))
    ax.text(b1[0]*1.08, b1[1]*1.08, 'b₁ (new x\')', color='#e74c3c', fontsize=12, fontweight='bold')
    
    ax.annotate('', xy=b2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=3))
    ax.text(b2[0]*1.08, b2[1]*1.08, 'b₂ (new y\')', color='#3498db', fontsize=12, fontweight='bold')
    
    # The vector w we want to express
    ax.annotate('', xy=w, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))
    ax.text(w[0]*1.05, w[1]*1.05, 'w  (standard coords)', color='#27ae60', fontsize=11, fontweight='bold')
    
    # Compute new coordinates
    B = np.column_stack([b1, b2])
    c = np.linalg.solve(B, w)
    
    # Show decomposition: c1*b1 and c2*b2
    scaled_b1 = c[0] * b1
    scaled_b2 = c[1] * b2
    
    # Dashed arrow for first component
    ax.annotate('', xy=scaled_b1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.8, linestyle='--'))
    mid1 = scaled_b1 / 2
    ax.text(mid1[0] - 0.4, mid1[1] + 0.3, f'{c[0]:.2f} × b₁', 
            color='#e74c3c', fontsize=10, fontweight='bold')
    
    # Dashed arrow for second component (from end of first)
    ax.annotate('', xy=w, xytext=scaled_b1,
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.8, linestyle='--'))
    mid2 = (scaled_b1 + w) / 2
    ax.text(mid2[0] + 0.2, mid2[1] - 0.4, f'{c[1]:.2f} × b₂', 
            color='#3498db', fontsize=10, fontweight='bold')
    
    # Small parallelogram hint (the "new coordinates grid cell")
    origin = np.zeros(2)
    para = np.array([origin, scaled_b1, w, scaled_b2])
    poly = Polygon(para, closed=True, alpha=0.12, facecolor='#9b59b6', edgecolor='#9b59b6', lw=1)
    ax.add_patch(poly)
    
    # Info box
    info_text = (f"w = {w}  (standard coordinates)\n\n"
                 f"In NEW basis {{b₁, b₂}}:\n"
                 f"w = {c[0]:.3f}·b₁ + {c[1]:.3f}·b₂\n\n"
                 f"New coordinates of w:  c = ({c[0]:.2f}, {c[1]:.2f})")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', 
                     edgecolor='#3498db', alpha=0.95))
    
    ax.set_title(title + "\n(How coordinates change when we pick a different basis)", 
                 fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig, ax, c


# ============================================================
# STATIC DEMOS (run these first to learn the concepts)
# ============================================================

def demo_linear_independence():
    """Demonstrate linear independence vs dependence with plots."""
    print("\n" + "="*70)
    print("📐 CONCEPT 1: LINEAR INDEPENDENCE vs DEPENDENCE")
    print("="*70)
    print("""
DEFINITION:
A set of vectors {v₁, v₂, ..., vₖ} is LINEARLY INDEPENDENT if the ONLY solution to
          c₁v₁ + c₂v₂ + ... + cₖvₖ = 0
is the trivial solution: c₁ = c₂ = ... = cₖ = 0.

If there exists a NONTRIVIAL solution (some cᵢ ≠ 0), the vectors are 
LINEARLY DEPENDENT. One vector can be written as a linear combination of others.

GEOMETRIC VIEW (R²):
- Independent → vectors point in "different directions" → they span the whole plane.
- Dependent   → vectors lie on the same line through origin → span only that line.
""")
    
    # Example 1: Independent
    v1 = np.array([2.0, 1.0])
    v2 = np.array([1.0, 3.0])
    is_indep, rank, n = check_linear_independence([v1, v2])
    det = np.linalg.det(np.column_stack([v1, v2]))
    
    print(f"\n✅ EXAMPLE 1 — INDEPENDENT VECTORS")
    print(f"   v₁ = {v1},   v₂ = {v2}")
    print(f"   Determinant = {det:.2f} (non-zero ⇒ independent)")
    print(f"   Rank = {rank} out of {n} vectors → INDEPENDENT")
    print(f"   They FORM A BASIS for R². The span is the entire 2D plane (dimension = 2).")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_vectors_2d([v1, v2], 
                    title="✅ LINEARLY INDEPENDENT\nThey span the whole plane (R²)\nDeterminant ≠ 0 → Basis for R²",
                    ax=axes[0], colors=['#e74c3c', '#3498db'])
    
    # Example 2: Dependent
    v3 = np.array([2.0, 1.0])
    v4 = np.array([4.0, 2.0])
    is_indep2, rank2, n2 = check_linear_independence([v3, v4])
    rel = find_dependence_relation([v3, v4])
    
    print(f"\n❌ EXAMPLE 2 — DEPENDENT VECTORS")
    print(f"   v₃ = {v3},   v₄ = {v4}")
    print(f"   Rank = {rank2} out of {n2} → DEPENDENT")
    if rel is not None:
        print(f"   Nontrivial relation: {rel[0]:.2f}·v₃ + {rel[1]:.2f}·v₄ ≈ 0")
        print(f"   (Indeed v₄ = 2·v₃, so they point in exactly the same direction)")
    print(f"   Span = only a 1-dimensional line. They do NOT form a basis for R².")
    
    plot_vectors_2d([v3, v4],
                    title="❌ LINEARLY DEPENDENT\nThey only span a 1D line\n(everything collapses visually)",
                    ax=axes[1], colors=['#e74c3c', '#9b59b6'])
    
    plt.suptitle("Linear Independence: The Foundation of Basis & Dimension", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("\n💡 KEY INSIGHT: Independence is about whether vectors 'add new directions'.")
    print("   The purple dots show all possible linear combinations. When independent,")
    print("   they fill the whole plane. When dependent, they stay on a line.")


def demo_basis_and_dimension():
    """Explain basis and dimension with examples."""
    print("\n" + "="*70)
    print("📐 CONCEPT 2 & 3: BASIS AND DIMENSION")
    print("="*70)
    print("""
BASIS:
A set B = {b₁, b₂, ..., bₙ} is a BASIS for a vector space V if:
   1. B is linearly independent, AND
   2. B spans V (every vector in V can be written as linear combo of vectors in B)

DIMENSION:
The DIMENSION of V is the number of vectors in ANY basis of V.
It is a fundamental invariant — all bases have the SAME size.

For Rⁿ (n-dimensional Euclidean space):
- Dimension = n
- Any set of n linearly independent vectors forms a basis.
- You cannot find more than n linearly independent vectors in Rⁿ.
- The standard basis is {e₁=(1,0,...,0), e₂=(0,1,0,...,0), ..., eₙ=(0,...,1)}
""")
    
    print("\n📌 EXAMPLE: Non-standard basis for R²")
    b1 = np.array([1.0, 2.0])
    b2 = np.array([2.0, 1.0])
    is_b, r_b, _ = check_linear_independence([b1, b2])
    print(f"   b₁ = {b1},  b₂ = {b2}")
    print(f"   Rank = {r_b} → Independent → This IS a valid basis for R² (even though not orthogonal or unit length)")
    
    # Show a vector in both coordinate systems
    w = np.array([4.0, 5.0])
    B_mat = np.column_stack([b1, b2])
    coords_new = np.linalg.solve(B_mat, w)
    print(f"\n   Take vector w = {w} (in standard coordinates)")
    print(f"   In the NEW basis {{b₁, b₂}}, w has coordinates c = {coords_new}")
    print(f"   Check: {coords_new[0]:.2f}·b₁ + {coords_new[1]:.2f}·b₂ = {B_mat @ coords_new}")
    
    fig, ax, c = plot_change_of_basis(b1, b2, w)
    plt.show()
    
    print("\n📌 WHY DIMENSION MATTERS")
    print("   In R² you need exactly 2 independent vectors for a basis.")
    print("   If you take 3 vectors in R², they MUST be linearly dependent (rank ≤ 2).")
    print("   This is true in any dimension: max number of independent vectors = dimension.")


def demo_3d_and_higher_dimension():
    """Quick 3D visualization and higher dimension intuition."""
    print("\n" + "="*70)
    print("📐 CONCEPT 4: DIMENSION IN R³ AND HIGHER")
    print("="*70)
    print("""
In R³:
- Dimension = 3
- A basis needs exactly 3 linearly independent vectors.
- 4 or more vectors in R³ are always linearly dependent.
- The span of 1 vector = line (dim 1 subspace)
- The span of 2 independent vectors = plane (dim 2 subspace)
- The span of 3 independent vectors = whole space R³ (dim 3)
""")
    
    # Standard basis in 3D
    e1 = np.array([1., 0., 0.])
    e2 = np.array([0., 1., 0.])
    e3 = np.array([0., 0., 1.])
    
    print("\n✅ Standard basis for R³ (the unit vectors along the axes)")
    print(f"   e₁ = {e1}, e₂ = {e2}, e₃ = {e3}")
    is_3d, rank_3d, _ = check_linear_independence([e1, e2, e3])
    print(f"   Rank = {rank_3d} → Independent → Forms a basis. Dimension = 3")
    
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_zlim(-0.5, 2.5)
    
    # Plot the three basis vectors
    ax.quiver(0,0,0, e1[0],e1[1],e1[2], color='#e74c3c', linewidth=3, arrow_length_ratio=0.15)
    ax.quiver(0,0,0, e2[0],e2[1],e2[2], color='#3498db', linewidth=3, arrow_length_ratio=0.15)
    ax.quiver(0,0,0, e3[0],e3[1],e3[2], color='#27ae60', linewidth=3, arrow_length_ratio=0.15)
    
    ax.text(e1[0]+0.1, e1[1], e1[2], 'e₁', color='#e74c3c', fontsize=12, fontweight='bold')
    ax.text(e2[0], e2[1]+0.1, e2[2], 'e₂', color='#3498db', fontsize=12, fontweight='bold')
    ax.text(e3[0], e3[1], e3[2]+0.1, 'e₃', color='#27ae60', fontsize=12, fontweight='bold')
    
    # Draw a little cube to show the volume they span
    ax.plot([0,1,1,0,0], [0,0,1,1,0], [0,0,0,0,0], 'k--', alpha=0.4)  # bottom
    ax.plot([0,1,1,0,0], [0,0,1,1,0], [1,1,1,1,1], 'k--', alpha=0.4)  # top
    ax.plot([0,0], [0,0], [0,1], 'k--', alpha=0.4)
    ax.plot([1,1], [0,0], [0,1], 'k--', alpha=0.4)
    ax.plot([1,1], [1,1], [0,1], 'k--', alpha=0.4)
    ax.plot([0,0], [1,1], [0,1], 'k--', alpha=0.4)
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title("Standard Basis of R³\nThree independent vectors span the whole 3D space\n(they form a unit cube of volume = 1)", 
                 fontsize=12, fontweight='bold')
    ax.view_init(elev=22, azim=45)
    plt.tight_layout()
    plt.show()
    
    print("\n💡 HIGHER DIMENSIONS")
    print("   The same ideas hold in R⁴, R⁵, ... even if we can't visualize them.")
    print("   A basis always has exactly 'dimension' number of vectors.")
    print("   Linear algebra lets us work in abstract high-dimensional spaces using")
    print("   the same rules (independence, span, coordinates w.r.t. a basis).")


# ============================================================
# INTERACTIVE EXPLORER (the highlight for active learning)
# ============================================================

def interactive_span_explorer():
    """
    Interactive widget to explore linear independence and span in real time.
    Change the vectors with sliders and instantly see:
    - Whether they are independent
    - The geometric span (purple dots fill plane vs collapse to line)
    - Rank and determinant
    - Nontrivial dependence relation (if any)
    """
    print("\n" + "="*70)
    print("🎮 INTERACTIVE EXPLORER: Play with vectors and see the concepts come alive!")
    print("="*70)
    print("""
Instructions:
• Drag the sliders to change v₁ and v₂ components.
• Watch the plot update instantly.
• When vectors are independent → purple dots fill the whole plane + green parallelogram.
• When dependent → everything collapses to a single line through the origin.
• The title and text box tell you the rank, determinant, and dependence relation.
• Try making them almost parallel (e.g. v1=(2,1), v2=(2.1,1)) to see near-dependence.
""")
    
    # Sliders
    v1x = widgets.FloatSlider(value=2.0, min=-5.0, max=5.0, step=0.1, description='v₁ x:', continuous_update=True)
    v1y = widgets.FloatSlider(value=1.0, min=-5.0, max=5.0, step=0.1, description='v₁ y:', continuous_update=True)
    v2x = widgets.FloatSlider(value=1.0, min=-5.0, max=5.0, step=0.1, description='v₂ x:', continuous_update=True)
    v2y = widgets.FloatSlider(value=3.0, min=-5.0, max=5.0, step=0.1, description='v₂ y:', continuous_update=True)
    
    output_plot = widgets.Output(layout=widgets.Layout(width='70%'))
    output_text = widgets.Output(layout=widgets.Layout(width='30%', border='1px solid #ddd'))
    
    def update(change=None):
        with output_plot:
            clear_output(wait=True)
            v1 = np.array([v1x.value, v1y.value])
            v2 = np.array([v2x.value, v2y.value])
            vectors = [v1, v2]
            
            is_indep, rank, n = check_linear_independence(vectors)
            det = np.linalg.det(np.column_stack(vectors))
            rel = find_dependence_relation(vectors)
            
            # Dynamic title
            if is_indep:
                title = (f"✅ INDEPENDENT  |  Rank = {rank}/{n}  |  det = {det:.2f}\n"
                         f"These two vectors FORM A BASIS for R²\n"
                         f"Span = entire 2D plane  (Dimension = 2)")
                main_color = '#27ae60'
            else:
                title = (f"❌ DEPENDENT  |  Rank = {rank}/{n}  |  det = {det:.2f}\n"
                         f"These vectors only span a 1D LINE  (not a basis for R²)")
                main_color = '#e74c3c'
            
            fig, ax = plt.subplots(figsize=(8.5, 7.5))
            plot_vectors_2d(vectors, title=title, ax=ax, 
                           colors=[main_color, '#8e44ad'])
            
            # Extra info on plot
            extra = f"v₁ = ({v1[0]:.1f}, {v1[1]:.1f})\nv₂ = ({v2[0]:.1f}, {v2[1]:.1f})"
            ax.text(0.02, 0.02, extra, transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
            plt.show()
        
        # Text panel with math explanation
        with output_text:
            clear_output(wait=True)
            if is_indep:
                html = f"""
                <div style="padding:10px; background:#e8f8f5; border-radius:8px;">
                <h4 style="color:#27ae60; margin-top:0;">✅ LINEARLY INDEPENDENT</h4>
                <p>The only solution to<br>
                <b>c₁v₁ + c₂v₂ = 0</b><br>
                is <b>c₁ = c₂ = 0</b>.</p>
                <p>They point in different directions and<br>
                <b>span the whole plane</b>.</p>
                <p><b>Dimension of span = 2</b></p>
                </div>
                """
            else:
                html = f"""
                <div style="padding:10px; background:#fdedec; border-radius:8px;">
                <h4 style="color:#e74c3c; margin-top:0;">❌ LINEARLY DEPENDENT</h4>
                <p>There exist nonzero c₁, c₂ such that<br>
                <b>c₁v₁ + c₂v₂ = 0</b></p>
                """
                if rel is not None:
                    html += f"""
                    <p><b>Dependence relation found:</b><br>
                    {rel[0]:.2f}·v₁ + {rel[1]:.2f}·v₂ ≈ 0</p>
                    """
                html += """
                <p>One vector is a scalar multiple of the other.<br>
                <b>Span = 1D line only</b><br>
                <b>Dimension of span = 1</b></p>
                </div>
                """
            display(HTML(html))
    
    # Wire up sliders
    for slider in [v1x, v1y, v2x, v2y]:
        slider.observe(update, names='value')
    
    # Quick example buttons
    def load_example(b):
        if b.description == "Independent":
            v1x.value, v1y.value = 2.0, 1.0
            v2x.value, v2y.value = 1.0, 3.0
        elif b.description == "Dependent":
            v1x.value, v1y.value = 2.0, 1.0
            v2x.value, v2y.value = 4.0, 2.0
        elif b.description == "Almost dependent":
            v1x.value, v1y.value = 2.0, 1.0
            v2x.value, v2y.value = 2.1, 1.05
        update()
    
    btn_ind = widgets.Button(description="Independent", button_style='success')
    btn_dep = widgets.Button(description="Dependent", button_style='danger')
    btn_near = widgets.Button(description="Almost dependent", button_style='warning')
    
    btn_ind.on_click(load_example)
    btn_dep.on_click(load_example)
    btn_near.on_click(load_example)
    
    button_box = widgets.HBox([btn_ind, btn_dep, btn_near])
    slider_box = widgets.VBox([v1x, v1y, v2x, v2y, button_box])
    main_box = widgets.HBox([slider_box, output_plot, output_text])
    
    display(main_box)
    
    # Initial plot
    update()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__" or True:  # Always run when imported or executed
    print(HTML("<h2 style='color:#2c3e50;'>🚀 Linear Algebra Explorer — Independence, Basis & Dimension</h2>").data)
    print("Run the functions below one by one (in Jupyter) to learn step by step.\n")
    
    # 1. Core concepts with static visualizations
    demo_linear_independence()
    
    # 2. Basis + change of coordinates
    demo_basis_and_dimension()
    
    # 3. 3D and higher dimensions
    demo_3d_and_higher_dimension()
    
    # 4. The interactive explorer (most fun part)
    print("\n" + "="*70)
    print("Now try the INTERACTIVE version below — move the sliders!")
    print("="*70)
    interactive_span_explorer()
    
    print("\n" + "="*70)
    print("✅ You have explored linear independence, span, basis, and dimension!")
    print("   Key takeaways:")
    print("   • Independence = vectors add new directions (rank = #vectors)")
    print("   • Basis = independent + spanning set")
    print("   • Dimension = size of basis (unique for the space)")
    print("   • In Rⁿ you need exactly n independent vectors for a basis.")
    print("="*70)