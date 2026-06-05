# Module 1: Soccer Coordinate Frames & Affine Transformations

**Project:** spatiotrack-engine  
**Theme:** Football (Soccer) Player & Ball Tracking  
**Focus:** Building geometric intuition for linear algebra through real football data problems

> **Module Goal (adapted from goal.md):**  
> Learn to ingest, normalize, and transform continuous 2D/3D trajectories of players and the ball on a soccer pitch. Every matrix operation is motivated by an actual problem that appears in modern football analytics (broadcast tracking, GPS/optical fusion, expected goals pipelines, physical performance analysis, offside technology, etc.).

---

## 1. Learning Objectives

By the end of this module you will be able to:

- Represent different soccer reference frames (camera, global pitch, attacking-direction normalized, player-local) as **basis matrices**.
- Perform **change of basis** to move player positions between frames.
- Build and apply **affine transformations** (rotation + scaling + translation) using homogeneous coordinates.
- **Compose** sequences of transformations (e.g. camera → pitch → attacking normalized → player local).
- Compute and interpret the **condition number** of a transformation matrix in the context of tracking accuracy.
- Detect numerically dangerous frames (e.g. a poorly calibrated sideline camera) before they ruin downstream analysis.
- Visualize everything on a realistic soccer pitch with proper scale (meters).

You will *feel* why a 1-pixel error in a camera view can become a 2-meter error on the pitch — and why that matters for offside calls, progressive passes, and physical metrics.

---

## 2. Why Transformations Matter in Modern Football

Real football analytics companies (Second Spectrum, Hawk-Eye, Opta, Stats Perform, SkillCorner, etc.) deal with this problem every single match:

- Multiple cameras around the stadium, each with its own distorted view.
- Optical tracking systems output in "stadium coordinates".
- GPS / LPS wearable data uses a completely different origin and orientation.
- For any model (xG, pressing intensity, physical load, pass probability), all data must live in one consistent, meter-based, direction-normalized coordinate system.
- For TV graphics or VAR, you often need the reverse: pitch coordinates → specific camera view.
- Player physical metrics are far more useful when expressed in the player's **own direction of movement** ("he carried the ball 12m forward and 3m laterally") rather than global x/y.

**Module 1 is the foundation** that makes all of that possible — and makes you understand the numerical dangers along the way.

---

## 3. The Soccer Pitch Coordinate System (Our "Global" Frame)

We will use the following **standard global pitch frame** (inspired by many tracking providers, with center origin for symmetry):

- **Origin (0, 0)**: Center of the pitch (center spot).
- **+x direction**: Towards the goal that the home team is attacking in the first half (we will handle direction normalization later).
- **+y direction**: To the left when facing the +x direction (right-handed coordinate system).
- **Units**: Meters.
- **Pitch dimensions** (FIFA standard):
  - Length (along x): 105 m → x ranges from **-52.5** to **+52.5**
  - Width (along y): 68 m → y ranges from **-34.0** to **+34.0**
- Penalty area: 16.5 m deep × 40.3 m wide (from the goal line).
- Goal: 7.32 m wide, centered.

**Alternative corner-based convention** (very common in event data like StatsBomb):
- (0, 0) at bottom-left corner from the perspective of the team attacking left-to-right.
- x: 0 → 105 (or 120 in some datasets), y: 0 → 68 (or 80).

**In this project we will support transforming between both conventions.**

You should implement a `SoccerPitch` class or module that knows the dimensions and can draw itself.

---

## 4. Common Reference Frames in Soccer Tracking

Here are the frames you will implement transformations for:

### 4.1 Camera / Pixel Frame (C)
- Origin usually at top-left of the video frame.
- Units: pixels (e.g. 1920×1080 or 1280×720).
- Axes: x right, y down (image convention) or y up.
- Almost never perfectly aligned with the pitch.
- For Module 1 we will approximate the mapping as **affine** (good enough for learning; real systems use full homographies for perspective).

**Example problem:** A high sideline camera tilted at ~15°. Its "x-axis" is not parallel to the touchline.

### 4.2 Global Pitch Frame (P) — the one defined above
This is our "truth" frame. All serious analysis eventually lives here.

### 4.3 Attacking-Direction Normalized Frame (A)
- Always attack **left → right** (negative x to positive x).
- When a team switches halves (or when analyzing the away team), you flip and/or rotate the data.
- This is critical for models that are direction-sensitive (e.g. "progressive carries", "passes into the box").

**Real use case:** A player makes a 25 m carry in the second half. In raw data it might be negative x. After normalization it becomes +25 m forward.

### 4.4 Player-Local Frame (L)
- Origin at the player's current position at time t.
- +x' points in the direction the player is facing / running (his "forward").
- +y' points to his left (or right — decide on a convention).
- Extremely powerful for:
  - "Forward distance covered"
  - "Lateral acceleration" (change of direction)
  - "Carries in behind the defense"

You get the player's orientation from velocity (smoothed) or from IMU data in wearables.

### 4.5 Other Useful Frames (stretch goals)
- Goal frame (origin at center of goal, x pointing out from goal)
- Half-pitch frame
- Broadcast "TV graphic" frame (sometimes slightly rotated for aesthetic reasons)

---

## 5. Mathematical Concepts — Explained Through Soccer

### 5.1 Basis = Coordinate Frame on the Pitch

A basis matrix **B** for a frame has its **columns** as the unit vectors of the new frame, expressed in the parent frame.

Example: A "tilted camera" frame whose x-axis is rotated 12° relative to the pitch sideline.

```python
theta = np.deg2rad(12)
b1 = np.array([np.cos(theta), np.sin(theta)])   # new x' direction in pitch coords
b2 = np.array([-np.sin(theta), np.cos(theta)])  # new y' direction (perpendicular)
B = np.column_stack([b1, b2])
```

This is exactly the same idea as the `plot_change_of_basis` function you already built in `independence_basis_dimension.py`.

### 5.2 Change of Basis

If a player is observed at position `p_cam` in the camera frame, his position in the global pitch frame is:

```python
p_pitch = B @ p_cam + origin_offset
```

Or, when going the other way (more common for normalization):

```python
p_cam = B_inv @ (p_pitch - origin)
```

The formula in goal.md:

> $[v]_B = P^{-1} [v]_A$

### 5.3 Affine Transformations (The Real Workhorse)

Most soccer transforms are **affine**:

- Rotate the view to align with the pitch
- Scale from pixels → meters
- Translate the origin from "top-left of camera" to "center of pitch"

We represent them uniformly with **homogeneous coordinates**.

For 2D:

A point $(x, y)$ becomes the 3-vector $\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$

An affine transform is the 3×3 matrix:

$$
T = \begin{bmatrix}
a_{11} & a_{12} & t_x \\
a_{21} & a_{22} & t_y \\
0 & 0 & 1
\end{bmatrix}
$$

Applying it:

```python
p_h = np.array([x, y, 1.0])
p_transformed = T @ p_h
```

You will implement helpers such as:

- `rotation(theta)`
- `translation(tx, ty)`
- `scaling(sx, sy)`
- `rigid_transform(theta, tx, ty)`  (rotation + translation)

### 5.4 Composition

If you have:

1. `T1` = pixels → undistorted camera plane
2. `T2` = camera plane → global pitch
3. `T3` = global pitch → attacking normalized

The full transform is:

```python
T_full = T3 @ T2 @ T1
```

**Important:** Matrix multiplication order is the reverse of the logical pipeline order when points are column vectors on the right (`T @ p`).

### 5.5 Condition Number — The Soccer Horror Metric

```python
kappa = np.linalg.cond(A)          # or compute via SVD yourself
# or
kappa = np.linalg.norm(A) * np.linalg.norm(np.linalg.inv(A))
```

**Interpretation in football:**

- κ ≈ 1–10 : Excellent. Tiny pixel noise stays tiny on the pitch.
- κ ≈ 100–1000 : Acceptable for many uses.
- κ > 10^4–10^5 : Getting dangerous.
- κ > 10^6 : **Flag it.** A 0.5 pixel error can become > 50 cm error. In tight offside situations this is match-changing.

**The classic example you must reproduce:**

A sideline camera whose image plane is almost parallel to the line of sight of the far touchline. The two "basis vectors" in the homography/affine fit are nearly linearly dependent → very small singular value → huge condition number.

Add realistic tracking noise (0.3–1.0 pixels) in the camera frame → see the position jump by 1–3 meters on the pitch after transformation. That is enough to turn a "clear onside" into "offside" or vice versa.

---

## 6. Implementation Roadmap (Soccer-Themed)

### Recommended File Structure

```
spatiotrack-engine/
├── docs/
│   └── module-01-soccer-transformations.md   # (this file)
├── spatiotrack/
│   ├── __init__.py
│   ├── pitch.py           # SoccerPitch class + draw_pitch()
│   ├── frames.py          # SoccerCoordinateFrame
│   ├── transforms.py      # AffineTransform, composition, factories
│   └── trajectories.py    # helpers for lists of (t, x, y) or pandas
├── examples/
│   ├── 01_camera_to_pitch.py
│   ├── 02_normalize_attacking_direction.py
│   ├── 03_player_local_frame.py
│   └── 04_horror_story_condition_number.py
└── independence_basis_dimension.py   # (existing foundation)
```

### Core Classes / Functions You Should Build

1. **SoccerPitch**
   - Constants: `LENGTH = 105.0`, `WIDTH = 68.0`, `GOAL_WIDTH = 7.32`, etc.
   - `draw_pitch(ax)` — beautiful matplotlib pitch with all lines, circles, penalty areas.
   - Methods to convert between center-origin and corner-origin conventions.

2. **AffineTransform** (or `SoccerTransform`)
   - Store the 3×3 homogeneous matrix internally.
   - `apply(points)` — works on (N, 2) or (N, 3) arrays, or single points.
   - `compose(other)` or `then(other)`
   - `inverse()` (carefully — only when well-conditioned)
   - `condition_number()` — must implement / expose this.
   - `is_stable(threshold=1e6)` 

3. **Factory functions** (in `transforms.py`)
   ```python
   rotation(theta_rad, degrees=False)
   translation(dx, dy)
   uniform_scale(s)
   anisotropic_scale(sx, sy)
   rigid_body(theta, tx, ty)          # rotation + translation
   from_basis(b1, b2, origin=np.zeros(2))
   ```

4. **SoccerCoordinateFrame**
   - `name`
   - `basis` (2×2 matrix whose columns are the axes)
   - `origin`
   - `to_global_transform()` → returns an `AffineTransform`
   - `from_global_transform()`

5. **Trajectory helpers**
   - Simple dataclass or just numpy arrays of shape `(N, 3)` for `(t, x, y)`
   - `transform_trajectory(traj, T)`
   - Add small Gaussian noise in a given frame.

### Suggested Implementation Order (for this soccer module)

1. Implement `SoccerPitch` + `draw_pitch()` (you will use this for every visualization).
2. Implement basic 2D affine factories + `apply` using homogeneous coords.
3. Implement `from_basis(...)` and change-of-basis between two frames.
4. Add `condition_number()` and `is_stable()`.
5. Build composition and a clean way to chain "camera → pitch → attacking".
6. Create synthetic soccer trajectories (winger run, central striker movement, curved pass).
7. Build the four main frames (Camera, Pitch, Attacking, PlayerLocal).
8. Create the "horror story" example that demonstrates error amplification.
9. Add roundtrip tests and visualization functions that extend your existing `plot_change_of_basis` style.

---

## 7. Concrete Soccer Exercises & Milestones

### Exercise 1: Basic Camera → Pitch
- Create a synthetic "camera" frame that is rotated 8° and scaled (pixels to meters roughly 0.05 m/px at that zoom) with a translation.
- Generate a straight-line run of a right winger: from (x=20, y=28) to (x=48, y=32) in pitch meters.
- Transform the run into the camera frame.
- Plot both on a drawn pitch (one as "ground truth", one as "what the camera saw").

### Exercise 2: Attacking Direction Normalization
- Take the same run above.
- Simulate second half data by reflecting it across the center (or rotating 180°).
- Apply the normalization transform so the carry is always shown as moving positive x.
- Verify distances are preserved.

### Exercise 3: Player Local Frame
- At a given timestamp, the player has position `p` and velocity `v = (vx, vy)`.
- Build a local frame where +x_local is the normalized velocity direction.
- Transform a short sequence of positions around that moment into the player's local frame.
- Compute "forward meters gained" vs "lateral meters".

### Exercise 4: Full Pipeline Composition
Build this chain and apply it to a full 5-second sequence of a player receiving the ball and driving forward:

`raw_camera_pixels → corrected_camera → global_pitch → attacking_normalized`

Print the condition number at each step.

### Exercise 5: The Condition Number Horror Story (Required)
1. Create a "bad" camera matrix with two nearly parallel basis vectors (angle < 2°).
2. Generate a realistic player trajectory on the pitch.
3. Add 0.5 pixel Gaussian noise in the camera frame.
4. Transform to pitch coordinates using both a good frame and the bad frame.
5. Plot the true path vs the noisy transformed path for both.
6. Quantify the maximum position error.
7. Print the condition numbers.
8. **Write a short paragraph** in comments or a notebook cell explaining what you observed and why this is dangerous in real football (mention offside, xG, physical metrics).

### Exercise 6: Roundtrips + Stability Checks
- For every transform you create, test `T.inverse().apply(T.apply(points)) ≈ points`
- Flag any frame with κ > 1e6 during pipeline construction and raise a warning or return a status.

---

## 8. Visualization Requirements

You should produce (at minimum) these kinds of plots, all using a proper soccer pitch background:

- Side-by-side: "Camera view" (abstract rectangle with distorted axes) vs "Pitch view" (real 105×68 pitch).
- Trajectories as lines with time-colored dots or arrows.
- Basis vectors drawn as thick colored arrows on the pitch (reuse/extend `plot_change_of_basis` logic).
- Error ellipses or multiple noisy realizations when demonstrating condition number effects.
- Before/after for attacking direction normalization (two full pitch plots).

Reuse as much as possible from `independence_basis_dimension.py` (the interactive spirit and clean matplotlib style are excellent).

---

## 9. Data Generation Helpers (You Should Implement)

```python
def generate_straight_run(start, end, n_points=50, noise=0.0, frame='pitch'):
    ...

def generate_curved_winger_run(touchline_y=30, length=40, n_points=60):
    """A realistic curved run down the right wing."""

def generate_ball_pass(from_pos, to_pos, height=0.0, n_points=20):
    ...
```

Add realistic timing (e.g. `t` from 0 to 4.2 seconds at 25 fps tracking rate).

---

## 10. Numerical & Testing Guidelines

- Use `float64`.
- For solving change-of-basis systems, prefer `np.linalg.solve(B, p)` over explicit inversion when possible.
- Implement condition number yourself at least once using SVD (`np.linalg.svd`) so you understand what it represents (`sigma_max / sigma_min`).
- Tolerance for roundtrips: `1e-9` or better for well-conditioned transforms.
- Always report condition numbers when printing transforms in examples.

---

## 11. How This Module Connects to the Rest of the Project

- **Module 2 (Trajectory Optimization)** will fit polynomials to runs in **consistent pitch coordinates**. Bad transforms from Module 1 will make your predictions meaningless.
- **Module 3 (Null Space)** will interpret "unobservable motion" — e.g. depth ambiguity from a single sideline camera.
- **Module 5 (Influence Graphs)** will build passing/proximity networks in normalized pitch space.
- The **existing `independence_basis_dimension.py`** already gave you the geometric intuition for bases and change of coordinates. Module 1 scales that intuition up to real trajectories + affine transforms + numerical danger.

---

## 12. Completion Checklist for Module 1 (Soccer Edition)

- [ ] `SoccerPitch` with correct dimensions and a nice `draw_pitch(ax)` function
- [ ] `AffineTransform` class using homogeneous coordinates internally
- [ ] Factory functions for rotation, translation, scaling, and basis-derived transforms
- [ ] Support for 2D (and optionally 3D with 4×4 matrices)
- [ ] Change of basis between at least 3–4 soccer frames
- [ ] Condition number computation + stability flagging (`> 1e6`)
- [ ] Composition of transforms
- [ ] At least 4 synthetic soccer trajectories (winger, central run, pass, etc.)
- [ ] Full "horror story" demo with quantitative error amplification
- [ ] Visualizations on real pitch backgrounds (not abstract plots)
- [ ] Roundtrip tests pass for all well-conditioned transforms
- [ ] Clear documentation / docstrings explaining the soccer meaning of each transform

---

## 13. Stretch Goals (After Basic Implementation)

- Support full 3D (ball z-coordinate, player height)
- Add a simple perspective distortion model (homography) and compare condition numbers
- Build an interactive widget (like your existing one) that lets you drag a camera tilt angle and instantly see condition number + error on a player trajectory
- Export transformed trajectories to a simple CSV that could be used by Module 2

---

## 14. Reference Formulas (Soccer Context)

| Concept                    | Formula / Operation                              | Soccer Meaning |
|---------------------------|--------------------------------------------------|--------------|
| Change of basis           | $[v]_B = P^{-1}[v]_A$                            | Convert player position from camera pixels to pitch meters |
| Affine transform          | $\begin{bmatrix}A & b\\0&1\end{bmatrix} \begin{bmatrix}x\\1\end{bmatrix}$ | Rotate + scale + move origin in one matrix |
| Composition               | $T_{full} = T_3 @ T_2 @ T_1$                     | Camera → Pitch → Attacking normalized |
| Condition number          | $\kappa = \sigma_{max}/\sigma_{min}$             | How much small camera noise gets amplified on the pitch |
| Stability threshold       | $\kappa > 10^6$                                  | Dangerous for fine decisions (offsides, carries) |

---

**Next action recommendation:**

Once you have read and understood this guide, the best way to proceed is:

1. Start by implementing `SoccerPitch` + `draw_pitch()` (you'll want this immediately for everything else).
2. Implement the basic `AffineTransform` machinery.
3. Re-use and extend the change-of-basis visualization code you already have.
4. Build the horror story last — it will be the most satisfying (and scary) part.

This module is deliberately front-loaded with geometry and numerical awareness because **everything** in the later modules depends on having trustworthy coordinates.

Good luck. When you're ready to code, tell me which piece you want to tackle first (pitch drawing, transform class, first soccer example, etc.) and I'll help you implement it step by step while keeping the learning goals from the original `goal.md` intact.