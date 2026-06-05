# spatiotrack-engine

A hands-on linear algebra engine for **spatial trajectory tracking and network influence analysis**, built around real soccer (football) data problems.

> **Core Philosophy**  
> Build deep geometric and numerical intuition for linear algebra by implementing the actual computational tools needed for a real tracking & analytics system. Every matrix, every algorithm, and every stability concern is motivated by concrete problems that appear when working with player and ball trajectories.

---

## The Vision

This project teaches advanced linear algebra through two progressive, application-driven educational projects.

### Project 1 — Spatiotemporal Tracking & Network Analytics Engine
**Focus:** Low-dimensional geometric linear algebra (R² → Rⁿ)

Key topics implemented and explored:
- Coordinate frame transformations and change of basis
- Affine transforms (rotation, scaling, translation) using homogeneous coordinates
- Condition numbers and numerical stability (why bad camera calibrations destroy tracking accuracy)
- Overdetermined systems & polynomial trajectory fitting (Normal Equations vs QR)
- Null spaces and observability (what motion your sensors cannot see)
- LU decomposition with pivoting
- Power iteration for network influence (proximity & passing graphs)

**Motivating domain:** Soccer player tracking — normalizing multi-camera data onto a pitch, analyzing runs in attacking direction, computing player-local metrics, and more.

📄 Full detailed specification: [goal.md](goal.md)

### Project 2 — High-Dimensional Latent Semantic & Embedding Engine
**Focus:** Statistical and numerical reasoning in high-dimensional spaces (R¹⁰⁰⁰+)

Topics include from-scratch SVD via bidiagonalization, sparse matrix algebra, PCA, truncated rank selection, cosine similarity retrievers, and Frobenius drift detection.

📄 Full detailed specification: [next_goal.md](next_goal.md)

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Linear independence, basis & dimension explorer | ✅ Implemented | Rich visualizations + interactive ipywidgets |
| Module 1: Soccer Coordinate Frames & Transformations | 📖 Guide Ready | Full implementation guide + exercises available |
| Core engine modules (QR, LU, Power Iteration, etc.) | 🚧 In Progress | Following the soccer-themed guide |

The project is primarily educational. The "engine" is the vehicle for learning — not a production library (yet).

---

## Getting Started

### Prerequisites
```bash
pip install numpy matplotlib ipywidgets
```

### Run the Foundational Explorer
```bash
python independence_basis_dimension.py
```

**Recommended environment:** Jupyter Notebook or Google Colab (for the interactive widgets).

---

## Documentation

- **[Module 1: Soccer Coordinate Frames & Affine Transformations](docs/module-01-soccer-transformations.md)** — Complete learning + implementation guide  
  Includes real soccer scenarios (camera calibration, attacking direction normalization, player-local frames), condition number "horror stories", visualization requirements, and concrete coding exercises.

- [goal.md](goal.md) — Detailed architecture and learning objectives for Project 1
- [next_goal.md](next_goal.md) — Detailed architecture for Project 2

---

## Repository Structure

```
spatiotrack-engine/
├── docs/
│   └── module-01-soccer-transformations.md   # Soccer-specific Module 1 guide + exercises
├── goal.md                                   # Project 1 full specification
├── next_goal.md                              # Project 2 full specification
├── independence_basis_dimension.py           # Implemented: basis, independence, dimension + visuals
├── LICENSE
├── .gitignore
└── README.md
```

---

## Recommended Learning Path

1. **Start here** — Run and study `independence_basis_dimension.py` to build intuition for bases and change of coordinates.
2. Read the [soccer Module 1 guide](docs/module-01-soccer-transformations.md).
3. Implement the components step-by-step (begin with `SoccerPitch` drawing + affine transform primitives).
4. Work through the critical numerical stability experiments.
5. Continue into the later modules described in `goal.md`.

---

## Why Soccer?

Soccer tracking data provides an excellent concrete domain:
- Multiple reference frames (broadcast cameras, tactical overhead, GPS, player body frames)
- Real need for accurate meter-scale positions (offsides, progressive carries, physical load)
- Clear geometric problems (curved runs, proximity networks)
- Easy to generate realistic synthetic trajectories for learning

The same mathematical machinery applies to any spatial tracking domain (basketball, autonomous vehicles, robotics, etc.).

---

## License

MIT License

Copyright (c) 2026 Adithyan0122

See [LICENSE](LICENSE) for details.

---

This is a deep learning project. If you're here to truly understand linear algebra (rather than just call library functions), you're in the right place. Pull requests that improve explanations, add better visualizations, or cleanly implement modules are very welcome.