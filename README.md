# Points

Find the simplest algebraic curve through a set of 2D points.

Click points on a grid, and the solver instantly finds the most elegant implicit curve `h(x,y) = 0` — lines, circles, conics, cubics, and elliptic curves.

## How it works

The solver uses **null-space computation on the monomial Vandermonde matrix** — a pure linear algebra approach, not combinatorial search.

For degree d, an implicit curve has `N(d) = (d+1)(d+2)/2` possible monomial terms (like x², xy, y²). Each clicked point gives one linear constraint. The solver:

1. Builds the monomial matrix M (points × monomials)
2. Finds the **sparsest null vector** — the curve with fewest terms
3. Computes its **integer coefficients exactly**, however large they get: nine scattered points give a cubic with nine-digit coefficients rather than nothing
4. Verifies with exact integer arithmetic
5. Skips curves that split into lower-degree pieces, so five points on `y² = x³ + 1` get the cubic rather than the pair of lines `y = ±(x + 1)`. A product of lines is returned only when nothing else fits within degree 4.

This naturally produces elegant results:
- `x² + y² = 25` for a circle (not some ugly equivalent)
- `y² = x³ + 1` for an elliptic curve
- `x * y = 6` for a hyperbola
- `7 * y = 30 + 3 * x` for a line with arbitrary slope

| Degree | Monomials | Curves | Points needed |
|--------|-----------|--------|---------------|
| 1 | 3 | Lines | 2 |
| 2 | 6 | Conics (circles, parabolas, hyperbolas) | 3-5 |
| 3 | 10 | Cubics (elliptic curves) | 4-9 |
| 4 | 15 | Quartics | 5-14 |

Performance: **under a millisecond** for typical queries, about 10ms for eleven scattered points. Monomial subsets are walked by the terms they exclude, with rank tests modulo a 61-bit prime, and the walk stops once the exclusions rule out every null vector. Exact big-integer arithmetic runs only on the vectors that survive. No precomputation, no pool files, no enumeration. Just linear algebra on tiny matrices.

## Architecture

```
src/
  lib.rs       WASM entry point (~30 lines)
  solver.rs    Core algorithm: Vandermonde → null space → sparsest irreducible integer curve
web/
  index.html   Minimal shell
  style.css    Responsive styles
  app.js       Wires grid + solver + curve rendering
  grid.js      Interactive canvas grid with marching-squares curve overlay
  solver.js    Web Worker management
  worker.js    WASM bridge
```

Total Rust: ~900 lines. Dependencies: wasm-bindgen and the num crates for big integers.

## Development

```bash
cargo test --release                          # Run tests
wasm-pack build --target web --out-dir web/pkg  # Build WASM into web/
cd web && python3 -m http.server 8080         # Open http://localhost:8080
```

## Deployment

The GitHub Actions workflow builds and deploys to GitHub Pages. No precomputation step needed.
