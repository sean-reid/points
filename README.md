# Points

Find the simplest algebraic curve through a set of 2D points.

Click points on a grid, and the solver instantly finds the most elegant implicit curve `h(x,y) = 0` — lines, circles, conics, cubics, and elliptic curves.

## How it works

The solver uses **null-space computation on the monomial Vandermonde matrix** — a pure linear algebra approach, not combinatorial search.

For degree d, an implicit curve has `N(d) = (d+1)(d+2)/2` possible monomial terms (like x², xy, y²). Each clicked point gives one linear constraint. The solver:

1. Builds the monomial matrix M (points × monomials)
2. Finds the **sparsest null vector** — the curve with fewest terms
3. Rationalizes to **smallest integer coefficients**
4. Verifies with exact integer arithmetic

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

Performance: **< 1ms** for any query. No precomputation, no pool files, no enumeration. Just linear algebra on tiny matrices.

## Architecture

```
src/
  lib.rs       WASM entry point (~30 lines)
  solver.rs    Core algorithm: Vandermonde → null space → sparsest integers (~490 lines)
web/
  index.html   Minimal shell
  style.css    Responsive styles
  app.js       Wires grid + solver + curve rendering
  grid.js      Interactive canvas grid with marching-squares curve overlay
  solver.js    Web Worker management
  worker.js    WASM bridge
```

Total Rust: ~520 lines. No external dependencies beyond wasm-bindgen.

## Development

```bash
cargo test --release           # Run tests (10 tests, <1s)
wasm-pack build --target web   # Build WASM (~1s)
python3 -m http.server 8080    # Serve, open http://localhost:8080/web/
```

## Deployment

The GitHub Actions workflow builds and deploys to GitHub Pages. No precomputation step needed.
