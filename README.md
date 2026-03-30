# Points

Find a short mathematical expression that passes through a set of points.

Given 2D points, the tool searches for an equation `f = g` using `+`, `-`, `*`, `/`, `^` and constants `0-9` that holds at every input point. It returns the shortest expression found and plots the analytic curve.

## Usage

Open `index.html` in a browser. Enter points (one per line, space-separated x y), set a max expression length, and click Search.

## Building

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/):

```
wasm-pack build --target web --release
```

Then serve the directory (e.g. `python3 -m http.server`) and open in a browser.

## How it works

The Rust/WASM core enumerates expression trees bottom-up by size. Expressions that produce the same value vector at the input points are candidate matches. Algebraic equivalences (like `x+y` vs `y+x`) are filtered out via canonical polynomial expansion, leaving only non-trivial identities — equations that hold at the given points but not in general.
