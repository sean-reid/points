//! Core solver: find the simplest implicit algebraic curve through a set of 2D points.
//!
//! Approach: for increasing polynomial degree d, build the monomial Vandermonde matrix and
//! find the sparsest integer null vector that is irreducible over the rationals. A product
//! of lower-degree curves is only returned when nothing else fits within the degree cap.

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// A monomial x^i * y^j, represented as (i, j).
pub type Mono = (u8, u8);

/// A point scaled onto the integer grid.
type IPoint = (i64, i64);

/// Result of the solver: an implicit curve h(x,y) = 0 defined by
/// integer coefficients on a set of monomials.
#[derive(Debug, Clone)]
pub struct CurveResult {
    /// Nonzero integer coefficients.
    pub coefficients: Vec<i64>,
    /// Corresponding monomials (same length as coefficients).
    pub monomials: Vec<Mono>,
    /// Degree of the curve.
    pub degree: u8,
    /// Formatted equation string.
    pub equation: String,
    /// Scale factor applied to coordinates (e.g. 4 means coords were multiplied by 4).
    pub scale: f64,
}

/// A curve through the points: the exact search's answer and whether it is a product of
/// lower-degree curves.
struct Exact {
    coefficients: Vec<i64>,
    monomials: Vec<Mono>,
    degree: u8,
    reducible: bool,
}

/// Find the simplest implicit algebraic curve through the points.
/// The exact curve of lowest degree through the points is found first. When it is unwieldy,
/// more than [`APPROX_MAX_TERMS`] terms or a coefficient of four or more digits, a curve of
/// few small integer terms passing within `tolerance` of every point replaces it if it
/// scores lower, so hand-placed points on a circle get the circle rather than the exact
/// curve through their grid positions.
/// Returns None if no curve of degree ≤ max_degree fits.
pub fn solve(points: &[(f64, f64)], max_degree: u8, tolerance: f64) -> Option<CurveResult> {
    if points.is_empty() { return None; }

    // Find the finest grid the points lie on and scale to integers.
    // E.g. if points are on a 0.25 grid, scale = 4 so 2.25 → 9.
    let scale = find_scale(points);
    let pts = dedup_points(&points.iter()
        .map(|&(x, y)| ((x * scale).round() as i64, (y * scale).round() as i64))
        .collect::<Vec<_>>());
    if pts.is_empty() { return None; }

    // Single point: return simplest line through it
    if pts.len() == 1 {
        let (x, y) = pts[0];
        // Prefer x = xi or y = yi, whichever is simpler
        if x.abs() <= y.abs() {
            return Some(make_result(vec![-x, 1, 0], &all_monomials(1), 1, scale));
        } else {
            return Some(make_result(vec![-y, 0, 1], &all_monomials(1), 1, scale));
        }
    }

    let exact = solve_exact(&pts, max_degree);
    // A presentable exact curve stands. Half a unit is wide next to points a few units
    // apart, and the five points of y^2 = x^3 + 1 lie within it of a hyperbola.
    let max_score = match &exact {
        Some(e) if !e.reducible && e.is_presentable() => {
            return exact.map(|e| make_result(e.coefficients, &e.monomials, e.degree, scale));
        }
        Some(e) if !e.reducible => score_curve(&e.coefficients, &e.monomials),
        _ => u128::MAX,
    };
    if let Some((coeffs, monos, degree)) = solve_within(&pts, tolerance * scale, max_score) {
        return Some(make_result(coeffs, &monos, degree, scale));
    }
    exact.map(|e| make_result(e.coefficients, &e.monomials, e.degree, scale))
}

impl Exact {
    /// At most [`APPROX_MAX_TERMS`] terms, every coefficient under four digits.
    fn is_presentable(&self) -> bool {
        self.coefficients.iter().filter(|&&c| c != 0).count() <= APPROX_MAX_TERMS
            && self.coefficients.iter().all(|c| c.abs() < 1000)
    }
}

/// The sparsest irreducible curve of lowest degree through every point, or the sparsest
/// reducible one when nothing irreducible exists within max_degree.
fn solve_exact(pts: &[IPoint], max_degree: u8) -> Option<Exact> {
    // Sparsest reducible curve, keyed by (degree, terms, score).
    let mut fallback: Option<((u8, usize, u128), Vec<i64>, Vec<Mono>)> = None;

    for d in 1..=max_degree {
        let monos = all_monomials(d);
        let n_mono = monos.len();
        let search = DegreeSearch::new(pts, &monos);
        if search.null_exact.is_empty() { continue; }
        let supported = search.supported_subsets();

        // Score candidates from fewest terms up; reducibility is only checked from the best
        // score down until an irreducible curve appears.
        for k in 1..=n_mono {
            let mut scored: Vec<(u128, Vec<i64>, Vec<Mono>)> = Vec::new();
            for (subset, patterns, modular_dim) in &supported[k] {
                let sub_monos: Vec<Mono> = subset.iter().map(|&i| monos[i]).collect();
                for int_vec in search.exact_candidates(subset, patterns, *modular_dim) {
                    if !verify_exact(pts, &sub_monos, &int_vec) { continue; }
                    // Skip if all non-constant coefficients are zero
                    let has_vars = sub_monos.iter().zip(&int_vec)
                        .any(|(&m, &c)| c != 0 && m != (0, 0));
                    if !has_vars { continue; }
                    scored.push((score_curve(&int_vec, &sub_monos), int_vec, sub_monos.clone()));
                }
            }
            scored.sort_by_key(|(score, _, _)| *score);

            for (score, coeffs, sub_monos) in scored {
                if is_reducible(pts, &sub_monos, &coeffs) {
                    let key = (d, k, score);
                    if fallback.as_ref().map_or(true, |(fk, _, _)| key < *fk) {
                        fallback = Some((key, coeffs, sub_monos));
                    }
                    continue;
                }
                return Some(Exact { coefficients: coeffs, monomials: sub_monos, degree: d, reducible: false });
            }
        }
    }

    fallback.map(|((d, _, _), coefficients, monomials)| Exact { coefficients, monomials, degree: d, reducible: true })
}

// --- Grid scale detection ---

/// Find the smallest integer scale factor such that all coordinates become integers.
/// Tries scale = 1, 2, 4, 5, 10, 20, 100 (common grid denominators).
fn find_scale(points: &[(f64, f64)]) -> f64 {
    let candidates = [1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 100.0];
    for &s in &candidates {
        let all_int = points.iter().all(|&(x, y)| {
            let sx = x * s;
            let sy = y * s;
            (sx - sx.round()).abs() < 1e-6 && (sy - sy.round()).abs() < 1e-6
        });
        if all_int { return s; }
    }
    // Fallback: round to integers
    1.0
}

// --- Monomial generation ---

/// All monomials of degree ≤ d, ordered by total degree, then by x power descending.
fn all_monomials(d: u8) -> Vec<Mono> {
    let mut result = Vec::new();
    for total in 0..=d {
        for i in (0..=total).rev() {
            result.push((i, total - i));
        }
    }
    result
}

// --- Vandermonde matrices ---

fn vandermonde_exact(points: &[IPoint], monos: &[Mono]) -> Vec<Vec<BigInt>> {
    points.iter().map(|&(x, y)| {
        monos.iter().map(|&(i, j)| {
            BigInt::from(x).pow(i as u32) * BigInt::from(y).pow(j as u32)
        }).collect()
    }).collect()
}

// --- Arithmetic modulo a Mersenne prime ---

const P: u64 = (1 << 61) - 1;

fn add_mod(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= P { s - P } else { s }
}

fn sub_mod(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { a + P - b }
}

fn mul_mod(a: u64, b: u64) -> u64 {
    let x = a as u128 * b as u128;
    let r = ((x & P as u128) + (x >> 61)) as u64;
    let r = (r & P) + (r >> 61);
    if r >= P { r - P } else { r }
}

fn pow_mod(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1;
    while exp > 0 {
        if exp & 1 == 1 { result = mul_mod(result, base); }
        base = mul_mod(base, base);
        exp >>= 1;
    }
    result
}

fn inv_mod(a: u64) -> u64 {
    pow_mod(a, P - 2)
}

// --- Null space computation ---

/// One degree's Vandermonde matrix and its null space, modular for the rank walk over
/// monomial subsets and exact for the few vectors that survive it.
struct DegreeSearch<'a> {
    pts: &'a [IPoint],
    n_mono: usize,
    mat_exact: Vec<Vec<BigInt>>,
    null_mod: Vec<Vec<u64>>,
    null_exact: Vec<Vec<BigInt>>,
    /// Monomials some null vector uses; the others are zero in every null vector.
    active: Vec<usize>,
}

/// A monomial subset carrying null vectors with full support: the subset, the sign
/// patterns of its reduced modular basis with full support, and that basis's size.
type Supported = (Vec<usize>, Vec<u32>, usize);

impl<'a> DegreeSearch<'a> {
    fn new(pts: &'a [IPoint], monos: &[Mono]) -> Self {
        let mat_exact = vandermonde_exact(pts, monos);
        let null_exact = null_space_exact(&mat_exact, monos.len());
        let null_mod: Vec<Vec<u64>> = null_exact.iter()
            .map(|v| v.iter().map(big_to_mod).collect())
            .collect();
        let active = (0..monos.len())
            .filter(|&c| null_exact.iter().any(|v| !v[c].is_zero()))
            .collect();
        DegreeSearch { pts, n_mono: monos.len(), mat_exact, null_mod, null_exact, active }
    }

    /// Every monomial subset carrying a null vector with full support, grouped by size.
    /// A subset is walked by the columns it excludes. Its null vectors are the combinations
    /// of the null basis vanishing on those columns, so each exclusion adds a row to a
    /// system in nullity unknowns; once the rows reach the nullity nothing survives on any
    /// superset of the exclusions, and the walk stops there.
    fn supported_subsets(&self) -> Vec<Vec<Supported>> {
        let mut found = vec![Vec::new(); self.n_mono + 1];
        self.walk(0, &mut Vec::new(), &mut Vec::new(), &mut found);
        found
    }

    fn walk(
        &self,
        start: usize,
        echelon: &mut Vec<(usize, Vec<u64>)>,
        excluded: &mut Vec<usize>,
        found: &mut Vec<Vec<Supported>>,
    ) {
        let m = self.null_mod.len();
        let subset: Vec<usize> = self.active.iter().copied().filter(|c| !excluded.contains(c)).collect();
        if !subset.is_empty() {
            let mut basis: Vec<Vec<u64>> = null_space_from_echelon(echelon, m).iter()
                .map(|c| subset.iter().map(|&s| self.combine_mod(c, s)).collect())
                .collect();
            rref_mod(&mut basis);
            let patterns = full_support_patterns(&basis);
            if !patterns.is_empty() {
                found[subset.len()].push((subset, patterns, basis.len()));
            }
        }

        for idx in start..self.active.len() {
            let col = self.active[idx];
            let row: Vec<u64> = self.null_mod.iter().map(|v| v[col]).collect();
            let independent = reduce_row(echelon, row);
            if independent.is_some() && echelon.len() + 1 == m { continue; }
            let pushed = independent.is_some();
            if let Some(pivot_row) = independent { echelon.push(pivot_row); }
            excluded.push(col);
            self.walk(idx + 1, echelon, excluded, found);
            excluded.pop();
            if pushed { echelon.pop(); }
        }
    }

    fn combine_mod(&self, c: &[u64], col: usize) -> u64 {
        self.null_mod.iter().zip(c).fold(0, |acc, (row, &ci)| add_mod(acc, mul_mod(ci, row[col])))
    }

    /// Exact integer null vectors on the subset for the given sign patterns.
    /// A one-dimensional null space gives its single vector. In a larger one every reduced
    /// basis vector has a zero at another pivot, so it lives in a smaller monomial subset
    /// already searched; the vectors new to this subset are combinations with every basis
    /// coefficient nonzero, and the signed sums stand in for the whole family.
    fn exact_candidates(&self, subset: &[usize], patterns: &[u32], modular_dim: usize) -> Vec<Vec<i64>> {
        let k = subset.len();
        let n = self.pts.len();
        let m = self.null_exact.len();
        let excluded: Vec<usize> = self.active.iter().copied().filter(|c| !subset.contains(c)).collect();
        let z = excluded.len();
        // Null vectors on the subset are either the null space of the n × k submatrix or
        // the combinations of the degree's null basis that vanish on the z other columns.
        let via_null_basis = z * m * z.min(m) < n * k * n.min(k);

        let basis = if via_null_basis {
            let b = extract_rows_of_columns(&self.null_exact, &excluded);
            let v: Vec<Vec<BigInt>> = null_space_exact(&b, m).iter().map(|c| {
                subset.iter().map(|&s| {
                    self.null_exact.iter().zip(c).map(|(row, ci)| ci * &row[s]).sum()
                }).collect()
            }).collect();
            if v.len() > 1 { rref_exact_rows(v) } else { v }
        } else {
            null_space_exact(&extract_columns(&self.mat_exact, subset), k)
        };
        combine_basis(&basis, patterns, modular_dim)
    }
}

/// Sign patterns whose modular sum has no zero entry.
fn full_support_patterns(basis_mod: &[Vec<u64>]) -> Vec<u32> {
    let m = basis_mod.len();
    if m == 0 { return vec![]; }
    let ncols = basis_mod[0].len();
    (0..1u32 << (m - 1)).filter(|&signs| {
        (0..ncols).all(|c| {
            let mut sum = 0;
            for (i, b) in basis_mod.iter().enumerate() {
                sum = if sign_is_negative(signs, i) { sub_mod(sum, b[c]) } else { add_mod(sum, b[c]) };
            }
            sum != 0
        })
    }).collect()
}

/// Signed sums of the exact basis, reduced to primitive vectors that fit in i64.
fn combine_basis(basis: &[Vec<BigInt>], full_support: &[u32], modular_dim: usize) -> Vec<Vec<i64>> {
    // The prime divides a minor with negligible probability, but the exact basis decides.
    let patterns: Vec<u32> = if basis.len() == modular_dim {
        full_support.to_vec()
    } else if basis.is_empty() {
        return vec![];
    } else {
        (0..1u32 << (basis.len() - 1)).collect()
    };

    let ncols = basis[0].len();
    patterns.iter().filter_map(|&signs| {
        let mut sum = vec![BigInt::zero(); ncols];
        for (i, b) in basis.iter().enumerate() {
            for (s, x) in sum.iter_mut().zip(b) {
                if sign_is_negative(signs, i) { *s -= x; } else { *s += x; }
            }
        }
        if sum.iter().any(|x| x.is_zero()) { return None; }
        let g = sum.iter().fold(BigInt::zero(), |acc, x| acc.gcd(x));
        sum.iter().map(|x| (x / &g).to_i64()).collect::<Option<Vec<i64>>>()
    }).collect()
}

/// Basis vector 0 is always added; the bits of `signs` choose for the rest.
fn sign_is_negative(signs: u32, i: usize) -> bool {
    i > 0 && signs >> (i - 1) & 1 == 1
}

fn extract_rows_of_columns<T: Clone>(mat: &[Vec<T>], cols: &[usize]) -> Vec<Vec<T>> {
    cols.iter().map(|&c| mat.iter().map(|row| row[c].clone()).collect()).collect()
}

/// Reduce the rows to reduced row echelon form modulo P; returns the pivot columns.
fn rref_mod(m: &mut Vec<Vec<u64>>) -> Vec<usize> {
    let nrows = m.len();
    let ncols = if nrows == 0 { return vec![]; } else { m[0].len() };
    let mut pivot_cols = Vec::new();
    let mut row = 0;
    for col in 0..ncols {
        if row >= nrows { break; }
        let Some(pivot_row) = (row..nrows).find(|&r| m[r][col] != 0) else { continue };
        m.swap(row, pivot_row);
        pivot_cols.push(col);

        let inv = inv_mod(m[row][col]);
        for c in 0..ncols { m[row][c] = mul_mod(m[row][c], inv); }
        for r in 0..nrows {
            if r == row || m[r][col] == 0 { continue; }
            let factor = m[r][col];
            for c in 0..ncols {
                let t = mul_mod(factor, m[row][c]);
                m[r][c] = sub_mod(m[r][c], t);
            }
        }
        row += 1;
    }
    pivot_cols
}

/// Reduce a row against the echelon rows; the normalized row and its pivot column if it
/// is independent of them.
fn reduce_row(echelon: &[(usize, Vec<u64>)], mut row: Vec<u64>) -> Option<(usize, Vec<u64>)> {
    for (p, r) in echelon {
        let f = row[*p];
        if f == 0 { continue; }
        for (x, y) in row.iter_mut().zip(r) { *x = sub_mod(*x, mul_mod(f, *y)); }
    }
    let pivot = row.iter().position(|&x| x != 0)?;
    let inv = inv_mod(row[pivot]);
    for x in row.iter_mut() { *x = mul_mod(*x, inv); }
    Some((pivot, row))
}

/// Null space modulo P of the echelon rows in `ncols` unknowns, one vector per free column.
/// Each row is zero at the pivots of the rows before it, so back substitution runs from the
/// last row to the first.
fn null_space_from_echelon(echelon: &[(usize, Vec<u64>)], ncols: usize) -> Vec<Vec<u64>> {
    let pivots: Vec<usize> = echelon.iter().map(|(p, _)| *p).collect();
    (0..ncols).filter(|c| !pivots.contains(c)).map(|free| {
        let mut v = vec![0; ncols];
        v[free] = 1;
        for (p, r) in echelon.iter().rev() {
            let s = r.iter().zip(&v).enumerate()
                .filter(|(i, _)| i != p)
                .fold(0, |acc, (_, (&ri, &vi))| add_mod(acc, mul_mod(ri, vi)));
            v[*p] = sub_mod(0, s);
        }
        v
    }).collect()
}

/// Fraction-free Gauss-Jordan elimination (Bareiss). Every update divides by the previous
/// pivot, which is exact because each entry is a minor of the original matrix. On return
/// all pivot entries equal the returned scalar. Returns that scalar and the pivot columns.
fn rref_fraction_free(m: &mut Vec<Vec<BigInt>>) -> (BigInt, Vec<usize>) {
    let nrows = m.len();
    let ncols = if nrows == 0 { return (BigInt::one(), vec![]); } else { m[0].len() };
    let mut prev = BigInt::one();
    let mut pivot_cols = Vec::new();
    let mut row = 0;
    for col in 0..ncols {
        if row >= nrows { break; }
        let Some(pivot_row) = (row..nrows).find(|&r| !m[r][col].is_zero()) else { continue };
        m.swap(row, pivot_row);
        pivot_cols.push(col);

        let pivot = m[row][col].clone();
        for r in 0..nrows {
            if r == row { continue; }
            let f = m[r][col].clone();
            for c in 0..ncols {
                let (q, rem) = (&pivot * &m[r][c] - &f * &m[row][c]).div_rem(&prev);
                debug_assert!(rem.is_zero());
                m[r][c] = q;
            }
        }
        prev = pivot;
        row += 1;
    }
    (prev, pivot_cols)
}

/// Basis of the rational null space of an nrows × ncols matrix as primitive integer
/// vectors, one per free column.
fn null_space_exact(mat: &[Vec<BigInt>], ncols: usize) -> Vec<Vec<BigInt>> {
    let mut m = mat.to_vec();
    let (d, pivot_cols) = rref_fraction_free(&mut m);

    (0..ncols).filter(|c| !pivot_cols.contains(c)).map(|free_col| {
        let mut null_vec = vec![BigInt::zero(); ncols];
        null_vec[free_col] = d.clone();
        for (r, &pc) in pivot_cols.iter().enumerate() {
            null_vec[pc] = -m[r][free_col].clone();
        }
        primitive(null_vec)
    }).collect()
}

/// The rows in reduced row echelon form, each scaled to a primitive integer vector.
fn rref_exact_rows(mut rows: Vec<Vec<BigInt>>) -> Vec<Vec<BigInt>> {
    let (_, pivot_cols) = rref_fraction_free(&mut rows);
    rows.truncate(pivot_cols.len());
    rows.into_iter().map(primitive).collect()
}

/// Divide out the content.
fn primitive(v: Vec<BigInt>) -> Vec<BigInt> {
    let g = v.iter().fold(BigInt::zero(), |acc, x| acc.gcd(x));
    if g.is_zero() || g.is_one() { return v; }
    v.into_iter().map(|x| x / &g).collect()
}

fn big_to_mod(x: &BigInt) -> u64 {
    x.mod_floor(&BigInt::from(P)).to_u64().unwrap()
}

// --- Exact integer verification ---

/// Verify that the curve passes through all points using exact integer arithmetic.
fn verify_exact(points: &[IPoint], monos: &[Mono], coeffs: &[i64]) -> bool {
    points.iter().all(|&(x, y)| {
        let sum: BigInt = monos.iter().zip(coeffs).map(|(&(i, j), &c)| {
            BigInt::from(c) * BigInt::from(x).pow(i as u32) * BigInt::from(y).pow(j as u32)
        }).sum();
        sum.is_zero()
    })
}

// --- Approximate fit ---

/// Most terms in an approximate curve.
const APPROX_MAX_TERMS: usize = 5;
/// Highest degree searched for an approximate curve.
const APPROX_MAX_DEGREE: u8 = 3;
/// Largest multiplier applied to the least-squares direction before rounding, which bounds
/// the non-constant coefficients of an approximate curve.
const APPROX_MAX_MULTIPLIER: usize = 60;

/// The lowest-scoring irreducible curve of at most [`APPROX_MAX_TERMS`] small integer
/// terms that passes within `radius` of every point and scores below `max_score`.
/// For each monomial support the least-squares direction is rounded at increasing
/// multipliers; the first rounding within tolerance is the smallest curve on that support.
fn solve_within(pts: &[IPoint], radius: f64, max_score: u128) -> Option<(Vec<i64>, Vec<Mono>, u8)> {
    if radius <= 0.0 { return None; }
    let monos = all_monomials(APPROX_MAX_DEGREE);
    let n_mono = monos.len();
    let values: Vec<Vec<f64>> = pts.iter().map(|&(x, y)| {
        monos.iter().map(|&(i, j)| (x as f64).powi(i as i32) * (y as f64).powi(j as i32)).collect()
    }).collect();
    // Partial derivatives of each monomial at each point, for first-order distance checks.
    let (dx, dy): (Vec<Vec<f64>>, Vec<Vec<f64>>) = pts.iter().map(|&(x, y)| {
        let (x, y) = (x as f64, y as f64);
        monos.iter().map(|&(i, j)| (
            if i == 0 { 0.0 } else { i as f64 * x.powi(i as i32 - 1) * y.powi(j as i32) },
            if j == 0 { 0.0 } else { j as f64 * x.powi(i as i32) * y.powi(j as i32 - 1) },
        )).unzip()
    }).unzip();
    // Whether every point lies within `r` of the curve to first order.
    let near = |support: &[usize], c: &[f64], r: f64| {
        pts.iter().enumerate().all(|(pi, _)| {
            let (mut h, mut gx, mut gy) = (0.0, 0.0, 0.0);
            for (a, &j) in support.iter().enumerate() {
                h += c[a] * values[pi][j];
                gx += c[a] * dx[pi][j];
                gy += c[a] * dy[pi][j];
            }
            h * h <= r * r * (gx * gx + gy * gy)
        })
    };
    // Columns scaled to unit RMS so the Gram matrix is well conditioned.
    let col_scale: Vec<f64> = (0..n_mono).map(|c| {
        let rms = (values.iter().map(|r| r[c] * r[c]).sum::<f64>() / pts.len() as f64).sqrt();
        if rms > 0.0 { 1.0 / rms } else { 1.0 }
    }).collect();
    let gram: Vec<Vec<f64>> = (0..n_mono).map(|a| (0..n_mono).map(|b| {
        values.iter().map(|r| r[a] * r[b]).sum::<f64>() * col_scale[a] * col_scale[b]
    }).collect()).collect();

    let mut best: Option<(Vec<i64>, Vec<Mono>)> = None;
    let mut best_score = max_score;
    let mut supports: Vec<u32> = (1..1u32 << n_mono).filter(|m| (m.count_ones() as usize) <= APPROX_MAX_TERMS).collect();
    supports.sort_by_key(|m| m.count_ones());

    const K: usize = APPROX_MAX_TERMS;
    for mask in supports {
        let k = mask.count_ones() as usize;
        // Sorted by size, and a k-term curve scores at least 1000k + k + 1.
        if (1001 * k + 1) as u128 >= best_score { break; }
        let mut support = [0usize; K];
        let mut sub_monos = [(0u8, 0u8); K];
        for (slot, c) in (0..n_mono).filter(|c| mask >> c & 1 == 1).enumerate() {
            support[slot] = c;
            sub_monos[slot] = monos[c];
        }
        let (support, sub_monos) = (&support[..k], &sub_monos[..k]);
        if sub_monos.iter().all(|&m| m == (0, 0)) { continue; }

        let mut sub = [[0.0; K]; K];
        for a in 0..k {
            for b in 0..k { sub[a][b] = gram[support[a]][support[b]]; }
        }
        let mut v = smallest_eigenvector(sub, k);
        for a in 0..k { v[a] *= col_scale[support[a]]; }
        let v = &mut v[..k];
        let max_var = v.iter().zip(sub_monos)
            .filter(|(_, &m)| m != (0, 0))
            .map(|(x, _)| x.abs())
            .fold(0.0, f64::max);
        if max_var <= 0.0 { continue; }
        for x in v.iter_mut() { *x /= max_var; }
        // Roundings cannot land inside a tolerance the direction itself misses by half.
        if !near(support, v, 1.5 * radius) { continue; }

        let mut last = [i64::MIN; K];
        let mut reducible_hits = 0;
        for t in 1..=APPROX_MAX_MULTIPLIER {
            let mut c = [0i64; K];
            for a in 0..k { c[a] = (v[a] * t as f64).round() as i64; }
            let g = c[..k].iter().fold(0i64, |acc, &x| gcd(acc, x));
            if g == 0 { continue; }
            if g > 1 { for x in c[..k].iter_mut() { *x /= g; } }
            if c[..k] == last[..k] { continue; }
            last = c;

            let score = score_curve(&c[..k], sub_monos);
            if score >= best_score { continue; }
            let mut cf = [0.0; K];
            for a in 0..k { cf[a] = c[a] as f64; }
            if !near(support, &cf[..k], 1.5 * radius) { continue; }
            if !within_tolerance(pts, sub_monos, &cf[..k], radius) { continue; }
            if is_reducible_small(sub_monos, &c[..k]) {
                // Points on a product of lines make every rounding of this direction one.
                reducible_hits += 1;
                if reducible_hits == 3 { break; }
                continue;
            }
            best_score = score;
            best = Some((c[..k].to_vec(), sub_monos.to_vec()));
            break;
        }
    }

    best.map(|(c, m)| {
        let degree = curve_degree(&m, &c);
        (c, m, degree)
    })
}

/// Whether the curve passes within `radius` of every point, by Newton projection of each
/// point onto the curve. A point whose first step is over twice the radius is rejected
/// without iterating.
fn within_tolerance(pts: &[IPoint], monos: &[Mono], coeffs: &[f64], radius: f64) -> bool {
    pts.iter().all(|&(x0, y0)| {
        let (px, py) = (x0 as f64, y0 as f64);
        let (mut x, mut y) = (px, py);
        let (mut h, mut g) = (0.0, 0.0);
        for iter in 0..8 {
            let (hv, gx, gy) = eval_with_gradient(monos, coeffs, x, y);
            let g2 = gx * gx + gy * gy;
            if g2 < 1e-18 { return hv.abs() < 1e-9; }
            h = hv;
            g = g2.sqrt();
            let step = h.abs() / g;
            if iter == 0 && step > 2.0 * radius { return false; }
            if step <= 1e-9 * radius { break; }
            x -= h * gx / g2;
            y -= h * gy / g2;
        }
        let dist = ((x - px).powi(2) + (y - py).powi(2)).sqrt();
        h.abs() <= 1e-6 * g * radius && dist <= radius
    })
}

fn eval_with_gradient(monos: &[Mono], coeffs: &[f64], x: f64, y: f64) -> (f64, f64, f64) {
    let mut xp = [1.0; 5];
    let mut yp = [1.0; 5];
    for k in 1..5 {
        xp[k] = xp[k - 1] * x;
        yp[k] = yp[k - 1] * y;
    }
    let (mut h, mut gx, mut gy) = (0.0, 0.0, 0.0);
    for (&(i, j), &c) in monos.iter().zip(coeffs) {
        if c == 0.0 { continue; }
        let (i, j) = (i as usize, j as usize);
        h += c * xp[i] * yp[j];
        if i > 0 { gx += c * i as f64 * xp[i - 1] * yp[j]; }
        if j > 0 { gy += c * j as f64 * xp[i] * yp[j - 1]; }
    }
    (h, gx, gy)
}

/// Eigenvector of the smallest eigenvalue of the leading n × n block of a symmetric
/// positive semidefinite matrix, by inverse iteration on the matrix shifted just off
/// singular.
fn smallest_eigenvector(mut a: [[f64; APPROX_MAX_TERMS]; APPROX_MAX_TERMS], n: usize) -> [f64; APPROX_MAX_TERMS] {
    let trace: f64 = (0..n).map(|i| a[i][i]).sum();
    for i in 0..n { a[i][i] += trace * 1e-12 + f64::MIN_POSITIVE; }
    let mut v = [0.0; APPROX_MAX_TERMS];
    for i in 0..n { v[i] = 1.0 + 0.1 * i as f64; }
    for _ in 0..8 {
        let Some(mut w) = solve_linear(&a, &v, n) else { break };
        let norm = w[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(norm > 0.0 && norm.is_finite()) { break; }
        for x in w[..n].iter_mut() { *x /= norm; }
        let change: f64 = (0..n).map(|i| (v[i] - w[i]).abs().min((v[i] + w[i]).abs())).sum();
        v = w;
        if change < 1e-9 { break; }
    }
    v
}

/// Solve a x = b on the leading n × n block by Gaussian elimination with partial pivoting.
fn solve_linear(a: &[[f64; APPROX_MAX_TERMS]; APPROX_MAX_TERMS], b: &[f64; APPROX_MAX_TERMS], n: usize) -> Option<[f64; APPROX_MAX_TERMS]> {
    let mut m = [[0.0; APPROX_MAX_TERMS + 1]; APPROX_MAX_TERMS];
    for r in 0..n {
        m[r][..n].copy_from_slice(&a[r][..n]);
        m[r][n] = b[r];
    }
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[pivot][col] == 0.0 { return None; }
        m.swap(col, pivot);
        for r in col + 1..n {
            let f = m[r][col] / m[col][col];
            if f == 0.0 { continue; }
            for c in col..=n { m[r][c] -= f * m[col][c]; }
        }
    }
    let mut x = [0.0; APPROX_MAX_TERMS];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|c| m[r][c] * x[c]).sum();
        x[r] = (m[r][n] - s) / m[r][r];
    }
    Some(x)
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

// --- Irreducibility over the rationals ---

/// Whether the curve is a product of lower-degree curves over the rationals.
/// A conic is reducible iff its matrix is singular. For higher degrees the test looks for a
/// line factor through one of the points, since a factor through none of them would leave
/// the cofactor alone fitting at a lower degree. A quartic that is a product of two conics
/// with no line factor is not detected.
fn is_reducible(points: &[IPoint], monos: &[Mono], coeffs: &[i64]) -> bool {
    let degree = curve_degree(monos, coeffs);
    match degree {
        0 | 1 => false,
        2 => conic_is_degenerate(monos, coeffs),
        _ => {
            let big: Vec<BigInt> = coeffs.iter().map(|&c| BigInt::from(c)).collect();
            points.iter().any(|&p| has_line_factor_through(p, monos, &big, degree))
        }
    }
}

/// Whether a curve with small coefficients, not necessarily through any of the points, is
/// a product of lower-degree curves over the rationals. For a cubic that means a line
/// factor, whose direction is a rational root of the top-degree form and whose offset is a
/// rational root of the curve restricted to an axis; both come from divisors of the
/// coefficients, which is only feasible because they are small.
fn is_reducible_small(monos: &[Mono], coeffs: &[i64]) -> bool {
    let degree = curve_degree(monos, coeffs);
    match degree {
        0 | 1 => false,
        2 => conic_is_degenerate(monos, coeffs),
        _ => has_line_factor_small(monos, coeffs, degree),
    }
}

fn curve_degree(monos: &[Mono], coeffs: &[i64]) -> u8 {
    monos.iter().zip(coeffs)
        .filter(|(_, &c)| c != 0)
        .map(|(&(i, j), _)| i + j)
        .max().unwrap_or(0)
}

fn has_line_factor_small(monos: &[Mono], coeffs: &[i64], degree: u8) -> bool {
    let c = |i: u8, j: u8| coeff_of(monos, coeffs, (i, j));
    let on_x_axis: Vec<BigInt> = (0..=degree).map(|i| c(i, 0)).collect();
    let on_y_axis: Vec<BigInt> = (0..=degree).map(|j| c(0, j)).collect();
    // h(x, 0) or h(0, y) identically zero means y or x divides h.
    if on_x_axis.iter().all(|x| x.is_zero()) || on_y_axis.iter().all(|x| x.is_zero()) { return true; }

    // Directions (u, v) of linear factors ux + vy of the top form.
    let top: Vec<BigInt> = (0..=degree).map(|i| c(i, degree - i)).collect();
    let mut dirs: Vec<(i64, i64)> = Vec::new();
    if top[0].is_zero() { dirs.push((1, 0)); }
    if top[degree as usize].is_zero() { dirs.push((0, 1)); }
    dirs.extend(rational_roots(&top).into_iter().filter(|&(p, _)| p != 0).map(|(p, q)| (q, -p)));

    let big: Vec<BigInt> = coeffs.iter().map(|&x| BigInt::from(x)).collect();
    dirs.iter().any(|&(u, v)| {
        if v == 0 {
            // The line x = a/b meets the x axis where h(x, 0) has a rational root.
            rational_roots(&on_x_axis).iter().any(|&(a, b)| vanishes_on_rational_line(monos, &big, degree, (a, 0), b, (0, 1)))
        } else {
            // The line ux + vy + w = 0 meets the y axis where h(0, y) has a rational root.
            rational_roots(&on_y_axis).iter().any(|&(a, b)| vanishes_on_rational_line(monos, &big, degree, (0, a), b, (v, -u)))
        }
    })
}

/// Whether the curve vanishes along the line through (origin / b) with the given direction.
/// Scaling coordinates by b turns the question into one about the integer polynomial with
/// coefficients c_ij * b^(degree - i - j) along an integer line.
fn vanishes_on_rational_line(monos: &[Mono], coeffs: &[BigInt], degree: u8, origin: IPoint, b: i64, dir: (i64, i64)) -> bool {
    let scaled: Vec<BigInt> = monos.iter().zip(coeffs)
        .map(|(&(i, j), c)| c * BigInt::from(b).pow((degree - i - j) as u32))
        .collect();
    let forms = expand_about(origin, monos, &scaled, degree);
    let (u, v) = (BigInt::from(dir.0 * b), BigInt::from(dir.1 * b));
    forms.iter().all(|f| eval_form(f, &u, &v).is_zero())
}

/// Rational roots p/q of the polynomial with the given coefficients (index = power), as
/// (p, q) with q > 0 in lowest terms. The zero polynomial has none reported.
fn rational_roots(poly: &[BigInt]) -> Vec<(i64, i64)> {
    let Some(lo) = poly.iter().position(|c| !c.is_zero()) else { return vec![] };
    let hi = poly.iter().rposition(|c| !c.is_zero()).unwrap();
    let mut roots = Vec::new();
    if lo > 0 { roots.push((0, 1)); }
    if lo == hi { return roots; }
    let (Some(trailing), Some(leading)) = (poly[lo].to_i64(), poly[hi].to_i64()) else { return roots };
    for q in divisors(leading) {
        for p in divisors(trailing) {
            if gcd(p, q) != 1 { continue; }
            for p in [p, -p] {
                let sum: BigInt = (lo..=hi)
                    .map(|i| &poly[i] * BigInt::from(p).pow((i - lo) as u32) * BigInt::from(q).pow((hi - i) as u32))
                    .sum();
                if sum.is_zero() { roots.push((p, q)); }
            }
        }
    }
    roots
}

fn divisors(n: i64) -> Vec<i64> {
    let n = n.abs();
    let mut result = Vec::new();
    let mut d = 1;
    while d * d <= n {
        if n % d == 0 {
            result.push(d);
            if d * d != n { result.push(n / d); }
        }
        d += 1;
    }
    result
}

fn coeff_of(monos: &[Mono], coeffs: &[i64], mono: Mono) -> BigInt {
    monos.iter().zip(coeffs)
        .find(|(&m, _)| m == mono)
        .map_or(BigInt::zero(), |(_, &c)| BigInt::from(c))
}

fn conic_is_degenerate(monos: &[Mono], coeffs: &[i64]) -> bool {
    let a = coeff_of(monos, coeffs, (2, 0));
    let b = coeff_of(monos, coeffs, (1, 1));
    let c = coeff_of(monos, coeffs, (0, 2));
    let d = coeff_of(monos, coeffs, (1, 0));
    let e = coeff_of(monos, coeffs, (0, 1));
    let f = coeff_of(monos, coeffs, (0, 0));
    // Determinant of the doubled symmetric matrix [[2a, b, d], [b, 2c, e], [d, e, 2f]].
    let two = BigInt::from(2);
    let det = &two * &a * (&two * &two * &c * &f - &e * &e) - &b * (&two * &b * &f - &e * &d) + &d * (&b * &e - &two * &c * &d);
    det.is_zero()
}

/// Whether some line through the point divides the curve.
/// Along a line through p in direction (u, v), h(p + t(u, v)) is a polynomial in t whose
/// t^n coefficient is the degree-n form of h expanded about p. The line divides h iff every
/// form vanishes at (u, v). The linear form pins the direction when p is a smooth point; at
/// a double point the quadratic form has at most two rational roots. A point of multiplicity
/// three or more is skipped; some other point on the same line is then smooth or double.
fn has_line_factor_through(p: IPoint, monos: &[Mono], coeffs: &[BigInt], degree: u8) -> bool {
    let forms = expand_about(p, monos, coeffs, degree);
    let vanishes = |u: &BigInt, v: &BigInt| forms.iter().all(|f| eval_form(f, u, v).is_zero());

    let (hx, hy) = (&forms[1][1], &forms[1][0]);
    if !hx.is_zero() || !hy.is_zero() {
        return vanishes(hy, &-hx);
    }

    let (a, b, c) = (&forms[2][2], &forms[2][1], &forms[2][0]);
    if a.is_zero() {
        if b.is_zero() && c.is_zero() { return false; }
        return vanishes(&BigInt::one(), &BigInt::zero()) || (!b.is_zero() && vanishes(c, &-b));
    }
    let disc = b * b - BigInt::from(4) * a * c;
    if disc.is_negative() { return false; }
    let s = disc.sqrt();
    if &s * &s != disc { return false; }
    let two_a = BigInt::from(2) * a;
    vanishes(&(-b + &s), &two_a) || vanishes(&(-b - &s), &two_a)
}

/// Homogeneous forms of h expanded about p: forms[n][a] is the coefficient of u^a v^(n-a)
/// in h(p.0 + u, p.1 + v).
fn expand_about(p: IPoint, monos: &[Mono], coeffs: &[BigInt], degree: u8) -> Vec<Vec<BigInt>> {
    let mut forms: Vec<Vec<BigInt>> = (0..=degree as usize).map(|n| vec![BigInt::zero(); n + 1]).collect();
    for (&(i, j), c) in monos.iter().zip(coeffs) {
        if c.is_zero() { continue; }
        let px = binomial_powers(p.0, i);
        let py = binomial_powers(p.1, j);
        for (a, cx) in px.iter().enumerate() {
            for (b, cy) in py.iter().enumerate() {
                forms[a + b][a] += c * cx * cy;
            }
        }
    }
    forms
}

/// Coefficients of (x0 + u)^n as a polynomial in u.
fn binomial_powers(x0: i64, n: u8) -> Vec<BigInt> {
    let mut result = vec![BigInt::one()];
    for _ in 0..n {
        let mut next = vec![BigInt::zero(); result.len() + 1];
        for (k, c) in result.iter().enumerate() {
            next[k] += c * x0;
            next[k + 1] += c;
        }
        result = next;
    }
    result
}

fn eval_form(form: &[BigInt], u: &BigInt, v: &BigInt) -> BigInt {
    let n = form.len() - 1;
    form.iter().enumerate()
        .map(|(a, c)| c * u.pow(a as u32) * v.pow((n - a) as u32))
        .sum()
}

// --- Elegance scoring ---

/// Score a curve: lower is better. Prefers fewer terms, smaller coefficients,
/// and curves that use both x and y.
fn score_curve(coeffs: &[i64], monos: &[Mono]) -> u128 {
    let n_terms = coeffs.iter().filter(|&&c| c != 0).count() as u128;
    let max_coeff = coeffs.iter().map(|c| c.unsigned_abs() as u128).max().unwrap_or(0);
    let coeff_sum = coeffs.iter().map(|c| c.unsigned_abs() as u128).sum::<u128>();

    let has_x = monos.iter().zip(coeffs).any(|(&(i,_), &c)| i > 0 && c != 0);
    let has_y = monos.iter().zip(coeffs).any(|(&(_,j), &c)| j > 0 && c != 0);
    let var_penalty = if has_x && has_y { 0 } else { 100 };

    n_terms * 1000 + coeff_sum + max_coeff + var_penalty
}

// --- Equation formatting ---

fn make_result(mut coeffs: Vec<i64>, monos: &[Mono], degree: u8, scale: f64) -> CurveResult {
    // Orient so the highest-order monomial has a positive coefficient.
    if coeffs.iter().rev().find(|&&c| c != 0).is_some_and(|&c| c < 0) {
        for c in coeffs.iter_mut() { *c = -*c; }
    }
    let equation = format_equation(&coeffs, monos);
    let (nz_coeffs, nz_monos): (Vec<i64>, Vec<Mono>) = coeffs.iter().zip(monos)
        .filter(|(&c, _)| c != 0)
        .map(|(&c, &m)| (c, m))
        .unzip();

    CurveResult {
        coefficients: nz_coeffs,
        monomials: nz_monos,
        degree,
        equation,
        scale,
    }
}

/// Format coefficients + monomials as "positive_terms = negative_terms".
fn format_equation(coeffs: &[i64], monos: &[Mono]) -> String {
    let mut pos_terms = Vec::new();
    let mut neg_terms = Vec::new();

    for (&c, &(i, j)) in coeffs.iter().zip(monos) {
        if c == 0 { continue; }
        let mono_str = format_monomial(i, j);
        let abs_c = c.unsigned_abs();

        let term = if mono_str.is_empty() {
            // Constant term
            format!("{}", abs_c)
        } else if abs_c == 1 {
            mono_str
        } else {
            format!("{} * {}", abs_c, mono_str)
        };

        if c > 0 { pos_terms.push(term); }
        else { neg_terms.push(term); }
    }

    let lhs = if pos_terms.is_empty() { "0".into() } else { pos_terms.join(" + ") };
    let rhs = if neg_terms.is_empty() { "0".into() } else { neg_terms.join(" + ") };

    format!("{} = {}", lhs, rhs)
}

fn format_monomial(i: u8, j: u8) -> String {
    match (i, j) {
        (0, 0) => String::new(), // constant
        (1, 0) => "x".into(),
        (0, 1) => "y".into(),
        (i, 0) => format!("x^{}", i),
        (0, j) => format!("y^{}", j),
        (1, 1) => "x * y".into(),
        (i, 1) => format!("x^{} * y", i),
        (1, j) => format!("x * y^{}", j),
        (i, j) => format!("x^{} * y^{}", i, j),
    }
}

fn extract_columns<T: Clone>(mat: &[Vec<T>], cols: &[usize]) -> Vec<Vec<T>> {
    mat.iter().map(|row| cols.iter().map(|&c| row[c].clone()).collect()).collect()
}

// --- Point deduplication ---

fn dedup_points(points: &[IPoint]) -> Vec<IPoint> {
    let mut result: Vec<IPoint> = Vec::new();
    for &p in points {
        if !result.contains(&p) { result.push(p); }
    }
    result
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn solve_t(points: &[(f64, f64)], max_degree: u8) -> Option<CurveResult> {
        solve(points, max_degree, 0.5)
    }

    #[test]
    fn test_line_through_two_points() {
        let result = solve_t(&[(0.0, 0.0), (3.0, 3.0)], 4).unwrap();
        assert_eq!(result.degree, 1);
        assert!(result.equation.contains('x') && result.equation.contains('y'));
        println!("y=x: {}", result.equation);
    }

    #[test]
    fn test_line_x_plus_y_eq_5() {
        let result = solve_t(&[(0.0, 5.0), (5.0, 0.0), (2.0, 3.0)], 4).unwrap();
        assert_eq!(result.degree, 1);
        println!("x+y=5: {}", result.equation);
    }

    #[test]
    fn test_circle() {
        let result = solve_t(&[(3.0, 4.0), (4.0, 3.0), (5.0, 0.0), (0.0, 5.0)], 4).unwrap();
        println!("circle: {}", result.equation);
        // Should contain x^2 and y^2
        assert!(result.equation.contains("x^2") && result.equation.contains("y^2"));
    }

    #[test]
    fn test_hyperbola_xy_eq_6() {
        let result = solve_t(&[(1.0, 6.0), (2.0, 3.0), (3.0, 2.0), (6.0, 1.0)], 4).unwrap();
        println!("xy=6: {}", result.equation);
        assert!(result.equation.contains("x * y"));
    }

    #[test]
    fn test_parabola_y_eq_x_squared() {
        let result = solve_t(&[(1.0, 1.0), (2.0, 4.0), (3.0, 9.0), (-1.0, 1.0)], 4).unwrap();
        println!("y=x²: {}", result.equation);
    }

    #[test]
    fn test_elliptic_curve() {
        // The only conic through these is (y - x - 1)(y + x + 1), so the answer is the cubic.
        let result = solve_t(&[(0.0,1.0), (0.0,-1.0), (-1.0,0.0), (2.0,3.0), (2.0,-3.0)], 4).unwrap();
        assert_eq!(result.equation, "1 + x^3 = y^2");
    }

    #[test]
    fn test_points_on_both_axes_give_ellipse_not_xy() {
        let result = solve_t(&[(4.0,0.0), (-4.0,0.0), (0.0,2.0), (0.0,-2.0)], 4).unwrap();
        assert_eq!(result.equation, "x^2 + 4 * y^2 = 16");
    }

    #[test]
    fn test_two_parallel_lines_give_irreducible_cubic() {
        // Every conic and every sparse cubic through these is a product of lines, and the
        // cubic that is not lives in a two-dimensional null space.
        let pts = [(0, 1), (1, 1), (2, 1), (0, -1), (1, -1), (2, -1)];
        let fpts: Vec<(f64, f64)> = pts.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
        let result = solve_t(&fpts, 4).unwrap();
        assert_eq!(result.degree, 3);
        assert!(!is_reducible(&pts, &result.monomials, &result.coefficients));
    }

    #[test]
    fn test_nine_points_give_the_exact_cubic() {
        // The unique cubic through these has nine-digit coefficients. The line x = 4 times a
        // smaller cubic through the other six points must not win by default.
        let pts = [(-9.0,3.0), (-2.0,5.0), (0.0,1.0), (2.0,-4.0), (3.0,6.0), (4.0,9.0), (4.0,4.0), (4.0,0.0), (7.0,5.0)];
        let result = solve_t(&pts, 4).unwrap();
        assert_eq!(result.degree, 3);
        assert_eq!(result.coefficients.iter().map(|c| c.abs()).max(), Some(161831344));
        assert_eq!(
            result.equation,
            "142822992 + 14322552 * x^2 + 67912625 * x * y + 17242003 * y^2 + 231902 * x^3 + 1766349 * y^3 = 96706388 * x + 161831344 * y + 2889412 * x^2 * y + 10051135 * x * y^2"
        );
    }

    #[test]
    fn test_eleven_scattered_points_give_a_quartic() {
        let pts = [(1.0,2.0), (3.0,5.0), (-2.0,4.0), (5.0,-1.0), (0.0,7.0), (-4.0,-3.0), (6.0,6.0), (2.0,-5.0), (-6.0,1.0), (7.0,3.0), (-3.0,-7.0)];
        let result = solve_t(&pts, 4).unwrap();
        assert_eq!(result.degree, 4);
    }

    #[test]
    fn test_exact_null_space_is_primitive_and_vanishes() {
        let pts = [(-9, 3), (-2, 5), (0, 1), (2, -4), (3, 6), (4, 9), (4, 4), (4, 0), (7, 5)];
        let monos = all_monomials(4);
        let exact = null_space_exact(&vandermonde_exact(&pts, &monos), monos.len());
        assert_eq!(exact.len(), 6);
        for v in &exact {
            for &(x, y) in &pts {
                let sum: BigInt = monos.iter().zip(v)
                    .map(|(&(i, j), c)| c * BigInt::from(x).pow(i as u32) * BigInt::from(y).pow(j as u32))
                    .sum();
                assert!(sum.is_zero());
            }
            let g = v.iter().fold(BigInt::zero(), |acc, x| acc.gcd(x));
            assert!(g.is_one());
        }
    }

    #[test]
    fn test_only_reducible_curves_fall_back() {
        // Five points on each axis: every curve of degree at most 4 is divisible by x * y.
        let pts: Vec<_> = (1..=5).map(|i| (i as f64, 0.0)).chain((1..=5).map(|i| (0.0, i as f64))).collect();
        let result = solve(&pts, 4, 0.0).unwrap();
        assert_eq!(result.equation, "x * y = 0");
        // Within half a unit the circle about (3, 3) passes all ten, and it is irreducible.
        let result = solve_t(&pts, 4).unwrap();
        assert_eq!(result.equation, "7 + x^2 + y^2 = 6 * x + 6 * y");
    }

    #[test]
    fn test_approximate_circle_beats_exact_conic() {
        // Five points, one of them off the circle x^2 + y^2 = 25 by 0.099, so the exact
        // conic through them has a four-digit coefficient.
        let pts = [(5.0, 0.0), (0.0, 5.0), (-5.0, 0.0), (3.0, 4.0), (1.0, -5.0)];
        let exact = solve(&pts, 4, 0.0).unwrap();
        assert_eq!(exact.equation, "15 * y + 145 * x^2 + 142 * y^2 = 3625 + x * y");
        let result = solve_t(&pts, 4).unwrap();
        assert_eq!(result.equation, "x^2 + y^2 = 25");
    }

    #[test]
    fn test_presentable_exact_curve_survives_tolerance() {
        // Four points near the circle have an exact three-term conic, which stands.
        let pts = [(5.0, 0.0), (0.0, 5.0), (-5.0, 0.0), (1.0, -5.0)];
        assert_eq!(solve_t(&pts, 4).unwrap().equation, solve(&pts, 4, 0.0).unwrap().equation);
        // The five points of y^2 = x^3 + 1 lie within half a unit of a hyperbola.
        let pts = [(0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (2.0, 3.0), (2.0, -3.0)];
        assert_eq!(solve_t(&pts, 4).unwrap().equation, "1 + x^3 = y^2");
    }

    #[test]
    fn test_approximate_shifted_circle_beats_exact_cubic() {
        // (x - 3)^2 + (y - 2)^2 = 25 with (8, 2) nudged to (8, 3).
        let pts = [(8.0, 3.0), (3.0, 7.0), (-2.0, 2.0), (3.0, -3.0), (7.0, 5.0), (-1.0, 5.0)];
        let exact = solve(&pts, 4, 0.0).unwrap();
        assert_eq!(exact.degree, 3);
        let result = solve_t(&pts, 4).unwrap();
        assert_eq!(result.equation, "x^2 + y^2 = 12 + 6 * x + 4 * y");
    }

    #[test]
    fn test_exact_line_through_two_points_survives_tolerance() {
        // x^2 = 3y passes within 0.24 of both points with two terms, but the exact line is
        // presentable and stands.
        let result = solve_t(&[(-3.0, 3.0), (4.0, 6.0)], 4).unwrap();
        assert_eq!(result.equation, "7 * y = 30 + 3 * x");
    }

    #[test]
    fn test_small_line_factors() {
        // (y - x^2)(x - 3) = x y - 3 y - x^3 + 3 x^2
        assert!(is_reducible_small(&[(1, 1), (0, 1), (3, 0), (2, 0)], &[1, -3, -1, 3]));
        // (2x + y - 1)(x^2 + y^2 - 4): a slanted line factor meeting the y axis at 1
        assert!(is_reducible_small(
            &[(3, 0), (2, 1), (1, 2), (0, 3), (2, 0), (1, 1), (0, 2), (1, 0), (0, 1), (0, 0)],
            &[2, 1, 2, 1, -1, 0, -1, -8, -4, 4],
        ));
        // x y (x - y): three lines through the origin
        assert!(is_reducible_small(&[(2, 1), (1, 2)], &[1, -1]));
        // y = x^3 and y^2 = x^3 + 1 are irreducible
        assert!(!is_reducible_small(&[(0, 1), (3, 0)], &[1, -1]));
        assert!(!is_reducible_small(&[(0, 2), (3, 0), (0, 0)], &[1, -1, -1]));
        // x y = 6 is irreducible; x y = 0 is not
        assert!(!is_reducible_small(&[(1, 1), (0, 0)], &[1, -6]));
        assert!(is_reducible_small(&[(1, 1)], &[1]));
    }

    #[test]
    fn test_degenerate_conics() {
        let m = all_monomials(2);
        // y^2 - x^2 - 2x - 1
        assert!(conic_is_degenerate(&m, &[-1, -2, 0, -1, 0, 1]));
        // x * y
        assert!(conic_is_degenerate(&m, &[0, 0, 0, 0, 1, 0]));
        // x^2 + y^2 - 25
        assert!(!conic_is_degenerate(&m, &[-25, 0, 0, 1, 0, 1]));
        // y - x^2
        assert!(!conic_is_degenerate(&m, &[0, 0, 1, -1, 0, 0]));
    }

    #[test]
    fn test_supported_subsets_carry_exact_full_support_vectors() {
        // Eleven points at degree 4: nullity 4, so at most three exclusions are independent
        // and the walk visits a few hundred subsets out of 32767.
        let pts = [(1, 2), (3, 5), (-2, 4), (5, -1), (0, 7), (-4, -3), (6, 6), (2, -5), (-6, 1), (7, 3), (-3, -7)];
        let monos = all_monomials(4);
        let search = DegreeSearch::new(&pts, &monos);
        let supported = search.supported_subsets();
        let total: usize = supported.iter().map(|b| b.len()).sum();
        assert!(total > 0 && total < 600, "{total}");
        for (k, bucket) in supported.iter().enumerate() {
            for (subset, patterns, dim) in bucket {
                assert_eq!(subset.len(), k);
                for v in search.exact_candidates(subset, patterns, *dim) {
                    assert!(v.iter().all(|&c| c != 0));
                    let sub_monos: Vec<Mono> = subset.iter().map(|&i| monos[i]).collect();
                    assert!(verify_exact(&pts, &sub_monos, &v));
                }
            }
        }
    }

    #[test]
    fn test_walk_matches_brute_force() {
        // Cases with dependent columns, where the walk keeps excluding without gaining rank.
        let cases: [&[IPoint]; 3] = [
            &[(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)],
            &[(1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)],
            &[(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (3, 0)],
        ];
        for pts in cases {
            let monos = all_monomials(3);
            let search = DegreeSearch::new(pts, &monos);
            let mut walked: Vec<Vec<usize>> = search.supported_subsets().into_iter().flatten().map(|(s, _, _)| s).collect();
            walked.sort();

            let mut brute = Vec::new();
            for mask in 1u32..1 << monos.len() {
                let subset: Vec<usize> = (0..monos.len()).filter(|c| mask >> c & 1 == 1).collect();
                let basis = null_space_exact(&extract_columns(&search.mat_exact, &subset), subset.len());
                if basis.is_empty() { continue; }
                let basis = if basis.len() > 1 { rref_exact_rows(basis) } else { basis };
                let basis_mod: Vec<Vec<u64>> = basis.iter().map(|v| v.iter().map(big_to_mod).collect()).collect();
                if !full_support_patterns(&basis_mod).is_empty() { brute.push(subset); }
            }
            brute.sort();
            assert_eq!(walked, brute);
        }
    }

    #[test]
    fn test_line_factors_in_cubics() {
        let pts = [(0, 1), (0, -1), (-1, 0), (2, 3), (2, -3)];
        // 2xy - x^2 y = x * y * (2 - x)
        assert!(is_reducible(&pts, &[(1, 1), (2, 1)], &[2, -1]));
        // (x + y + 1)(x^2 + y^2 - 1), line through (0, -1) in a slanted direction
        assert!(is_reducible(&[(0, -1)], &[(3, 0), (2, 1), (1, 2), (0, 3), (2, 0), (0, 2), (1, 0), (0, 1), (0, 0)], &[1, 1, 1, 1, 1, 1, -1, -1, -1]));
        // x * y * (x - y) through the origin only: a triple point, so undetected there,
        // but (1, 0) is a smooth point of the curve on y = 0.
        assert!(!is_reducible(&[(0, 0)], &[(2, 1), (1, 2)], &[1, -1]));
        assert!(is_reducible(&[(0, 0), (1, 0)], &[(2, 1), (1, 2)], &[1, -1]));
        // x * y * (x - y) * (x + y - 2): the origin is a triple point, (1, 1) a double point.
        assert!(is_reducible(&[(0, 0), (1, 1)], &[(3, 1), (2, 2), (1, 3), (2, 1), (1, 2)], &[1, 0, -1, -2, 2]));
        // y^2 - x^3 - 1
        assert!(!is_reducible(&pts, &[(0, 2), (3, 0), (0, 0)], &[1, -1, -1]));
        // y^2 - x^3 - x^2: singular but irreducible
        assert!(!is_reducible(&[(0, 0), (-1, 0)], &[(0, 2), (3, 0), (2, 0)], &[1, -1, -1]));
        // (x - 4) times the cubic through six of the nine points, with large coefficients
        let nine = [(-9, 3), (-2, 5), (0, 1), (2, -4), (3, 6), (4, 9), (4, 4), (4, 0), (7, 5)];
        let monos = [(1,0), (0,1), (2,0), (3,1), (2,2), (0,0), (0,2), (3,0), (2,1), (1,2)];
        assert!(is_reducible(&nine, &monos, &[57462, 23472, 6277, 1193, 582, -22248, -1224, -4813, -6239, -2022]));
    }

    #[test]
    fn test_cubic_y_eq_x_cubed() {
        let result = solve_t(&[(1.0, 1.0), (2.0, 8.0), (-1.0, -1.0), (0.0, 0.0)], 4).unwrap();
        println!("y=x³: {}", result.equation);
    }

    #[test]
    fn test_line_arbitrary_slope() {
        // y = 3x/7 + 30/7, or 7y - 3x = 30
        let result = solve_t(&[(-3.0, 3.0), (4.0, 6.0)], 4).unwrap();
        println!("line -3,3 to 4,6: {}", result.equation);
        assert_eq!(result.degree, 1);
    }

    #[test]
    fn test_horizontal_line() {
        let result = solve_t(&[(0.0, 3.0), (5.0, 3.0), (-3.0, 3.0)], 4).unwrap();
        println!("y=3: {}", result.equation);
    }

    #[test]
    fn test_vertical_line() {
        let result = solve_t(&[(5.0, 0.0), (5.0, 3.0), (5.0, -2.0)], 4).unwrap();
        println!("x=5: {}", result.equation);
    }

    #[test]
    fn test_half_integer_line() {
        let result = solve_t(&[(0.5, 0.5), (2.5, 2.5)], 4).unwrap();
        println!("half y=x: {} (scale={})", result.equation, result.scale);
        assert_eq!(result.scale, 2.0);
        assert_eq!(result.degree, 1);
    }

    #[test]
    fn test_half_integer_circle() {
        let result = solve_t(&[(1.5, 2.0), (2.0, 1.5), (2.5, 0.0), (0.0, 2.5)], 4).unwrap();
        println!("half circle: {} (scale={})", result.equation, result.scale);
        assert_eq!(result.scale, 2.0);
    }

    #[test]
    fn test_half_integer_5_points() {
        // 5 points on y=x^2 at half-integer x values
        let result = solve_t(&[(0.5, 0.25), (1.0, 1.0), (1.5, 2.25), (2.0, 4.0), (-1.5, 2.25)], 4).unwrap();
        println!("half parabola: {} (scale={})", result.equation, result.scale);
    }
}
