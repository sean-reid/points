//! Core solver: find the simplest implicit algebraic curve through a set of 2D points.
//!
//! Approach: for increasing polynomial degree d, build the monomial Vandermonde matrix and
//! find the sparsest integer null vector that is irreducible over the rationals. A product
//! of lower-degree curves is only returned when nothing else fits within the degree cap.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
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

/// Find the simplest implicit algebraic curve passing through the given points.
/// Returns None if no curve of degree ≤ max_degree fits.
pub fn solve(points: &[(f64, f64)], max_degree: u8) -> Option<CurveResult> {
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

    // Sparsest reducible curve, keyed by (degree, terms, score). Returned only when no
    // irreducible curve exists within max_degree.
    let mut fallback: Option<((u8, usize, u128), Vec<i64>, Vec<Mono>)> = None;

    for d in 1..=max_degree {
        let monos = all_monomials(d);
        let n_mono = monos.len();
        let search = DegreeSearch::new(&pts, &monos);
        if search.null_exact.is_empty() { continue; }

        // Search for the sparsest null vector, starting from fewest terms
        for k in 1..=n_mono {
            let mut best: Option<(Vec<i64>, Vec<usize>)> = None;
            let mut best_score = u128::MAX;

            // Enumerate all k-subsets of monomials
            let subsets = combinations(n_mono, k);
            for subset in &subsets {
                let sub_monos: Vec<Mono> = subset.iter().map(|&i| monos[i]).collect();

                for int_vec in search.candidates(subset) {
                    if !verify_exact(&pts, &sub_monos, &int_vec) { continue; }

                    // Skip if all non-constant coefficients are zero
                    let has_vars = sub_monos.iter().zip(&int_vec)
                        .any(|(&m, &c)| c != 0 && m != (0, 0));
                    if !has_vars { continue; }

                    // Score: prefer fewer terms, smaller coefficients
                    let score = score_curve(&int_vec, &sub_monos);
                    if is_reducible(&pts, &sub_monos, &int_vec) {
                        let key = (d, k, score);
                        if fallback.as_ref().map_or(true, |(fk, _, _)| key < *fk) {
                            fallback = Some((key, int_vec, sub_monos.clone()));
                        }
                        continue;
                    }
                    if score < best_score {
                        best_score = score;
                        best = Some((int_vec, subset.clone()));
                    }
                }
            }

            if let Some((coeffs, subset)) = best {
                let sub_monos: Vec<Mono> = subset.iter().map(|&i| monos[i]).collect();
                return Some(make_result(coeffs, &sub_monos, d, scale));
            }
        }
    }

    fallback.map(|((d, _, _), coeffs, monos)| make_result(coeffs, &monos, d, scale))
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

fn vandermonde_mod(points: &[IPoint], monos: &[Mono]) -> Vec<Vec<u64>> {
    points.iter().map(|&(x, y)| {
        let (x, y) = (to_mod(x), to_mod(y));
        monos.iter().map(|&(i, j)| mul_mod(pow_mod(x, i as u64), pow_mod(y, j as u64))).collect()
    }).collect()
}

// --- Arithmetic modulo a Mersenne prime ---

const P: u64 = (1 << 61) - 1;

fn to_mod(x: i64) -> u64 {
    x.rem_euclid(P as i64) as u64
}

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

/// One degree's Vandermonde matrix and its null space, modular for the rank tests that run
/// on every monomial subset and exact for the few vectors that survive them.
struct DegreeSearch<'a> {
    pts: &'a [IPoint],
    n_mono: usize,
    mat_mod: Vec<Vec<u64>>,
    mat_exact: Vec<Vec<BigInt>>,
    null_mod: Vec<Vec<u64>>,
    null_exact: Vec<Vec<BigInt>>,
}

impl<'a> DegreeSearch<'a> {
    fn new(pts: &'a [IPoint], monos: &[Mono]) -> Self {
        let mat_exact = vandermonde_exact(pts, monos);
        let null_exact = null_space_exact(&mat_exact, monos.len());
        let null_mod = null_exact.iter()
            .map(|v| v.iter().map(big_to_mod).collect())
            .collect();
        DegreeSearch {
            pts,
            n_mono: monos.len(),
            mat_mod: vandermonde_mod(pts, monos),
            mat_exact,
            null_mod,
            null_exact,
        }
    }

    /// Integer null vectors supported on the monomial subset that are worth scoring.
    /// A one-dimensional null space gives its single vector. In a larger one every reduced
    /// basis vector has a zero at another pivot, so it lives in a smaller monomial subset
    /// already searched; the vectors new to this subset are combinations with every basis
    /// coefficient nonzero, and the signed sums stand in for the whole family.
    /// The modular basis decides which sign patterns have full support; exact arithmetic
    /// runs only for subsets with at least one.
    fn candidates(&self, subset: &[usize]) -> Vec<Vec<i64>> {
        let k = subset.len();
        let n = self.pts.len();
        let m = self.null_exact.len();
        let z = self.n_mono - k;
        // Null vectors on the subset are either the null space of the n × k submatrix or
        // the combinations of the degree's null basis that vanish on the z other columns.
        let via_null_basis = z * m * z.min(m) < n * k * n.min(k);
        let excluded: Vec<usize> = (0..self.n_mono).filter(|c| !subset.contains(c)).collect();

        let basis_mod = if via_null_basis {
            let b = extract_rows_of_columns(&self.null_mod, &excluded);
            let mut v: Vec<Vec<u64>> = null_space_mod(&b, m).iter().map(|c| {
                subset.iter().map(|&s| {
                    self.null_mod.iter().zip(c).fold(0, |acc, (row, &ci)| add_mod(acc, mul_mod(ci, row[s])))
                }).collect()
            }).collect();
            rref_mod(&mut v);
            v
        } else {
            null_space_mod(&extract_columns(&self.mat_mod, subset), k)
        };
        let full_support = full_support_patterns(&basis_mod);
        if full_support.is_empty() { return vec![]; }

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
        combine_basis(&basis, &full_support, basis_mod.len())
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

/// Basis of the null space modulo P of an nrows × ncols matrix, one vector per free column.
fn null_space_mod(mat: &[Vec<u64>], ncols: usize) -> Vec<Vec<u64>> {
    let mut m: Vec<Vec<u64>> = mat.to_vec();
    let pivot_cols = rref_mod(&mut m);

    (0..ncols).filter(|c| !pivot_cols.contains(c)).map(|free_col| {
        let mut null_vec = vec![0; ncols];
        null_vec[free_col] = 1;
        for (r, &pc) in pivot_cols.iter().enumerate() {
            null_vec[pc] = sub_mod(0, m[r][free_col]);
        }
        null_vec
    }).collect()
}

/// Reduce the rows to reduced row echelon form over the rationals; returns the pivot columns.
fn rref_exact(m: &mut Vec<Vec<BigRational>>) -> Vec<usize> {
    let nrows = m.len();
    let ncols = if nrows == 0 { return vec![]; } else { m[0].len() };
    let mut pivot_cols = Vec::new();
    let mut row = 0;
    for col in 0..ncols {
        if row >= nrows { break; }
        let Some(pivot_row) = (row..nrows).find(|&r| !m[r][col].is_zero()) else { continue };
        m.swap(row, pivot_row);
        pivot_cols.push(col);

        let pivot = m[row][col].clone();
        for c in 0..ncols { m[row][c] /= &pivot; }
        for r in 0..nrows {
            if r == row || m[r][col].is_zero() { continue; }
            let factor = m[r][col].clone();
            for c in 0..ncols {
                let t = &factor * &m[row][c];
                m[r][c] -= t;
            }
        }
        row += 1;
    }
    pivot_cols
}

/// Basis of the rational null space of an nrows × ncols matrix as primitive integer
/// vectors, one per free column.
fn null_space_exact(mat: &[Vec<BigInt>], ncols: usize) -> Vec<Vec<BigInt>> {
    let mut m: Vec<Vec<BigRational>> = mat.iter()
        .map(|row| row.iter().map(|x| BigRational::from_integer(x.clone())).collect())
        .collect();
    let pivot_cols = rref_exact(&mut m);

    (0..ncols).filter(|c| !pivot_cols.contains(c)).map(|free_col| {
        let mut null_vec = vec![BigRational::zero(); ncols];
        null_vec[free_col] = BigRational::one();
        for (r, &pc) in pivot_cols.iter().enumerate() {
            null_vec[pc] = -m[r][free_col].clone();
        }
        primitive(&null_vec)
    }).collect()
}

/// The rows in reduced row echelon form, each scaled to a primitive integer vector.
fn rref_exact_rows(rows: Vec<Vec<BigInt>>) -> Vec<Vec<BigInt>> {
    let mut m: Vec<Vec<BigRational>> = rows.iter()
        .map(|row| row.iter().map(|x| BigRational::from_integer(x.clone())).collect())
        .collect();
    let rank = rref_exact(&mut m).len();
    m[..rank].iter().map(|row| primitive(row)).collect()
}

/// Clear denominators and divide out the content.
fn primitive(v: &[BigRational]) -> Vec<BigInt> {
    let lcm = v.iter().fold(BigInt::one(), |acc, x| acc.lcm(x.denom()));
    let ints: Vec<BigInt> = v.iter().map(|x| (x * &lcm).to_integer()).collect();
    let g = ints.iter().fold(BigInt::zero(), |acc, x| acc.gcd(x));
    ints.iter().map(|x| x / &g).collect()
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

// --- Irreducibility over the rationals ---

/// Whether the curve is a product of lower-degree curves over the rationals.
/// A conic is reducible iff its matrix is singular. For higher degrees the test looks for a
/// line factor through one of the points, since a factor through none of them would leave
/// the cofactor alone fitting at a lower degree. A quartic that is a product of two conics
/// with no line factor is not detected.
fn is_reducible(points: &[IPoint], monos: &[Mono], coeffs: &[i64]) -> bool {
    let degree = monos.iter().zip(coeffs)
        .filter(|(_, &c)| c != 0)
        .map(|(&(i, j), _)| i + j)
        .max().unwrap_or(0);
    match degree {
        0 | 1 => false,
        2 => conic_is_degenerate(monos, coeffs),
        _ => points.iter().any(|&p| has_line_factor_through(p, monos, coeffs, degree)),
    }
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
fn has_line_factor_through(p: IPoint, monos: &[Mono], coeffs: &[i64], degree: u8) -> bool {
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
fn expand_about(p: IPoint, monos: &[Mono], coeffs: &[i64], degree: u8) -> Vec<Vec<BigInt>> {
    let mut forms: Vec<Vec<BigInt>> = (0..=degree as usize).map(|n| vec![BigInt::zero(); n + 1]).collect();
    for (&(i, j), &c) in monos.iter().zip(coeffs) {
        if c == 0 { continue; }
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
/// and curves that use both x and y independently.
fn score_curve(coeffs: &[i64], monos: &[Mono]) -> u128 {
    let n_terms = coeffs.iter().filter(|&&c| c != 0).count() as u128;
    let max_coeff = coeffs.iter().map(|c| c.unsigned_abs() as u128).max().unwrap_or(0);
    let coeff_sum = coeffs.iter().map(|c| c.unsigned_abs() as u128).sum::<u128>();

    // Prefer both variables used
    let has_x = monos.iter().zip(coeffs).any(|(&(i,_), &c)| i > 0 && c != 0);
    let has_y = monos.iter().zip(coeffs).any(|(&(_,j), &c)| j > 0 && c != 0);
    let var_penalty = if has_x && has_y { 0 } else { 100 };

    // Prefer symmetric use (both x^2 and y^2 present → circle-like)
    let has_x2 = monos.iter().zip(coeffs).any(|(&(i,j), &c)| i == 2 && j == 0 && c != 0);
    let has_y2 = monos.iter().zip(coeffs).any(|(&(i,j), &c)| i == 0 && j == 2 && c != 0);
    let symmetry_bonus = if has_x2 && has_y2 { 0 } else { 10 };

    n_terms * 1000 + coeff_sum + max_coeff + var_penalty + symmetry_bonus
}

// --- Equation formatting ---

fn make_result(coeffs: Vec<i64>, monos: &[Mono], degree: u8, scale: f64) -> CurveResult {
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

// --- Subset enumeration ---

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = Vec::with_capacity(k);
    combinations_helper(n, k, 0, &mut current, &mut result);
    result
}

fn combinations_helper(n: usize, k: usize, start: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }
    let remaining = k - current.len();
    for i in start..=(n - remaining) {
        current.push(i);
        combinations_helper(n, k, i + 1, current, result);
        current.pop();
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

    #[test]
    fn test_line_through_two_points() {
        let result = solve(&[(0.0, 0.0), (3.0, 3.0)], 4).unwrap();
        assert_eq!(result.degree, 1);
        assert!(result.equation.contains('x') && result.equation.contains('y'));
        println!("y=x: {}", result.equation);
    }

    #[test]
    fn test_line_x_plus_y_eq_5() {
        let result = solve(&[(0.0, 5.0), (5.0, 0.0), (2.0, 3.0)], 4).unwrap();
        assert_eq!(result.degree, 1);
        println!("x+y=5: {}", result.equation);
    }

    #[test]
    fn test_circle() {
        let result = solve(&[(3.0, 4.0), (4.0, 3.0), (5.0, 0.0), (0.0, 5.0)], 4).unwrap();
        println!("circle: {}", result.equation);
        // Should contain x^2 and y^2
        assert!(result.equation.contains("x^2") && result.equation.contains("y^2"));
    }

    #[test]
    fn test_hyperbola_xy_eq_6() {
        let result = solve(&[(1.0, 6.0), (2.0, 3.0), (3.0, 2.0), (6.0, 1.0)], 4).unwrap();
        println!("xy=6: {}", result.equation);
        assert!(result.equation.contains("x * y"));
    }

    #[test]
    fn test_parabola_y_eq_x_squared() {
        let result = solve(&[(1.0, 1.0), (2.0, 4.0), (3.0, 9.0), (-1.0, 1.0)], 4).unwrap();
        println!("y=x²: {}", result.equation);
    }

    #[test]
    fn test_elliptic_curve() {
        // The only conic through these is (y - x - 1)(y + x + 1), so the answer is the cubic.
        let result = solve(&[(0.0,1.0), (0.0,-1.0), (-1.0,0.0), (2.0,3.0), (2.0,-3.0)], 4).unwrap();
        assert_eq!(result.equation, "1 + x^3 = y^2");
    }

    #[test]
    fn test_points_on_both_axes_give_ellipse_not_xy() {
        let result = solve(&[(4.0,0.0), (-4.0,0.0), (0.0,2.0), (0.0,-2.0)], 4).unwrap();
        assert_eq!(result.equation, "x^2 + 4 * y^2 = 16");
    }

    #[test]
    fn test_two_parallel_lines_give_irreducible_cubic() {
        // Every conic and every sparse cubic through these is a product of lines, and the
        // cubic that is not lives in a two-dimensional null space.
        let pts = [(0, 1), (1, 1), (2, 1), (0, -1), (1, -1), (2, -1)];
        let fpts: Vec<(f64, f64)> = pts.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
        let result = solve(&fpts, 4).unwrap();
        assert_eq!(result.degree, 3);
        assert!(!is_reducible(&pts, &result.monomials, &result.coefficients));
    }

    #[test]
    fn test_nine_points_give_the_exact_cubic() {
        // The unique cubic through these has nine-digit coefficients. The line x = 4 times a
        // smaller cubic through the other six points must not win by default.
        let pts = [(-9.0,3.0), (-2.0,5.0), (0.0,1.0), (2.0,-4.0), (3.0,6.0), (4.0,9.0), (4.0,4.0), (4.0,0.0), (7.0,5.0)];
        let result = solve(&pts, 4).unwrap();
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
        let result = solve(&pts, 4).unwrap();
        assert_eq!(result.degree, 4);
    }

    #[test]
    fn test_exact_null_space_matches_modular() {
        let pts = [(-9, 3), (-2, 5), (0, 1), (2, -4), (3, 6), (4, 9), (4, 4), (4, 0), (7, 5)];
        let monos = all_monomials(4);
        let exact = null_space_exact(&vandermonde_exact(&pts, &monos), monos.len());
        let modular = null_space_mod(&vandermonde_mod(&pts, &monos), monos.len());
        assert_eq!(exact.len(), 6);
        assert_eq!(modular.len(), 6);
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
        let result = solve(&pts, 4).unwrap();
        assert_eq!(result.equation, "x * y = 0");
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
        let result = solve(&[(1.0, 1.0), (2.0, 8.0), (-1.0, -1.0), (0.0, 0.0)], 4).unwrap();
        println!("y=x³: {}", result.equation);
    }

    #[test]
    fn test_line_arbitrary_slope() {
        // y = 3x/7 + 30/7, or 7y - 3x = 30
        let result = solve(&[(-3.0, 3.0), (4.0, 6.0)], 4).unwrap();
        println!("line -3,3 to 4,6: {}", result.equation);
        assert_eq!(result.degree, 1);
    }

    #[test]
    fn test_horizontal_line() {
        let result = solve(&[(0.0, 3.0), (5.0, 3.0), (-3.0, 3.0)], 4).unwrap();
        println!("y=3: {}", result.equation);
    }

    #[test]
    fn test_vertical_line() {
        let result = solve(&[(5.0, 0.0), (5.0, 3.0), (5.0, -2.0)], 4).unwrap();
        println!("x=5: {}", result.equation);
    }

    #[test]
    fn test_half_integer_line() {
        let result = solve(&[(0.5, 0.5), (2.5, 2.5)], 4).unwrap();
        println!("half y=x: {} (scale={})", result.equation, result.scale);
        assert_eq!(result.scale, 2.0);
        assert_eq!(result.degree, 1);
    }

    #[test]
    fn test_half_integer_circle() {
        let result = solve(&[(1.5, 2.0), (2.0, 1.5), (2.5, 0.0), (0.0, 2.5)], 4).unwrap();
        println!("half circle: {} (scale={})", result.equation, result.scale);
        assert_eq!(result.scale, 2.0);
    }

    #[test]
    fn test_half_integer_5_points() {
        // 5 points on y=x^2 at half-integer x values
        let result = solve(&[(0.5, 0.25), (1.0, 1.0), (1.5, 2.25), (2.0, 4.0), (-1.5, 2.25)], 4).unwrap();
        println!("half parabola: {} (scale={})", result.equation, result.scale);
    }
}
