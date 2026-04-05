//! Core solver: find the simplest implicit algebraic curve through a set of 2D points.
//!
//! Approach: for increasing polynomial degree d, build the monomial Vandermonde matrix,
//! find the sparsest integer null vector — that's the curve equation.
//!
//! This is a pure linear algebra problem, not a combinatorial search.

/// A monomial x^i * y^j, represented as (i, j).
pub type Mono = (u8, u8);

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
}

/// Find the simplest implicit algebraic curve passing through the given points.
/// Returns None if no curve of degree ≤ max_degree fits.
pub fn solve(points: &[(f64, f64)], max_degree: u8) -> Option<CurveResult> {
    if points.is_empty() { return None; }

    // Snap to integers and deduplicate
    let pts = dedup_points(&points.iter()
        .map(|&(x, y)| (x.round(), y.round()))
        .collect::<Vec<_>>());
    if pts.is_empty() { return None; }

    // Single point: return simplest line through it
    if pts.len() == 1 {
        let (x, y) = pts[0];
        let xi = x.round() as i64;
        let yi = y.round() as i64;
        // Prefer x = xi or y = yi, whichever is simpler
        if xi.abs() <= yi.abs() {
            return Some(make_result(vec![-xi, 1, 0], &all_monomials(1), 1));
        } else {
            return Some(make_result(vec![-yi, 0, 1], &all_monomials(1), 1));
        }
    }

    for d in 1..=max_degree {
        let monos = all_monomials(d);
        let n_mono = monos.len();
        let n_pts = pts.len();

        // Build Vandermonde matrix: M[point][monomial] = x^i * y^j
        let mat = build_vandermonde(&pts, &monos);

        // Search for the sparsest null vector, starting from fewest terms
        for k in 2..=n_mono {
            let mut best: Option<(Vec<i64>, Vec<usize>)> = None;
            let mut best_score = u64::MAX;

            // Enumerate all k-subsets of monomials
            let subsets = combinations(n_mono, k);
            for subset in &subsets {
                // Extract submatrix M_S
                let sub_mat = extract_columns(&mat, subset);

                // Find null vector of sub_mat (if rank < k)
                if let Some(null_vec) = find_null_vector(&sub_mat) {
                    // Rationalize to integers
                    if let Some(int_vec) = rationalize(&null_vec) {
                        // Verify with exact integer arithmetic
                        let sub_monos: Vec<Mono> = subset.iter().map(|&i| monos[i]).collect();
                        if verify_exact(&pts, &sub_monos, &int_vec) {
                            // Skip if all non-constant coefficients are zero
                            let has_vars = subset.iter().zip(int_vec.iter())
                                .any(|(&mi, &c)| c != 0 && monos[mi] != (0, 0));
                            if !has_vars { continue; }

                            // Score: prefer fewer terms, smaller coefficients
                            let score = score_curve(&int_vec, &sub_monos);
                            if score < best_score {
                                best_score = score;
                                best = Some((int_vec.clone(), subset.clone()));
                            }
                        }
                    }
                }
            }

            if let Some((coeffs, subset)) = best {
                let sub_monos: Vec<Mono> = subset.iter().map(|&i| monos[i]).collect();
                return Some(make_result(coeffs, &sub_monos, d));
            }
        }
    }

    None
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

// --- Vandermonde matrix ---

fn build_vandermonde(points: &[(f64, f64)], monos: &[Mono]) -> Vec<Vec<f64>> {
    points.iter().map(|&(x, y)| {
        monos.iter().map(|&(i, j)| {
            x.powi(i as i32) * y.powi(j as i32)
        }).collect()
    }).collect()
}

// --- Null space computation via Gaussian elimination ---

/// Find a non-trivial null vector of the matrix (nrows × ncols, nrows < ncols ideally).
/// Returns None if the matrix has full column rank.
fn find_null_vector(mat: &[Vec<f64>]) -> Option<Vec<f64>> {
    let nrows = mat.len();
    let ncols = if nrows == 0 { return None; } else { mat[0].len() };
    if ncols < 2 { return None; }

    // Augmented matrix for elimination
    let mut m: Vec<Vec<f64>> = mat.to_vec();

    // Gaussian elimination with partial pivoting
    let mut pivot_cols = Vec::new();
    let mut row = 0;
    for col in 0..ncols {
        if row >= nrows { break; }

        // Find pivot
        let mut max_val = 0.0f64;
        let mut max_row = row;
        for r in row..nrows {
            if m[r][col].abs() > max_val {
                max_val = m[r][col].abs();
                max_row = r;
            }
        }

        if max_val < 1e-10 { continue; } // Skip this column (free variable)

        // Swap rows
        m.swap(row, max_row);
        pivot_cols.push(col);

        // Eliminate below and above
        let pivot = m[row][col];
        for r in 0..nrows {
            if r == row { continue; }
            let factor = m[r][col] / pivot;
            for c in 0..ncols {
                m[r][c] -= factor * m[row][c];
            }
            m[r][col] = 0.0; // Exact zero
        }

        // Normalize pivot row
        for c in 0..ncols {
            m[row][c] /= pivot;
        }

        row += 1;
    }

    let rank = pivot_cols.len();
    if rank >= ncols { return None; } // Full rank, no null space

    // Find a free column (not a pivot column)
    let pivot_set: Vec<bool> = (0..ncols).map(|c| pivot_cols.contains(&c)).collect();
    let free_col = (0..ncols).find(|&c| !pivot_set[c])?;

    // Build null vector: set free variable to 1, solve for pivot variables
    let mut null_vec = vec![0.0; ncols];
    null_vec[free_col] = 1.0;

    for (r, &pc) in pivot_cols.iter().enumerate() {
        null_vec[pc] = -m[r][free_col];
    }

    Some(null_vec)
}

// --- Rationalization: float vector → integer vector ---

/// Convert a float vector to an integer vector by finding a common scaling factor.
fn rationalize(v: &[f64]) -> Option<Vec<i64>> {
    // Find the best rational approximation for each non-zero entry
    let mut denominators = Vec::new();
    for &x in v {
        if x.abs() < 1e-10 { continue; }
        let (_, d) = best_rational(x, 1000);
        denominators.push(d);
    }

    if denominators.is_empty() { return None; }

    // LCM of all denominators
    let mut lcm = 1i64;
    for &d in &denominators {
        lcm = lcm / gcd(lcm, d) * d;
        if lcm > 100000 { return None; } // Denominator too large
    }

    // Scale and round
    let scaled: Vec<i64> = v.iter().map(|&x| {
        (x * lcm as f64).round() as i64
    }).collect();

    // Divide by GCD of all entries
    let g = scaled.iter().fold(0i64, |acc, &x| gcd(acc, x.abs()));
    if g == 0 { return None; }

    let result: Vec<i64> = scaled.iter().map(|&x| x / g).collect();

    // Check coefficients are reasonable
    if result.iter().any(|&x| x.abs() > 10000) { return None; }

    Some(result)
}

/// Best rational approximation p/q of x with |q| ≤ max_q, via continued fractions.
fn best_rational(x: f64, max_q: i64) -> (i64, i64) {
    let sign = if x < 0.0 { -1 } else { 1 };
    let x = x.abs();

    let mut p0 = 0i64;
    let mut q0 = 1i64;
    let mut p1 = 1i64;
    let mut q1 = 0i64;

    let mut val = x;
    for _ in 0..20 {
        let a = val.floor() as i64;
        let p2 = a * p1 + p0;
        let q2 = a * q1 + q0;

        if q2 > max_q { break; }

        p0 = p1; q0 = q1;
        p1 = p2; q1 = q2;

        let frac = val - a as f64;
        if frac.abs() < 1e-12 { break; }
        val = 1.0 / frac;
    }

    if q1 == 0 { (0, 1) } else { (sign * p1, q1) }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs(); b = b.abs();
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

// --- Exact integer verification ---

/// Verify that the curve passes through all points using exact integer arithmetic.
fn verify_exact(points: &[(f64, f64)], monos: &[Mono], coeffs: &[i64]) -> bool {
    for &(px, py) in points {
        // Convert to integers (multiply by 2 if half-integers, but for now round)
        let x = px.round() as i64;
        let y = py.round() as i64;

        let mut sum: i128 = 0;
        for (k, &(i, j)) in monos.iter().enumerate() {
            let term = coeffs[k] as i128
                * pow_i128(x as i128, i as u32)
                * pow_i128(y as i128, j as u32);
            sum += term;
        }

        if sum != 0 { return false; }
    }
    true
}

fn pow_i128(base: i128, exp: u32) -> i128 {
    let mut result = 1i128;
    for _ in 0..exp { result *= base; }
    result
}

// --- Elegance scoring ---

/// Score a curve: lower is better. Prefers fewer terms, smaller coefficients,
/// and curves that use both x and y independently.
fn score_curve(coeffs: &[i64], monos: &[Mono]) -> u64 {
    let n_terms = coeffs.iter().filter(|&&c| c != 0).count() as u64;
    let max_coeff = coeffs.iter().map(|c| c.abs()).max().unwrap_or(0) as u64;
    let coeff_sum = coeffs.iter().map(|c| c.abs()).sum::<i64>() as u64;

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

fn make_result(coeffs: Vec<i64>, monos: &[Mono], degree: u8) -> CurveResult {
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
    }
}

/// Format coefficients + monomials as "positive_terms = negative_terms".
fn format_equation(coeffs: &[i64], monos: &[Mono]) -> String {
    let mut pos_terms = Vec::new();
    let mut neg_terms = Vec::new();

    for (&c, &(i, j)) in coeffs.iter().zip(monos) {
        if c == 0 { continue; }
        let mono_str = format_monomial(i, j);
        let abs_c = c.abs();

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

fn extract_columns(mat: &[Vec<f64>], cols: &[usize]) -> Vec<Vec<f64>> {
    mat.iter().map(|row| cols.iter().map(|&c| row[c]).collect()).collect()
}

// --- Point deduplication ---

fn dedup_points(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    for &p in points {
        if !result.iter().any(|&q: &(f64, f64)| (p.0 - q.0).abs() < 0.01 && (p.1 - q.1).abs() < 0.01) {
            result.push(p);
        }
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
        // y² = x³ + 1: points (0,1), (0,-1), (-1,0), (2,3), (2,-3)
        let result = solve(&[(0.0,1.0), (0.0,-1.0), (-1.0,0.0), (2.0,3.0), (2.0,-3.0)], 4).unwrap();
        println!("elliptic y²=x³+1: {}", result.equation);
        assert!(result.equation.contains("x^3") || result.equation.contains("y^2"));
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
}
