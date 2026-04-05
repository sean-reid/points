pub mod solver;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn solve(points_str: &str) -> JsValue {
    let mut points = Vec::new();
    for line in points_str.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let coords: Vec<f64> = line.split_whitespace()
            .filter_map(|s| s.parse().ok()).collect();
        if coords.len() == 2 {
            points.push((coords[0], coords[1]));
        }
    }

    match solver::solve(&points, 4) {
        Some(result) => {
            // Return as tab-separated: equation \t coeff1,coeff2,... \t i1:j1,i2:j2,...
            let coeffs_str: String = result.coefficients.iter()
                .map(|c| c.to_string()).collect::<Vec<_>>().join(",");
            let monos_str: String = result.monomials.iter()
                .map(|(i, j)| format!("{}:{}", i, j)).collect::<Vec<_>>().join(",");
            JsValue::from_str(&format!("{}\t{}\t{}", result.equation, coeffs_str, monos_str))
        }
        None => JsValue::NULL,
    }
}
