use std::collections::{HashMap, BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
}

// Compile-time log level (change this to control verbosity)
const COMPILE_LOG_LEVEL: LogLevel = LogLevel::Error;

// Logging wrapper
struct Logger<'a> {
    callback: &'a dyn Fn(String),
}

impl<'a> Logger<'a> {
    fn new(callback: &'a dyn Fn(String)) -> Self {
        Logger { callback }
    }
    
    fn log(&self, level: LogLevel, msg: String) {
        if level >= COMPILE_LOG_LEVEL {
            let prefix = match level {
                LogLevel::Debug => "[DEBUG]",
                LogLevel::Info => "[INFO]",
                LogLevel::Warn => "[WARN]",
                LogLevel::Error => "[ERROR]",
            };
            (self.callback)(format!("{} {}", prefix, msg));
        }
    }
    
    fn debug(&self, msg: String) {
        self.log(LogLevel::Debug, msg);
    }
    
    fn info(&self, msg: String) {
        self.log(LogLevel::Info, msg);
    }
    
    fn warn(&self, msg: String) {
        self.log(LogLevel::Warn, msg);
    }
    
    fn error(&self, msg: String) {
        self.log(LogLevel::Error, msg);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Expr {
    Var(char),
    Const(i32),
    Add(Arc<Expr>, Arc<Expr>),
    Sub(Arc<Expr>, Arc<Expr>),
    Neg(Arc<Expr>),
    Mul(Arc<Expr>, Arc<Expr>),
    Div(Arc<Expr>, Arc<Expr>),
    Pow(Arc<Expr>, Arc<Expr>),
}

impl Expr {
    fn size(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 1,
            Expr::Neg(e) => 1 + e.size(),
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) 
            | Expr::Div(l, r) | Expr::Pow(l, r) => 1 + l.size() + r.size(),
        }
    }

    fn to_string(&self) -> String {
        match self {
            Expr::Var(c) => c.to_string(),
            Expr::Const(n) => n.to_string(),
            Expr::Neg(e) => {
                match e.as_ref() {
                    Expr::Var(_) | Expr::Const(_) => format!("-{}", e.to_string()),
                    _ => format!("-({})", e.to_string())
                }
            },
            Expr::Add(l, r) => format!("{} + {}", l.to_string(), r.to_string()),
            Expr::Sub(l, r) => {
                let right = match r.as_ref() {
                    Expr::Add(..) | Expr::Sub(..) => format!("({})", r.to_string()),
                    _ => r.to_string()
                };
                format!("{} - {}", l.to_string(), right)
            },
            Expr::Mul(l, r) => {
                let left = match l.as_ref() {
                    Expr::Add(..) | Expr::Sub(..) => format!("({})", l.to_string()),
                    _ => l.to_string()
                };
                let right = match r.as_ref() {
                    Expr::Add(..) | Expr::Sub(..) => format!("({})", r.to_string()),
                    _ => r.to_string()
                };
                format!("{} * {}", left, right)
            },
            Expr::Div(l, r) => {
                let left = match l.as_ref() {
                    Expr::Add(..) | Expr::Sub(..) => format!("({})", l.to_string()),
                    _ => l.to_string()
                };
                let right = match r.as_ref() {
                    Expr::Add(..) | Expr::Sub(..) | Expr::Mul(..) | Expr::Div(..) => 
                        format!("({})", r.to_string()),
                    _ => r.to_string()
                };
                format!("{} / {}", left, right)
            },
            Expr::Pow(l, r) => {
                let left = match l.as_ref() {
                    Expr::Var(_) | Expr::Const(_) => l.to_string(),
                    _ => format!("({})", l.to_string())
                };
                let right = match r.as_ref() {
                    Expr::Var(_) | Expr::Const(_) => r.to_string(),
                    _ => format!("({})", r.to_string())
                };
                format!("{}^{}", left, right)
            },
        }
    }
}

#[derive(Default)]
struct ExprInterner {
    cache: HashMap<Expr, Arc<Expr>>,
}

impl ExprInterner {
    fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    fn intern(&mut self, expr: Expr) -> Arc<Expr> {
        if let Some(cached) = self.cache.get(&expr) {
            Arc::clone(cached)
        } else {
            let arc = Arc::new(expr.clone());
            self.cache.insert(expr, Arc::clone(&arc));
            arc
        }
    }

    fn mk_var(&mut self, c: char) -> Arc<Expr> {
        self.intern(Expr::Var(c))
    }

    fn mk_const(&mut self, n: i32) -> Arc<Expr> {
        self.intern(Expr::Const(n))
    }

    fn mk_neg(&mut self, e: Arc<Expr>) -> Arc<Expr> {
        if let Expr::Const(n) = e.as_ref() {
            return self.mk_const(-n);
        }
        if let Expr::Neg(inner) = e.as_ref() {
            return Arc::clone(inner);
        }
        self.intern(Expr::Neg(e))
    }

    fn mk_add(&mut self, l: Arc<Expr>, r: Arc<Expr>) -> Option<Arc<Expr>> {
        let (l, r) = if l <= r { (l, r) } else { (r, l) };

        if let (Expr::Const(a), Expr::Const(b)) = (l.as_ref(), r.as_ref()) {
            return Some(self.mk_const(a + b));
        }

        if matches!(r.as_ref(), Expr::Const(0)) {
            return Some(l);
        }
        if matches!(l.as_ref(), Expr::Const(0)) {
            return Some(r);
        }

        Some(self.intern(Expr::Add(l, r)))
    }

    fn mk_sub(&mut self, l: Arc<Expr>, r: Arc<Expr>) -> Option<Arc<Expr>> {
        if let (Expr::Const(a), Expr::Const(b)) = (l.as_ref(), r.as_ref()) {
            return Some(self.mk_const(a - b));
        }

        if matches!(r.as_ref(), Expr::Const(0)) {
            return Some(l);
        }

        if l == r {
            return Some(self.mk_const(0));
        }

        Some(self.intern(Expr::Sub(l, r)))
    }

    fn mk_mul(&mut self, l: Arc<Expr>, r: Arc<Expr>) -> Option<Arc<Expr>> {
        let (l, r) = if l <= r { (l, r) } else { (r, l) };

        if let (Expr::Const(a), Expr::Const(b)) = (l.as_ref(), r.as_ref()) {
            return Some(self.mk_const(a * b));
        }

        if matches!(l.as_ref(), Expr::Const(0)) || matches!(r.as_ref(), Expr::Const(0)) {
            return Some(self.mk_const(0));
        }

        if matches!(r.as_ref(), Expr::Const(1)) {
            return Some(l);
        }
        if matches!(l.as_ref(), Expr::Const(1)) {
            return Some(r);
        }

        Some(self.intern(Expr::Mul(l, r)))
    }

    fn mk_div(&mut self, l: Arc<Expr>, r: Arc<Expr>) -> Option<Arc<Expr>> {
        if let (Expr::Const(a), Expr::Const(b)) = (l.as_ref(), r.as_ref()) {
            if *b == 0 {
                return None;
            }
            if a % b == 0 {
                return Some(self.mk_const(a / b));
            }
        }

        if matches!(r.as_ref(), Expr::Const(0)) {
            return None;
        }

        if matches!(r.as_ref(), Expr::Const(1)) {
            return Some(l);
        }

        if l == r {
            return Some(self.mk_const(1));
        }

        Some(self.intern(Expr::Div(l, r)))
    }

    fn mk_pow(&mut self, l: Arc<Expr>, r: Arc<Expr>) -> Option<Arc<Expr>> {
        if let (Expr::Const(a), Expr::Const(b)) = (l.as_ref(), r.as_ref()) {
            if *b < 0 || *b > 10 {
                return None;
            }
            let result = (*a as i64).pow(*b as u32);
            if result.abs() > 1_000_000 {
                return None;
            }
            return Some(self.mk_const(result as i32));
        }

        if matches!(r.as_ref(), Expr::Const(0)) {
            return Some(self.mk_const(1));
        }

        if matches!(r.as_ref(), Expr::Const(1)) {
            return Some(l);
        }

        if matches!(l.as_ref(), Expr::Const(0)) {
            return Some(self.mk_const(0));
        }

        if matches!(l.as_ref(), Expr::Const(1)) {
            return Some(self.mk_const(1));
        }

        Some(self.intern(Expr::Pow(l, r)))
    }
}

type ValueVector = Vec<f64>;
type Point = Vec<f64>;

fn eval(expr: &Expr, point: &Point, var_map: &BTreeMap<char, usize>) -> Option<f64> {
    match expr {
        Expr::Var(c) => var_map.get(c).and_then(|&idx| point.get(idx)).copied(),
        Expr::Const(n) => Some(*n as f64),
        Expr::Neg(e) => eval(e, point, var_map).map(|v| -v),
        Expr::Add(l, r) => {
            let lv = eval(l, point, var_map)?;
            let rv = eval(r, point, var_map)?;
            Some(lv + rv)
        }
        Expr::Sub(l, r) => {
            let lv = eval(l, point, var_map)?;
            let rv = eval(r, point, var_map)?;
            Some(lv - rv)
        }
        Expr::Mul(l, r) => {
            let lv = eval(l, point, var_map)?;
            let rv = eval(r, point, var_map)?;
            Some(lv * rv)
        }
        Expr::Div(l, r) => {
            let lv = eval(l, point, var_map)?;
            let rv = eval(r, point, var_map)?;
            if rv.abs() < 1e-15 { None } else { Some(lv / rv) }
        }
        Expr::Pow(l, r) => {
            let lv = eval(l, point, var_map)?;
            let rv = eval(r, point, var_map)?;
            
            if rv.abs() > 20.0 {
                return None;
            }
            
            let result = lv.powf(rv);
            if result.is_finite() && result.abs() < 1e10 && result.abs() > 1e-10 { 
                Some(result) 
            } else { 
                None 
            }
        }
    }
}

fn compute_value_vector(
    expr: &Expr,
    points: &[Point],
    var_map: &BTreeMap<char, usize>
) -> Option<ValueVector> {
    points.iter()
        .map(|p| eval(expr, p, var_map))
        .collect()
}

fn normalize_vector(vec: &ValueVector) -> Vec<i64> {
    const SCALE: f64 = 1e12;
    vec.iter().map(|&v| (v * SCALE).round() as i64).collect()
}

#[derive(Default)]
struct ExpressionsBySize {
    by_size: HashMap<usize, Vec<Arc<Expr>>>,
}

impl ExpressionsBySize {
    fn add(&mut self, expr: Arc<Expr>) {
        let size = expr.size();
        self.by_size.entry(size).or_insert_with(Vec::new).push(expr);
    }

    fn get(&self, size: usize) -> Option<&Vec<Arc<Expr>>> {
        self.by_size.get(&size)
    }
}

fn are_structurally_equivalent(e1: &Expr, e2: &Expr) -> bool {
    match (e1, e2) {
        (Expr::Var(c1), Expr::Var(c2)) => c1 == c2,
        (Expr::Const(n1), Expr::Const(n2)) => n1 == n2,
        (Expr::Neg(a1), Expr::Neg(a2)) => are_structurally_equivalent(a1, a2),
        (Expr::Add(l1, r1), Expr::Add(l2, r2)) |
        (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) |
        (Expr::Mul(l1, r1), Expr::Mul(l2, r2)) |
        (Expr::Div(l1, r1), Expr::Div(l2, r2)) |
        (Expr::Pow(l1, r1), Expr::Pow(l2, r2)) => {
            are_structurally_equivalent(l1, l2) && are_structurally_equivalent(r1, r2)
        }
        _ => false,
    }
}

fn get_variables(expr: &Expr) -> HashSet<char> {
    let mut vars = HashSet::new();
    
    fn collect_vars(e: &Expr, vars: &mut HashSet<char>) {
        match e {
            Expr::Var(c) => { vars.insert(*c); }
            Expr::Const(_) => {}
            Expr::Neg(a) => collect_vars(a, vars),
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) 
            | Expr::Div(l, r) | Expr::Pow(l, r) => {
                collect_vars(l, vars);
                collect_vars(r, vars);
            }
        }
    }
    
    collect_vars(expr, &mut vars);
    vars
}

fn use_all_variables(e1: &Expr, e2: &Expr, required_vars: &BTreeSet<char>) -> bool {
    let vars1 = get_variables(e1);
    let vars2 = get_variables(e2);
    
    let all_vars: HashSet<char> = vars1.union(&vars2).copied().collect();
    
    required_vars.iter().all(|v| all_vars.contains(v))
}

fn are_algebraically_equivalent(e1: &Expr, e2: &Expr, var_map: &BTreeMap<char, usize>) -> bool {
    let canon1 = to_canonical_form(e1);
    let canon2 = to_canonical_form(e2);
    
    let both_valid = match (&canon1, &canon2) {
        (CanonicalExpr::Sum(v1), CanonicalExpr::Sum(v2)) => {
            !v1.is_empty() && !v2.is_empty()
        }
    };
    
    if both_valid {
        canon1 == canon2
    } else {
        // If canonicalization failed, fall back to evaluating on test points
        // to detect simple algebraic equivalences
        let test_points = vec![
            vec![1.0; var_map.len()],
            vec![2.0; var_map.len()],
            vec![3.0; var_map.len()],
            (0..var_map.len()).map(|i| (i + 1) as f64).collect::<Vec<_>>(),
        ];
        
        for test_point in test_points {
            let val1 = eval(e1, &test_point, var_map);
            let val2 = eval(e2, &test_point, var_map);
            
            match (val1, val2) {
                (Some(v1), Some(v2)) => {
                    if (v1 - v2).abs() > 1e-10 {
                        return false; // Different on this test point
                    }
                }
                (None, None) => continue,
                _ => return false,
            }
        }
        
        // If they match on all test points, likely equivalent
        true
    }
}

fn canonical_to_string(canon: &CanonicalExpr) -> String {
    match canon {
        CanonicalExpr::Sum(products) => {
            if products.is_empty() {
                return "FAILED".to_string();
            }
            let terms: Vec<String> = products.iter().map(|p| {
                let mut s = format!("{}", p.coeff);
                for factor in &p.terms {
                    let Factor::Var(c, exp) = factor;
                    if *exp == 1 {
                        s.push_str(&format!("*{}", c));
                    } else {
                        s.push_str(&format!("*{}^{}", c, exp));
                    }
                }
                s
            }).collect();
            terms.join(" + ")
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum CanonicalExpr {
    Sum(Vec<Product>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Product {
    coeff: i64,
    terms: Vec<Factor>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Factor {
    Var(char, i32),
}

fn to_canonical_form(expr: &Expr) -> CanonicalExpr {
    let expanded = expand_expr(expr);
    let products = expanded.0;
    
    let mut term_map: HashMap<Vec<Factor>, i64> = HashMap::new();
    for product in products {
        *term_map.entry(product.terms).or_insert(0) += product.coeff;
    }
    
    let mut result: Vec<Product> = term_map
        .into_iter()
        .filter(|(_, coeff)| *coeff != 0)
        .map(|(terms, coeff)| Product { coeff, terms })
        .collect();
    
    result.sort();
    
    CanonicalExpr::Sum(result)
}

#[derive(Debug, Clone)]
struct ExpandedExpr(Vec<Product>);

fn expand_expr(expr: &Expr) -> ExpandedExpr {
    match expr {
        Expr::Const(n) => {
            if *n == 0 {
                ExpandedExpr(vec![])
            } else {
                ExpandedExpr(vec![Product { coeff: *n as i64, terms: vec![] }])
            }
        }
        Expr::Var(c) => {
            ExpandedExpr(vec![Product { 
                coeff: 1, 
                terms: vec![Factor::Var(*c, 1)] 
            }])
        }
        Expr::Neg(e) => {
            let expanded = expand_expr(e);
            ExpandedExpr(
                expanded.0.into_iter()
                    .map(|mut p| { p.coeff = -p.coeff; p })
                    .collect()
            )
        }
        Expr::Add(l, r) => {
            let mut left = expand_expr(l);
            let right = expand_expr(r);
            left.0.extend(right.0);
            left
        }
        Expr::Sub(l, r) => {
            let mut left = expand_expr(l);
            let right = expand_expr(r);
            left.0.extend(
                right.0.into_iter()
                    .map(|mut p| { p.coeff = -p.coeff; p })
            );
            left
        }
        Expr::Mul(l, r) => {
            let left = expand_expr(l);
            let right = expand_expr(r);
            
            let mut result = Vec::new();
            for lp in &left.0 {
                for rp in &right.0 {
                    let mut terms = lp.terms.clone();
                    terms.extend(rp.terms.clone());
                    
                    let mut var_powers: HashMap<char, i32> = HashMap::new();
                    for factor in terms {
                        let Factor::Var(c, exp) = factor;
                        *var_powers.entry(c).or_insert(0) += exp;
                    }
                    
                    let mut new_terms: Vec<Factor> = var_powers
                        .into_iter()
                        .filter(|(_, exp)| *exp != 0)
                        .map(|(c, exp)| Factor::Var(c, exp))
                        .collect();
                    new_terms.sort();
                    
                    result.push(Product {
                        coeff: lp.coeff * rp.coeff,
                        terms: new_terms,
                    });
                }
            }
            
            ExpandedExpr(result)
        }
        Expr::Div(l, r) => {
            let left = expand_expr(l);
            let right = expand_expr(r);
            
            let mut result = Vec::new();
            for lp in &left.0 {
                for rp in &right.0 {
                    if rp.coeff == 0 {
                        continue;
                    }
                    
                    fn gcd(mut a: i64, mut b: i64) -> i64 {
                        a = a.abs();
                        b = b.abs();
                        while b != 0 {
                            let t = b;
                            b = a % b;
                            a = t;
                        }
                        a
                    }
                    
                    let g = gcd(lp.coeff, rp.coeff);
                    let num = lp.coeff / g;
                    let den = rp.coeff / g;
                    
                    if den != 1 && den != -1 {
                        return ExpandedExpr(vec![]);
                    }
                    
                    let final_coeff = if den == -1 { -num } else { num };
                    
                    let mut var_powers: HashMap<char, i32> = HashMap::new();
                    
                    for factor in &lp.terms {
                        let Factor::Var(c, exp) = factor;
                        *var_powers.entry(*c).or_insert(0) += exp;
                    }
                    
                    for factor in &rp.terms {
                        let Factor::Var(c, exp) = factor;
                        *var_powers.entry(*c).or_insert(0) -= exp;
                    }
                    
                    let mut new_terms: Vec<Factor> = var_powers
                        .into_iter()
                        .filter(|(_, exp)| *exp != 0)
                        .map(|(c, exp)| Factor::Var(c, exp))
                        .collect();
                    new_terms.sort();
                    
                    result.push(Product {
                        coeff: final_coeff,
                        terms: new_terms,
                    });
                }
            }
            
            ExpandedExpr(result)
        }
        Expr::Pow(base, exp) => {
            let exp_value = match exp.as_ref() {
                Expr::Const(n) => Some(*n),
                Expr::Neg(inner) => {
                    if let Expr::Const(n) = inner.as_ref() {
                        Some(-n)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            
            if let Some(n) = exp_value {
                if n >= 0 && n <= 10 {
                    let mut result = ExpandedExpr(vec![Product { coeff: 1, terms: vec![] }]);
                    let base_expanded = expand_expr(base);
                    
                    for _ in 0..n {
                        let mut new_result = Vec::new();
                        for rp in &result.0 {
                            for bp in &base_expanded.0 {
                                let mut terms = rp.terms.clone();
                                terms.extend(bp.terms.clone());
                                
                                let mut var_powers: HashMap<char, i32> = HashMap::new();
                                for factor in terms {
                                    let Factor::Var(c, exp) = factor;
                                    *var_powers.entry(c).or_insert(0) += exp;
                                }
                                
                                let mut new_terms: Vec<Factor> = var_powers
                                    .into_iter()
                                    .filter(|(_, exp)| *exp != 0)
                                    .map(|(c, exp)| Factor::Var(c, exp))
                                    .collect();
                                new_terms.sort();
                                
                                new_result.push(Product {
                                    coeff: rp.coeff * bp.coeff,
                                    terms: new_terms,
                                });
                            }
                        }
                        result = ExpandedExpr(new_result);
                    }
                    
                    return result;
                } else if n < 0 && n >= -10 {
                    let pos_n = -n;
                    let mut power_result = ExpandedExpr(vec![Product { coeff: 1, terms: vec![] }]);
                    let base_expanded = expand_expr(base);
                    
                    for _ in 0..pos_n {
                        let mut new_result = Vec::new();
                        for rp in &power_result.0 {
                            for bp in &base_expanded.0 {
                                let mut terms = rp.terms.clone();
                                terms.extend(bp.terms.clone());
                                
                                let mut var_powers: HashMap<char, i32> = HashMap::new();
                                for factor in terms {
                                    let Factor::Var(c, exp) = factor;
                                    *var_powers.entry(c).or_insert(0) += exp;
                                }
                                
                                let mut new_terms: Vec<Factor> = var_powers
                                    .into_iter()
                                    .filter(|(_, exp)| *exp != 0)
                                    .map(|(c, exp)| Factor::Var(c, exp))
                                    .collect();
                                new_terms.sort();
                                
                                new_result.push(Product {
                                    coeff: rp.coeff * bp.coeff,
                                    terms: new_terms,
                                });
                            }
                        }
                        power_result = ExpandedExpr(new_result);
                    }
                    
                    let numerator = ExpandedExpr(vec![Product { coeff: 1, terms: vec![] }]);
                    let mut result = Vec::new();
                    
                    for np in &numerator.0 {
                        for dp in &power_result.0 {
                            if dp.coeff == 0 {
                                continue;
                            }
                            
                            let mut var_powers: HashMap<char, i32> = HashMap::new();
                            for factor in &dp.terms {
                                let Factor::Var(c, exp) = factor;
                                *var_powers.entry(*c).or_insert(0) -= exp;
                            }
                            
                            let mut new_terms: Vec<Factor> = var_powers
                                .into_iter()
                                .filter(|(_, exp)| *exp != 0)
                                .map(|(c, exp)| Factor::Var(c, exp))
                                .collect();
                            new_terms.sort();
                            
                            result.push(Product {
                                coeff: np.coeff / dp.coeff,
                                terms: new_terms,
                            });
                        }
                    }
                    
                    return ExpandedExpr(result);
                }
            }
            
            ExpandedExpr(vec![])
        }
    }
}

fn find_shortest_equivalence(
    points: &[Point],
    var_map: &BTreeMap<char, usize>,
    max_total_size: usize,
    logger: &Logger
) -> Option<(Arc<Expr>, Arc<Expr>)> {
    let mut interner = ExprInterner::new();
    let mut value_to_expr: BTreeMap<Vec<i64>, Arc<Expr>> = BTreeMap::new();
    let mut exprs_by_size = ExpressionsBySize::default();
    
    let required_vars: BTreeSet<char> = var_map.keys().copied().collect();
    
    let mut initial_exprs = Vec::new();
    
    let mut vars: Vec<_> = var_map.keys().collect();
    vars.sort();
    for c in vars {
        let expr = interner.mk_var(*c);
        initial_exprs.push(expr);
    }
    
    for n in 0..=9 {
        let expr = interner.mk_const(n);
        initial_exprs.push(expr);
    }
    
    for expr in initial_exprs {
        if let Some(vv) = compute_value_vector(&expr, points, var_map) {
            let norm = normalize_vector(&vv);
            if !value_to_expr.contains_key(&norm) {
                value_to_expr.insert(norm, Arc::clone(&expr));
                exprs_by_size.add(expr);
            }
        }
    }
    
    logger.info(format!("Size 1: {} unique expressions", 
             exprs_by_size.get(1).map_or(0, |v| v.len())));
    
    for size in 2..=max_total_size {
        let mut new_exprs = Vec::new();
        
        if let Some(size_minus_1) = exprs_by_size.get(size - 1) {
            for e in size_minus_1 {
                let expr = interner.mk_neg(Arc::clone(e));
                if let Some(vv) = compute_value_vector(&expr, points, var_map) {
                    let norm = normalize_vector(&vv);
                    
                    if let Some(existing) = value_to_expr.get(&norm) {
                        if existing.as_ref() != expr.as_ref() 
                            && !are_structurally_equivalent(existing, &expr) {
                            
                            if matches!(existing.as_ref(), Expr::Const(_)) && matches!(expr.as_ref(), Expr::Const(_)) {
                                continue;
                            }
                            
                            let existing_vals: Vec<_> = points.iter()
                                .filter_map(|p| eval(existing, p, var_map))
                                .collect();
                            let expr_vals: Vec<_> = points.iter()
                                .filter_map(|p| eval(&expr, p, var_map))
                                .collect();
                            
                            if !existing_vals.is_empty() && !expr_vals.is_empty() {
                                let existing_constant = existing_vals.iter().all(|&v| (v - existing_vals[0]).abs() < 1e-15);
                                let expr_constant = expr_vals.iter().all(|&v| (v - expr_vals[0]).abs() < 1e-15);
                                
                                if existing_constant && expr_constant {
                                    continue;
                                }
                                
                                let has_extreme_existing = existing_vals.iter().any(|&v| v.abs() < 1e-100 || v.abs() > 1e100);
                                let has_extreme_expr = expr_vals.iter().any(|&v| v.abs() < 1e-100 || v.abs() > 1e100);
                                
                                if has_extreme_existing || has_extreme_expr {
                                    continue;
                                }
                            }
                            
                            let total_size = existing.size() + expr.size();
                            if total_size > max_total_size {
                                continue;
                            }
                            
                            if !use_all_variables(existing, &expr, &required_vars) {
                                continue;
                            }
                            
                            let are_alg_equiv = are_algebraically_equivalent(existing, &expr, var_map);
                            
                            logger.debug(format!("Comparing: {} vs {} | Alg equiv: {} | Total: {}", 
                                               existing.to_string(), expr.to_string(), are_alg_equiv, total_size));
                            logger.debug(format!("  Canon1: {}", canonical_to_string(&to_canonical_form(existing))));
                            logger.debug(format!("  Canon2: {}", canonical_to_string(&to_canonical_form(&expr))));
                            
                            if !are_alg_equiv {
                                logger.info(format!("FOUND equivalence with total size {}", total_size));
                                return Some((Arc::clone(existing), expr));
                            }
                        }
                    } else {
                        value_to_expr.insert(norm, Arc::clone(&expr));
                        new_exprs.push(expr);
                    }
                }
            }
        }
        
        for left_size in 1..size {
            let right_size = size - left_size - 1;
            
            if right_size < 1 {
                continue;
            }
            
            let left_exprs = match exprs_by_size.get(left_size) {
                Some(exprs) => exprs,
                None => continue,
            };
            
            let right_exprs = match exprs_by_size.get(right_size) {
                Some(exprs) => exprs,
                None => continue,
            };
            
            for l in left_exprs {
                for r in right_exprs {
                    let candidates = vec![
                        interner.mk_add(Arc::clone(l), Arc::clone(r)),
                        interner.mk_sub(Arc::clone(l), Arc::clone(r)),
                        interner.mk_mul(Arc::clone(l), Arc::clone(r)),
                        interner.mk_div(Arc::clone(l), Arc::clone(r)),
                        interner.mk_pow(Arc::clone(l), Arc::clone(r)),
                    ];
                    
                    for expr_opt in candidates {
                        if let Some(expr) = expr_opt {
                            if expr.size() != size {
                                continue;
                            }
                            
                            if let Some(vv) = compute_value_vector(&expr, points, var_map) {
                                let norm = normalize_vector(&vv);
                                
                                if let Some(existing) = value_to_expr.get(&norm) {
                                    if existing.as_ref() != expr.as_ref()
                                        && !are_structurally_equivalent(existing, &expr) {
                                        
                                        if matches!(existing.as_ref(), Expr::Const(_)) && matches!(expr.as_ref(), Expr::Const(_)) {
                                            continue;
                                        }
                                        
                                        let existing_vals: Vec<_> = points.iter()
                                            .filter_map(|p| eval(existing, p, var_map))
                                            .collect();
                                        let expr_vals: Vec<_> = points.iter()
                                            .filter_map(|p| eval(&expr, p, var_map))
                                            .collect();
                                        
                                        if !existing_vals.is_empty() && !expr_vals.is_empty() {
                                            let existing_constant = existing_vals.iter().all(|&v| (v - existing_vals[0]).abs() < 1e-15);
                                            let expr_constant = expr_vals.iter().all(|&v| (v - expr_vals[0]).abs() < 1e-15);
                                            
                                            if existing_constant && expr_constant {
                                                continue;
                                            }
                                            
                                            let has_extreme_existing = existing_vals.iter().any(|&v| v.abs() < 1e-100 || v.abs() > 1e100);
                                            let has_extreme_expr = expr_vals.iter().any(|&v| v.abs() < 1e-100 || v.abs() > 1e100);
                                            
                                            if has_extreme_existing || has_extreme_expr {
                                                continue;
                                            }
                                        }
                                        
                                        let total_size = existing.size() + expr.size();
                                        if total_size > max_total_size {
                                            continue;
                                        }
                                        
                                        if !use_all_variables(existing, &expr, &required_vars) {
                                            continue;
                                        }
                                        
                                        let are_alg_equiv = are_algebraically_equivalent(existing, &expr, var_map);
                                        
                                        logger.debug(format!("Comparing: {} vs {} | Alg equiv: {} | Total: {}", 
                                                           existing.to_string(), expr.to_string(), are_alg_equiv, total_size));
                                        logger.debug(format!("  Canon1: {}", canonical_to_string(&to_canonical_form(existing))));
                                        logger.debug(format!("  Canon2: {}", canonical_to_string(&to_canonical_form(&expr))));
                                        
                                        if !are_alg_equiv {
                                            logger.info(format!("FOUND equivalence with total size {}", total_size));
                                            return Some((Arc::clone(existing), expr));
                                        }
                                    }
                                } else {
                                    value_to_expr.insert(norm, Arc::clone(&expr));
                                    new_exprs.push(expr);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        for expr in new_exprs {
            exprs_by_size.add(expr);
        }
        
        logger.info(format!("Size {}: {} unique expressions (total: {})", 
                 size, 
                 exprs_by_size.get(size).map_or(0, |v| v.len()),
                 value_to_expr.len()));
    }
    
    None
}

#[wasm_bindgen]
pub struct SearchResult {
    found: bool,
    f_expr: String,
    g_expr: String,
    f_size: usize,
    g_size: usize,
    var_signature: String,
    errors: Vec<f64>,
}

#[wasm_bindgen]
impl SearchResult {
    #[wasm_bindgen(getter)]
    pub fn found(&self) -> bool {
        self.found
    }
    
    #[wasm_bindgen(getter)]
    pub fn f_expr(&self) -> String {
        self.f_expr.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn g_expr(&self) -> String {
        self.g_expr.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn f_size(&self) -> usize {
        self.f_size
    }
    
    #[wasm_bindgen(getter)]
    pub fn g_size(&self) -> usize {
        self.g_size
    }
    
    #[wasm_bindgen(getter)]
    pub fn var_signature(&self) -> String {
        self.var_signature.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn errors(&self) -> Vec<f64> {
        self.errors.clone()
    }
}

#[wasm_bindgen]
pub fn search_equivalence(points_str: &str, max_size: usize, log_fn: &js_sys::Function) -> SearchResult {
    let log_callback = |msg: String| {
        let _ = log_fn.call1(&JsValue::NULL, &JsValue::from_str(&msg));
    };
    let logger = Logger::new(&log_callback);
    
    logger.info(format!("Search started with max_total_size={}", max_size));
    
    let mut points = Vec::new();
    let mut dimension = None;
    
    for line in points_str.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        
        let coords: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        
        if !coords.is_empty() {
            if let Some(dim) = dimension {
                if coords.len() != dim {
                    logger.error("Inconsistent dimensions in input".to_string());
                    return SearchResult {
                        found: false,
                        f_expr: String::new(),
                        g_expr: String::new(),
                        f_size: 0,
                        g_size: 0,
                        var_signature: String::new(),
                        errors: vec![],
                    };
                }
            } else {
                dimension = Some(coords.len());
            }
            points.push(coords);
        }
    }
    
    let dim = dimension.unwrap_or(0);
    
    if points.is_empty() {
        logger.error("No points found in input".to_string());
        return SearchResult {
            found: false,
            f_expr: String::new(),
            g_expr: String::new(),
            f_size: 0,
            g_size: 0,
            var_signature: String::new(),
            errors: vec![],
        };
    }
    
    logger.info(format!("Read {} points of dimension {}", points.len(), dim));
    
    let mut var_map = BTreeMap::new();
    for (i, c) in "xyzwuvabcdefghijklmnopqrst".chars().enumerate() {
        if i < dim {
            var_map.insert(c, i);
        }
    }
    
    logger.info("Starting progressive search".to_string());
    
    for target_size in 2..=max_size {
        logger.info(format!("Searching for equivalences with total size <= {}", target_size));
        
        if let Some((f, g)) = find_shortest_equivalence(&points, &var_map, target_size, &logger) {
            let var_names: Vec<char> = var_map.keys().copied().collect();
            let var_signature = if var_names.len() == 1 {
                format!("({})", var_names[0])
            } else {
                format!("({})", var_names.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", "))
            };
            
            let mut errors = Vec::new();
            for (i, point) in points.iter().enumerate() {
                let f_val = eval(&f, point, &var_map).unwrap_or(f64::NAN);
                let g_val = eval(&g, point, &var_map).unwrap_or(f64::NAN);
                let error = (f_val - g_val).abs();
                errors.push(error);
                logger.debug(format!("Point {}: f={:.6}, g={:.6}, error={:.2e}", i+1, f_val, g_val, error));
            }
            
            return SearchResult {
                found: true,
                f_expr: f.to_string(),
                g_expr: g.to_string(),
                f_size: f.size(),
                g_size: g.size(),
                var_signature,
                errors,
            };
        }
    }
    
    logger.warn(format!("No equivalence found up to size {}", max_size));
    SearchResult {
        found: false,
        f_expr: String::new(),
        g_expr: String::new(),
        f_size: 0,
        g_size: 0,
        var_signature: String::new(),
        errors: vec![],
    }
}
