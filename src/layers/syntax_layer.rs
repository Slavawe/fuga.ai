use rand::SeedableRng;
use syn::visit::Visit;
use syn::{BinOp, Expr, ExprBinary, ExprCall, ExprLit, ExprMethodCall, ExprUnsafe, ItemEnum, ItemFn, ItemStruct, Lit, Macro};
use std::collections::HashMap;
use crate::core::hypervector::Hypervector;

/// Результат работы Синтаксического слоя
#[derive(Debug, Clone)]
pub struct SyntaxAnalysisResult {
    pub safety_score: f64,
    pub violation_vector: Hypervector,
    pub violations: Vec<SyntaxViolation>,
    pub stats: CodeStats,
}

#[derive(Debug, Clone)]
pub struct SyntaxViolation {
    pub kind: ViolationKind,
    pub location: String,
    pub severity: Severity,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ViolationKind {
    UnsafeBlock,
    UnwrapExpect,
    IntegerOverflow,
    ArrayIndexOutOfBounds,
    RecursionDepth,
    InfiniteLoop,
    DivisionByZero,
    NullPointerDeref,
    DataRace,
    BufferOverflow,
    FormatString,
    CommandInjection,
    SqlInjection,
    WeakRandom,
    HardcodedSecret,
    PanicInDrop,
    NonExhaustiveMatch,
    UnusedMustUse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone, Default)]
pub struct CodeStats {
    pub functions: usize,
    pub unsafe_blocks: usize,
    pub unwrap_calls: usize,
    pub loops: usize,
    pub match_expressions: usize,
    pub async_functions: usize,
    pub generic_functions: usize,
    pub public_functions: usize,
}

/// Слой 1: Синтаксический контроллер (Строгий закон)
pub struct SyntaxInvariantLayer {
    dim: usize,
    violation_vectors: HashMap<ViolationKind, Hypervector>,
}

impl SyntaxInvariantLayer {
    pub fn new(dim: usize) -> Self {
        let mut violation_vectors = HashMap::new();
        let kinds = [
            ViolationKind::UnsafeBlock,
            ViolationKind::UnwrapExpect,
            ViolationKind::IntegerOverflow,
            ViolationKind::ArrayIndexOutOfBounds,
            ViolationKind::RecursionDepth,
            ViolationKind::InfiniteLoop,
            ViolationKind::DivisionByZero,
            ViolationKind::NullPointerDeref,
            ViolationKind::DataRace,
            ViolationKind::BufferOverflow,
            ViolationKind::FormatString,
            ViolationKind::CommandInjection,
            ViolationKind::SqlInjection,
            ViolationKind::WeakRandom,
            ViolationKind::HardcodedSecret,
            ViolationKind::PanicInDrop,
            ViolationKind::NonExhaustiveMatch,
            ViolationKind::UnusedMustUse,
        ];
        for kind in &kinds {
            let hv = deterministic_vector(dim, &format!("{:?}", kind));
            violation_vectors.insert(kind.clone(), hv);
        }
        Self { dim, violation_vectors }
    }

    pub fn analyze(&self, source: &str) -> Result<SyntaxAnalysisResult, syn::Error> {
        let syntax_tree = syn::parse_file(source)?;
        let mut visitor = SyntaxVisitor::new(self.dim, &self.violation_vectors);
        visitor.visit_file(&syntax_tree);
        Ok(visitor.into_result())
    }
}

fn deterministic_vector(dim: usize, seed: &str) -> Hypervector {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use rand::RngCore;
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut rng = rand::rngs::StdRng::seed_from_u64(hasher.finish());
    let word_count = (dim + 63) / 64;
    let mut words = vec![0u64; word_count];
    for w in &mut words {
        *w = rng.next_u64();
    }
    let rem = dim % 64;
    if rem != 0 {
        words[word_count - 1] &= (1u64 << rem) - 1;
    }
    Hypervector { dim, words }
}

struct SyntaxVisitor {
    dim: usize,
    violation_vectors: HashMap<ViolationKind, Hypervector>,
    violations: Vec<SyntaxViolation>,
    pub stats: CodeStats,
    current_function: Option<String>,
    unsafe_depth: usize,
}

impl SyntaxVisitor {
    fn new(dim: usize, violation_vectors: &HashMap<ViolationKind, Hypervector>) -> Self {
        Self {
            dim,
            violation_vectors: violation_vectors.clone(),
            violations: Vec::new(),
            stats: CodeStats::default(),
            current_function: None,
            unsafe_depth: 0,
        }
    }

    fn into_result(self) -> SyntaxAnalysisResult {
        let violation_vector = self.build_violation_vector();
        let safety_score = self.compute_safety_score();
        SyntaxAnalysisResult {
            safety_score,
            violation_vector,
            violations: self.violations,
            stats: self.stats,
        }
    }

    fn build_violation_vector(&self) -> Hypervector {
        if self.violations.is_empty() {
            return Hypervector::random(self.dim);
        }
        let vecs: Vec<&Hypervector> = self.violations.iter()
            .filter_map(|v| self.violation_vectors.get(&v.kind))
            .collect();
        if vecs.is_empty() {
            return Hypervector::random(self.dim);
        }
        vecs[0].bundle(&vecs[1..])
    }

    fn compute_safety_score(&self) -> f64 {
        if self.violations.is_empty() { return 1.0; }
        let critical = self.violations.iter().filter(|v| v.severity == Severity::Critical).count();
        let high = self.violations.iter().filter(|v| v.severity == Severity::High).count();
        let medium = self.violations.iter().filter(|v| v.severity == Severity::Medium).count();
        let low = self.violations.iter().filter(|v| v.severity == Severity::Low).count();
        (1.0 - (critical as f64 * 0.4 + high as f64 * 0.2 + medium as f64 * 0.1 + low as f64 * 0.05)).max(0.0)
    }

    fn add_violation(&mut self, kind: ViolationKind, severity: Severity, msg: &str) {
        let loc = self.current_function.clone().unwrap_or_else(|| "top-level".into());
        self.violations.push(SyntaxViolation { kind, location: loc, severity, message: msg.into() });
    }
}

impl<'ast> Visit<'ast> for SyntaxVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.stats.functions += 1;
        if node.sig.asyncness.is_some() { self.stats.async_functions += 1; }
        if !node.sig.generics.params.is_empty() { self.stats.generic_functions += 1; }
        if matches!(node.vis, syn::Visibility::Public(_)) { self.stats.public_functions += 1; }

        let prev = self.current_function.take();
        self.current_function = Some(node.sig.ident.to_string());
        syn::visit::visit_item_fn(self, node);
        self.current_function = prev;
    }

    fn visit_expr_unsafe(&mut self, node: &'ast ExprUnsafe) {
        self.stats.unsafe_blocks += 1;
        self.unsafe_depth += 1;
        self.add_violation(
            ViolationKind::UnsafeBlock,
            Severity::Critical,
            &format!("Unsafe block (depth {})", self.unsafe_depth),
        );
        syn::visit::visit_expr_unsafe(self, node);
        self.unsafe_depth -= 1;
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(path) = &*node.func {
            if let Some(seg) = path.path.segments.last() {
                let name = seg.ident.to_string();
                if name == "unwrap" || name == "expect" {
                    self.stats.unwrap_calls += 1;
                    self.add_violation(
                        ViolationKind::UnwrapExpect,
                        Severity::High,
                        &format!("Use of {}() can panic", name),
                    );
                }
            }
        }
        syn::visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        let name = node.method.to_string();
        if name == "unwrap" || name == "expect" {
            self.stats.unwrap_calls += 1;
            self.add_violation(
                ViolationKind::UnwrapExpect,
                Severity::High,
                &format!("Use of {}() can panic", name),
            );
        }
        syn::visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_loop(&mut self, node: &'ast syn::ExprLoop) {
        self.stats.loops += 1;
        self.add_violation(ViolationKind::InfiniteLoop, Severity::Medium, "Unconditional loop");
        syn::visit::visit_expr_loop(self, node);
    }

    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        self.stats.loops += 1;
        syn::visit::visit_expr_while(self, node);
    }

    fn visit_expr_for_loop(&mut self, node: &'ast syn::ExprForLoop) {
        self.stats.loops += 1;
        syn::visit::visit_expr_for_loop(self, node);
    }

    fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
        self.stats.match_expressions += 1;
        syn::visit::visit_expr_match(self, node);
    }

    fn visit_expr_binary(&mut self, node: &'ast ExprBinary) {
        if matches!(node.op, BinOp::Div(_) | BinOp::Rem(_)) {
            if let Expr::Lit(ExprLit { lit: Lit::Int(lit), .. }) = &*node.right {
                if lit.base10_digits() == "0" {
                    self.add_violation(ViolationKind::DivisionByZero, Severity::Critical, "Division by zero literal");
                }
            }
        }
        syn::visit::visit_expr_binary(self, node);
    }

    fn visit_expr_index(&mut self, node: &'ast syn::ExprIndex) {
        self.add_violation(
            ViolationKind::ArrayIndexOutOfBounds,
            Severity::Medium,
            "Array indexing — bounds not statically verified",
        );
        syn::visit::visit_expr_index(self, node);
    }

    fn visit_macro(&mut self, mac: &'ast Macro) {
        let path_str = quote::quote!(#mac).to_string().to_lowercase();
        if path_str.contains("sql") || path_str.contains("query") {
            self.add_violation(ViolationKind::SqlInjection, Severity::High, "Possible SQL injection vector");
        }
        if path_str.contains("command") || path_str.contains("exec") || path_str.contains("spawn") {
            self.add_violation(ViolationKind::CommandInjection, Severity::High, "Possible command injection");
        }
        if path_str.contains("format") {
            self.add_violation(ViolationKind::FormatString, Severity::Low, "Format macro usage");
        }
        syn::visit::visit_macro(self, mac);
    }

    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        if node.attrs.iter().any(|a| a.path().is_ident("non_exhaustive")) {
            self.add_violation(ViolationKind::NonExhaustiveMatch, Severity::Low, "Non-exhaustive struct");
        }
        syn::visit::visit_item_struct(self, node);
    }

    fn visit_item_enum(&mut self, node: &'ast ItemEnum) {
        if node.attrs.iter().any(|a| a.path().is_ident("non_exhaustive")) {
            self.add_violation(ViolationKind::NonExhaustiveMatch, Severity::Low, "Non-exhaustive enum");
        }
        syn::visit::visit_item_enum(self, node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntax_layer_safe_code() {
        let layer = SyntaxInvariantLayer::new(10000);
        let code = r#"fn add(a: i32, b: i32) -> i32 { a + b }"#;
        let result = layer.analyze(code).unwrap();
        assert!(result.safety_score > 0.8, "got {}", result.safety_score);
    }

    #[test]
    fn test_syntax_layer_unsafe_code() {
        let layer = SyntaxInvariantLayer::new(10000);
        let code = r#"fn f() { unsafe { let _ = *(0 as *const i32); } }"#;
        let result = layer.analyze(code).unwrap();
        assert!(result.safety_score < 0.9, "got {}", result.safety_score);
        assert!(result.violations.iter().any(|v| v.kind == ViolationKind::UnsafeBlock));
    }

    #[test]
    fn test_syntax_layer_unwrap() {
        let layer = SyntaxInvariantLayer::new(10000);
        let code = r#"fn main() { let x: Option<i32> = Some(1); x.unwrap(); }"#;
        let result = layer.analyze(code).unwrap();
        assert!(result.stats.unwrap_calls > 0);
        assert!(result.violations.iter().any(|v| v.kind == ViolationKind::UnwrapExpect));
    }
}