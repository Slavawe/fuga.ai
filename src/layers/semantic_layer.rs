use crate::core::hypervector::Hypervector;
use rand::SeedableRng;
use syn::{
    Attribute, Block, Expr, File, FnArg, Generics, Item, ItemFn, ReturnType, Signature, Stmt, Type,
};

#[derive(Debug, Clone)]
pub struct SemanticAnalysis {
    pub semantic_vector: Hypervector,
    pub function_vectors: Vec<(String, Hypervector)>,
    pub coherence: f64,
    pub anomalies: Vec<SemanticAnomaly>,
}

#[derive(Debug, Clone)]
pub struct SemanticAnomaly {
    pub kind: AnomalyKind,
    pub location: String,
    pub description: String,
    pub severity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnomalyKind {
    SignatureMismatch,
    UnusedParameter,
    MissingBaseCase,
    HighComplexity,
    TypeInvariantViolation,
}

pub struct SemanticLayer {
    pub dim: usize,
}

impl SemanticLayer {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn analyze(&self, file: &File) -> SemanticAnalysis {
        let mut function_vectors = Vec::new();
        let mut anomalies = Vec::new();

        for item in &file.items {
            if let Item::Fn(item_fn) = item {
                let vec = self.encode_function(item_fn);
                let name = item_fn.sig.ident.to_string();
                function_vectors.push((name.clone(), vec.clone()));
                anomalies.extend(self.check_function_semantics(item_fn, &name));
            }
        }

        let semantic_vector = if function_vectors.is_empty() {
            Hypervector::random(self.dim)
        } else {
            let first = &function_vectors[0].1;
            let others: Vec<&Hypervector> =
                function_vectors.iter().skip(1).map(|(_, v)| v).collect();
            first.bundle(&others)
        };

        let coherence = self.compute_coherence(&function_vectors);

        SemanticAnalysis {
            semantic_vector,
            function_vectors,
            coherence,
            anomalies,
        }
    }

    fn encode_function(&self, func: &ItemFn) -> Hypervector {
        let mut comps = Vec::new();
        comps.push(self.encode_string(&func.sig.ident.to_string()));
        comps.push(self.encode_signature(&func.sig));
        if !func.sig.generics.params.is_empty() {
            comps.push(self.encode_generics(&func.sig.generics));
        }
        comps.push(self.encode_attributes(&func.attrs));
        comps.push(self.encode_block(&func.block));
        comps[0].bundle(&comps.iter().skip(1).collect::<Vec<_>>())
    }

    fn encode_signature(&self, sig: &Signature) -> Hypervector {
        let mut comps = Vec::new();
        for input in &sig.inputs {
            if let FnArg::Typed(pat_type) = input {
                comps.push(self.encode_type(&pat_type.ty));
            }
        }
        if let ReturnType::Type(_, ty) = &sig.output {
            comps.push(self.encode_type(ty));
        }
        if comps.is_empty() {
            Hypervector::random(self.dim)
        } else {
            comps[0].bundle(&comps.iter().skip(1).collect::<Vec<_>>())
        }
    }

    fn encode_type(&self, ty: &Type) -> Hypervector {
        match ty {
            Type::Path(tp) => {
                let segs: Vec<_> = tp
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                self.encode_string(&segs.join("::"))
            }
            Type::Reference(r) => {
                let elem = self.encode_type(&r.elem);
                let ref_marker = self.encode_string("&ref");
                elem.bind(&ref_marker)
            }
            Type::Tuple(t) => {
                let elems: Vec<_> = t.elems.iter().map(|e| self.encode_type(e)).collect();
                if elems.is_empty() {
                    Hypervector::random(self.dim)
                } else {
                    elems[0].bundle(&elems.iter().skip(1).collect::<Vec<_>>())
                }
            }
            Type::Array(a) => {
                let elem = self.encode_type(&a.elem);
                elem.bind(&self.encode_string("[]array"))
            }
            Type::Slice(s) => {
                let elem = self.encode_type(&s.elem);
                elem.bind(&self.encode_string("[]slice"))
            }
            _ => Hypervector::random(self.dim),
        }
    }

    fn encode_generics(&self, gens: &Generics) -> Hypervector {
        let names: Vec<_> = gens
            .params
            .iter()
            .map(|p| match p {
                syn::GenericParam::Type(t) => t.ident.to_string(),
                syn::GenericParam::Lifetime(l) => l.lifetime.ident.to_string(),
                syn::GenericParam::Const(c) => c.ident.to_string(),
            })
            .collect();
        self.encode_string(&names.join(","))
    }

    fn encode_attributes(&self, attrs: &[Attribute]) -> Hypervector {
        if attrs.is_empty() {
            return Hypervector::random(self.dim);
        }
        let names: Vec<_> = attrs
            .iter()
            .filter_map(|a| a.path().get_ident().map(|i| i.to_string()))
            .collect();
        self.encode_string(&names.join(","))
    }

    fn encode_block(&self, block: &Block) -> Hypervector {
        let mut comps = Vec::new();
        for stmt in &block.stmts {
            comps.push(self.encode_stmt(stmt));
        }
        if comps.is_empty() {
            Hypervector::random(self.dim)
        } else {
            comps[0].bundle(&comps.iter().skip(1).collect::<Vec<_>>())
        }
    }

    fn encode_stmt(&self, stmt: &Stmt) -> Hypervector {
        match stmt {
            Stmt::Expr(e, _) => self.encode_expr(e),
            Stmt::Local(l) => l
                .init
                .as_ref()
                .map(|init| self.encode_expr(&init.expr))
                .unwrap_or_else(|| Hypervector::random(self.dim)),
            _ => Hypervector::random(self.dim),
        }
    }

    fn encode_expr(&self, expr: &Expr) -> Hypervector {
        match expr {
            Expr::Call(_c) => self.encode_string("call"),
            Expr::MethodCall(m) => self.encode_string(&m.method.to_string()),
            Expr::Binary(_b) => self.encode_string("binary"),
            Expr::If(_) => self.encode_string("if"),
            Expr::Match(_) => self.encode_string("match"),
            Expr::Loop(_) => self.encode_string("loop"),
            Expr::While(_) => self.encode_string("while"),
            Expr::ForLoop(_) => self.encode_string("for"),
            Expr::Return(_) => self.encode_string("return"),
            Expr::Await(_) => self.encode_string("await"),
            Expr::Async(_) => self.encode_string("async"),
            _ => Hypervector::random(self.dim),
        }
    }

    fn encode_string(&self, s: &str) -> Hypervector {
        deterministic_vector(self.dim, s)
    }

    fn check_function_semantics(&self, func: &ItemFn, name: &str) -> Vec<SemanticAnomaly> {
        let mut anomalies = Vec::new();
        let params = func.sig.inputs.len();
        if params > 8 {
            anomalies.push(SemanticAnomaly {
                kind: AnomalyKind::HighComplexity,
                location: name.to_string(),
                description: format!("Function has {} parameters, consider refactoring", params),
                severity: 0.5 + (params as f64 / 20.0).min(0.5),
            });
        }
        anomalies
    }

    fn compute_coherence(&self, funcs: &[(String, Hypervector)]) -> f64 {
        if funcs.len() < 2 {
            return 1.0;
        }
        let mut sims = Vec::new();
        for i in 0..funcs.len() {
            for j in i + 1..funcs.len() {
                sims.push(funcs[i].1.similarity(&funcs[j].1));
            }
        }
        if sims.is_empty() {
            1.0
        } else {
            sims.iter().sum::<f64>() / sims.len() as f64
        }
    }
}

fn deterministic_vector(dim: usize, seed: &str) -> Hypervector {
    use rand::RngCore;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
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
