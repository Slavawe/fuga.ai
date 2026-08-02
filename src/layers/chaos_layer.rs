use crate::core::hypervector::Hypervector;
use rand::SeedableRng;
use std::collections::HashMap;
use syn::{BinOp, Block, Expr, File, Item, ItemFn, Stmt};

#[derive(Debug, Clone)]
pub struct ChaosAnalysis {
    pub attack_vectors: Vec<ChaosAttack>,
    pub stats: ChaosStats,
}

#[derive(Debug, Clone)]
pub struct ChaosAttack {
    pub kind: AttackKind,
    pub vector: Hypervector,
    pub description: String,
    pub priority: f64,
    pub metadata: AttackMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttackKind {
    IntegerOverflow,
    BufferOverflow,
    RaceCondition,
    DivisionByZero,
    NullPointerDeref,
    UseAfterFree,
    DoubleFree,
    UninitializedMemory,
    InfiniteLoop,
    StackOverflow,
    MemoryLeak,
    FormatStringInjection,
    CommandInjection,
    SqlInjection,
    PathTraversal,
    WeakRandomness,
    HardcodedSecrets,
    PanicInDrop,
    UnusedResult,
    Deadlock,
}

#[derive(Debug, Clone, Default)]
pub struct AttackMetadata {
    pub function: Option<String>,
    pub line: Option<usize>,
    pub suggested_input: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ChaosStats {
    pub attacks_generated: usize,
    pub functions_analyzed: usize,
    pub mutation_points: usize,
}

pub struct ChaosMutationLayer {
    dim: usize,
    attack_vectors: HashMap<AttackKind, Hypervector>,
}

impl ChaosMutationLayer {
    pub fn new(dim: usize) -> Self {
        let mut attack_vectors = HashMap::new();
        let kinds = [
            AttackKind::IntegerOverflow,
            AttackKind::BufferOverflow,
            AttackKind::RaceCondition,
            AttackKind::DivisionByZero,
            AttackKind::NullPointerDeref,
            AttackKind::UseAfterFree,
            AttackKind::DoubleFree,
            AttackKind::UninitializedMemory,
            AttackKind::InfiniteLoop,
            AttackKind::StackOverflow,
            AttackKind::MemoryLeak,
            AttackKind::FormatStringInjection,
            AttackKind::CommandInjection,
            AttackKind::SqlInjection,
            AttackKind::PathTraversal,
            AttackKind::WeakRandomness,
            AttackKind::HardcodedSecrets,
            AttackKind::PanicInDrop,
            AttackKind::UnusedResult,
            AttackKind::Deadlock,
        ];
        for kind in &kinds {
            let hv = deterministic_vector(dim, &format!("{:?}", kind));
            attack_vectors.insert(*kind, hv);
        }
        Self {
            dim,
            attack_vectors,
        }
    }

    pub fn analyze(&self, file: &File) -> ChaosAnalysis {
        let mut attacks = Vec::new();
        let mut stats = ChaosStats::default();

        for item in &file.items {
            if let Item::Fn(func) = item {
                stats.functions_analyzed += 1;
                let fn_name = func.sig.ident.to_string();
                let fn_attacks = self.generate_attacks_for_function(func, &fn_name);
                stats.attacks_generated += fn_attacks.len();
                stats.mutation_points += self.count_mutation_points(func);
                attacks.extend(fn_attacks);
            }
        }

        attacks.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        ChaosAnalysis {
            attack_vectors: attacks,
            stats,
        }
    }

    fn count_mutation_points(&self, func: &ItemFn) -> usize {
        func.block
            .stmts
            .iter()
            .map(|s| self.count_mutation_in_stmt(s))
            .sum()
    }

    fn count_mutation_in_stmt(&self, stmt: &Stmt) -> usize {
        match stmt {
            Stmt::Expr(e, _) => self.count_mutation_in_expr(e),
            Stmt::Local(l) => l
                .init
                .as_ref()
                .map(|i| self.count_mutation_in_expr(&i.expr))
                .unwrap_or(0),
            _ => 0,
        }
    }

    fn count_mutation_in_expr(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Binary(_) => 1,
            Expr::If(i) => 1 + self.count_mutation_in_block(&i.then_branch),
            Expr::Loop(l) => self.count_mutation_in_block(&l.body),
            Expr::While(w) => self.count_mutation_in_block(&w.body),
            Expr::ForLoop(f) => self.count_mutation_in_block(&f.body),
            Expr::Match(m) => m
                .arms
                .iter()
                .map(|a| match &a.body.as_ref() {
                    Expr::Block(b) => self.count_mutation_in_block(&b.block),
                    _ => 0,
                })
                .sum(),
            _ => 0,
        }
    }

    fn count_mutation_in_block(&self, block: &Block) -> usize {
        block
            .stmts
            .iter()
            .map(|s| self.count_mutation_in_stmt(s))
            .sum()
    }

    fn generate_attacks_for_function(&self, func: &ItemFn, fn_name: &str) -> Vec<ChaosAttack> {
        let mut attacks = Vec::new();

        // Integer overflow
        let overflow_risk = self.analyze_integer_ops(&func.block);
        if overflow_risk > 0.5 {
            attacks.push(self.create_attack(
                AttackKind::IntegerOverflow,
                &format!("Function {} has integer operations at risk", fn_name),
                overflow_risk,
                AttackMetadata {
                    function: Some(fn_name.into()),
                    ..Default::default()
                },
            ));
        }

        // Division by zero
        let div_risk = self.analyze_divisions(&func.block);
        if div_risk > 0.3 {
            attacks.push(self.create_attack(
                AttackKind::DivisionByZero,
                &format!("Function {} has unchecked division", fn_name),
                div_risk,
                AttackMetadata {
                    function: Some(fn_name.into()),
                    ..Default::default()
                },
            ));
        }

        // Infinite loop
        let loop_risk = self.analyze_loops(&func.block);
        if loop_risk > 0.4 {
            attacks.push(self.create_attack(
                AttackKind::InfiniteLoop,
                &format!("Function {} may have unbounded loops", fn_name),
                loop_risk,
                AttackMetadata {
                    function: Some(fn_name.into()),
                    ..Default::default()
                },
            ));
        }

        attacks
    }

    fn create_attack(
        &self,
        kind: AttackKind,
        desc: &str,
        priority: f64,
        meta: AttackMetadata,
    ) -> ChaosAttack {
        let vector = self
            .attack_vectors
            .get(&kind)
            .cloned()
            .unwrap_or_else(|| Hypervector::random(self.dim));
        ChaosAttack {
            kind,
            vector,
            description: desc.into(),
            priority,
            metadata: meta,
        }
    }

    fn analyze_integer_ops(&self, block: &Block) -> f64 {
        let mut ops = 0;
        for stmt in &block.stmts {
            ops += self.count_int_ops_in_stmt(stmt);
        }
        if ops > 0 { 0.7 } else { 0.0 }
    }

    fn count_int_ops_in_stmt(&self, stmt: &Stmt) -> usize {
        match stmt {
            Stmt::Expr(e, _) => self.count_int_ops_in_expr(e),
            Stmt::Local(l) => l
                .init
                .as_ref()
                .map(|i| self.count_int_ops_in_expr(&i.expr))
                .unwrap_or(0),
            _ => 0,
        }
    }

    fn count_int_ops_in_expr(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Binary(b) => {
                let is_int_op = matches!(
                    b.op,
                    BinOp::Add(_) | BinOp::Sub(_) | BinOp::Mul(_) | BinOp::Rem(_)
                );
                if is_int_op { 1 } else { 0 }
            }
            _ => 0,
        }
    }

    fn analyze_divisions(&self, block: &Block) -> f64 {
        let mut divs = 0;
        for stmt in &block.stmts {
            divs += self.count_divs_in_stmt(stmt);
        }
        if divs > 0 { 0.6 } else { 0.0 }
    }

    fn count_divs_in_stmt(&self, stmt: &Stmt) -> usize {
        match stmt {
            Stmt::Expr(e, _) => self.count_divs_in_expr(e),
            Stmt::Local(l) => l
                .init
                .as_ref()
                .map(|i| self.count_divs_in_expr(&i.expr))
                .unwrap_or(0),
            _ => 0,
        }
    }

    fn count_divs_in_expr(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Binary(b) if matches!(b.op, BinOp::Div(_)) => 1,
            _ => 0,
        }
    }

    fn analyze_loops(&self, block: &Block) -> f64 {
        let mut loops = 0;
        for stmt in &block.stmts {
            loops += self.count_loops_in_stmt(stmt);
        }
        if loops > 0 { 0.5 } else { 0.0 }
    }

    fn count_loops_in_stmt(&self, stmt: &Stmt) -> usize {
        match stmt {
            Stmt::Expr(Expr::Loop(_), _) => 1,
            Stmt::Expr(Expr::While(_), _) => 1,
            Stmt::Expr(Expr::ForLoop(_), _) => 1,
            _ => 0,
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
