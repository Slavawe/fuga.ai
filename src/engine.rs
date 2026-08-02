use crate::autofix::{FixGenerator, FixProposal};
use crate::core::fuga_synthesizer::{FugaResult, FugaSynthesizer};
use crate::core::information_triangle::InformationTriangle;
use crate::core::pentagon_storage::PentagonStorage;
use crate::core::wave_cube::WaveCube;
use crate::layers::{
    ChaosAnalysis, ChaosMutationLayer, SemanticAnalysis, SemanticLayer, SyntaxAnalysisResult,
    SyntaxInvariantLayer,
};
use crate::multi_engine::MultiEngineResult;
use syn::File;

#[derive(Debug, Clone)]
pub struct LayerResults {
    pub syntax: SyntaxAnalysisResult,
    pub semantic: SemanticAnalysis,
    pub chaos: ChaosAnalysis,
}

pub struct FugaEngine {
    dim: usize,
    syntax_layer: SyntaxInvariantLayer,
    semantic_layer: SemanticLayer,
    chaos_layer: ChaosMutationLayer,
    synthesizer: FugaSynthesizer,
    cube: WaveCube<3, 4>,
    triangle: InformationTriangle,
    pentagon: PentagonStorage,
    fix_generator: FixGenerator,
}

impl FugaEngine {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            syntax_layer: SyntaxInvariantLayer::new(dim),
            semantic_layer: SemanticLayer::new(dim),
            chaos_layer: ChaosMutationLayer::new(dim),
            synthesizer: FugaSynthesizer::new(dim),
            cube: WaveCube::<3, 4>::new(dim),
            triangle: InformationTriangle::new(dim),
            pentagon: PentagonStorage::new(dim, 0.6),
            fix_generator: FixGenerator::new(dim),
        }
    }

    pub fn analyze(&mut self, source: &str) -> Result<FugaEngineResult, FugaError> {
        let file: File =
            syn::parse_file(source).map_err(|e| FugaError::ParseError(e.to_string()))?;

        let syntax_result = self
            .syntax_layer
            .analyze(source)
            .map_err(|e| FugaError::LayerError(format!("Syntax layer: {}", e)))?;

        let semantic_result = self.semantic_layer.analyze(&file);
        let chaos_result = self.chaos_layer.analyze(&file);

        let best_attack = chaos_result
            .attack_vectors
            .first()
            .map(|a| a.vector.clone());
        let fuga_result = best_attack.map(|vector| {
            self.synthesizer
                .synthesize(&mut self.cube, &mut self.triangle, &self.pentagon, &vector)
        });

        let stats = SourceStats::from_source(source, syntax_result.stats.functions);

        Ok(FugaEngineResult {
            layer_results: LayerResults {
                syntax: syntax_result,
                semantic: semantic_result,
                chaos: chaos_result,
            },
            fuga_result,
            source_stats: stats,
            cube_entropy: self.cube.global_entropy(),
        })
    }

    pub fn analyze_file(&mut self, path: &str) -> Result<FugaEngineResult, FugaError> {
        let source =
            std::fs::read_to_string(path).map_err(|e| FugaError::IoError(e.to_string()))?;
        self.analyze(&source)
    }

    pub fn generate_fixes(&mut self, source: &str, result: &FugaEngineResult) -> Vec<FixProposal> {
        let priority = result
            .layer_results
            .chaos
            .attack_vectors
            .iter()
            .map(|a| a.priority)
            .fold(0.0, f64::max);
        self.fix_generator.generate_fixes(
            source,
            &FugaResult {
                bug_detected: result
                    .layer_results
                    .chaos
                    .attack_vectors
                    .iter()
                    .any(|a| a.priority > 0.6),
                confidence: priority,
                bug_vector: result
                    .layer_results
                    .chaos
                    .attack_vectors
                    .first()
                    .map(|a| a.vector.clone()),
                bug_location: None,
                counterpoint_description: result
                    .layer_results
                    .chaos
                    .attack_vectors
                    .first()
                    .map(|a| a.description.clone())
                    .unwrap_or_default(),
            },
            &result.layer_results.syntax.violations,
        )
    }
}

#[derive(Debug, Clone)]
pub struct FugaEngineResult {
    pub layer_results: LayerResults,
    pub fuga_result: Option<FugaResult>,
    pub source_stats: SourceStats,
    pub cube_entropy: f64,
}

#[derive(Debug, Clone)]
pub enum AnalysisResult {
    Rust(FugaEngineResult),
    Multi(MultiEngineResult),
}

#[derive(Debug, Clone)]
pub struct ViolationItem {
    pub severity: String,
    pub kind: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct AttackItem {
    pub kind: String,
    pub priority: f64,
    pub description: String,
}

impl AnalysisResult {
    pub fn lines(&self) -> usize {
        match self {
            AnalysisResult::Rust(r) => r.source_stats.lines,
            AnalysisResult::Multi(r) => r.source_stats.lines,
        }
    }
    pub fn functions(&self) -> usize {
        match self {
            AnalysisResult::Rust(r) => r.source_stats.functions,
            AnalysisResult::Multi(r) => r.source_stats.functions,
        }
    }
    pub fn violations_count(&self) -> usize {
        self.violations().len()
    }
    pub fn attacks_count(&self) -> usize {
        self.attacks().len()
    }
    pub fn bug_detected(&self) -> bool {
        match self {
            AnalysisResult::Rust(r) => r
                .layer_results
                .chaos
                .attack_vectors
                .iter()
                .any(|a| a.priority > 0.6),
            AnalysisResult::Multi(r) => r.chaos.attack_vectors.iter().any(|a| a.priority > 0.6),
        }
    }
    pub fn safety_score(&self) -> f64 {
        match self {
            AnalysisResult::Rust(r) => {
                1.0 - r
                    .layer_results
                    .chaos
                    .attack_vectors
                    .iter()
                    .map(|a| a.priority)
                    .fold(0.0, f64::max)
            }
            AnalysisResult::Multi(r) => {
                1.0 - r
                    .chaos
                    .attack_vectors
                    .iter()
                    .map(|a| a.priority)
                    .fold(0.0, f64::max)
            }
        }
    }
    pub fn violations(&self) -> Vec<ViolationItem> {
        match self {
            AnalysisResult::Rust(r) => r
                .layer_results
                .syntax
                .violations
                .iter()
                .map(|v| ViolationItem {
                    severity: format!("{:?}", v.severity),
                    kind: format!("{:?}", v.kind),
                    message: v.message.clone(),
                })
                .collect(),
            AnalysisResult::Multi(r) => r
                .syntax
                .violations
                .iter()
                .map(|v| ViolationItem {
                    severity: format!("{:?}", v.severity),
                    kind: format!("{:?}", v.pattern),
                    message: v.message.clone(),
                })
                .collect(),
        }
    }
    pub fn attacks(&self) -> Vec<AttackItem> {
        match self {
            AnalysisResult::Rust(r) => r
                .layer_results
                .chaos
                .attack_vectors
                .iter()
                .map(|a| AttackItem {
                    kind: format!("{:?}", a.kind),
                    priority: a.priority,
                    description: a.description.clone(),
                })
                .collect(),
            AnalysisResult::Multi(r) => r
                .chaos
                .attack_vectors
                .iter()
                .map(|a| AttackItem {
                    kind: format!("{:?}", a.pattern),
                    priority: a.priority,
                    description: a.description.clone(),
                })
                .collect(),
        }
    }
    pub fn anomalies_count(&self) -> usize {
        match self {
            AnalysisResult::Rust(r) => r.layer_results.semantic.anomalies.len(),
            AnalysisResult::Multi(r) => r.semantic.anomalies.len(),
        }
    }
    pub fn coherence(&self) -> f64 {
        match self {
            AnalysisResult::Rust(r) => r.cube_entropy,
            AnalysisResult::Multi(r) => r
                .semantic
                .semantic_vector
                .similarity(&r.syntax.violation_vector),
        }
    }
    pub fn bug_confidence(&self) -> f64 {
        match self {
            AnalysisResult::Rust(r) => r
                .layer_results
                .chaos
                .attack_vectors
                .iter()
                .map(|a| a.priority)
                .fold(0.0, f64::max),
            AnalysisResult::Multi(r) => r
                .chaos
                .attack_vectors
                .iter()
                .map(|a| a.priority)
                .fold(0.0, f64::max),
        }
    }
    pub fn violations_is_empty(&self) -> bool {
        self.violations().is_empty()
    }
    pub fn counterpoint_description(&self) -> String {
        match self {
            AnalysisResult::Rust(r) => r
                .layer_results
                .chaos
                .attack_vectors
                .first()
                .map(|a| a.description.clone())
                .unwrap_or_default(),
            AnalysisResult::Multi(r) => r
                .chaos
                .attack_vectors
                .first()
                .map(|a| a.description.clone())
                .unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ViolationInfo {
    pub line: usize,
    pub kind: String,
    pub severity: f64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct AttackInfo {
    pub pattern: String,
    pub confidence: f64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum FugaError {
    ParseError(String),
    LayerError(String),
    IoError(String),
}

impl std::fmt::Display for FugaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FugaError::ParseError(e) => write!(f, "Parse error: {}", e),
            FugaError::LayerError(e) => write!(f, "Layer error: {}", e),
            FugaError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SourceStats {
    pub lines: usize,
    pub chars: usize,
    pub functions: usize,
}

impl SourceStats {
    pub fn from_source(source: &str, functions: usize) -> Self {
        Self {
            lines: source.lines().count(),
            chars: source.chars().count(),
            functions,
        }
    }
}
