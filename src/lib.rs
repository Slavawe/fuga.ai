pub mod core;
pub mod sandbox;
pub mod sim;
pub mod spatial;
pub mod render;
pub mod physics;
pub mod layers;
pub mod engine;
pub mod autofix;
pub mod reporters;
pub mod multi;
pub mod multi_engine;
pub mod weaver;
pub mod ai;
pub mod quality_filter;
pub mod text_quality;
pub mod fisig_formatter;
pub mod omni;
pub mod speech;
pub mod microwave;
pub mod gpu;

pub use core::hypervector::Hypervector;
pub use core::information_triangle::InformationTriangle;
pub use core::pentagon_storage::PentagonStorage;
pub use core::wave_cube::WaveCube;
pub use core::fuga_synthesizer::{FugaResult, FugaSynthesizer, BugLocation};

pub use layers::syntax_layer::{
    SyntaxInvariantLayer, SyntaxAnalysisResult, SyntaxViolation,
    ViolationKind, Severity, CodeStats,
};
pub use layers::semantic_layer::{
    SemanticLayer, SemanticAnalysis, SemanticAnomaly, AnomalyKind,
};
pub use layers::chaos_layer::{
    ChaosMutationLayer, ChaosAnalysis, ChaosAttack,
    AttackKind, AttackMetadata, ChaosStats,
};

pub use engine::{
    FugaEngine, FugaEngineResult, AnalysisResult, ViolationInfo, AttackInfo,
    LayerResults, FugaError, SourceStats,
};

pub use autofix::{
    FixGenerator, FixValidator, PatchGenerator,
    FixStrategy, FixProposal, UnifiedDiff,
};

pub use reporters::{
    Reporter, OutputFormat, FileAnalysisResult, WorkspaceStats,
    JsonReporter, HtmlReporter, MarkdownReporter,
    WorkspaceScanner, ScanMode,
};

pub use multi_engine::{MultiEngine, MultiEngineResult};

pub use multi::{
    LanguageId, MultiSyntaxLayer, MultiSyntaxResult, MultiSyntaxViolation,
    MultiSemanticLayer, MultiSemanticResult, MultiSemanticAnomaly,
    MultiChaosLayer, MultiChaosResult, MultiChaosAttack,
    ViolationPattern, MultiFixGenerator, CodeTranslator,
};

pub use weaver::{
    WeaverEngine, WeaverResult, UnweaveResult,
    super_token::{SuperToken, TokenRole},
    token_builder::TokenBuilder,
    pattern_matcher::TokenInfo,
    vocabulary::TokenVocabulary,
    explorer::TokenExplorer,
};

pub use sim::{
    CubicController, Pipe, Valve, Heater, Boiler, Phase,
};

pub use ai::{
    FugaAI, AIOutput, ResonanceAttention, AttentionCell,
    DynamicRouter, TargetExpert, ExpertConfig,
    MemoryStore, MemoryEntry, MoEStore,
    AnswerEngine, AnswerResult, AnswerHit,
    CodegenResult, JepaPredictor, HierarchicalJEPA, PromptVectors,
};

pub use quality_filter::{
    CodeQualityFilter, QualityScore, summarize_quality,
};
pub use text_quality::{
    TextQualityFilter, TextQualityScore, TextSourceType,
    extract_dialogue_pairs, summarize_text_quality,
};

use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct CorpusDoc {
    pub title: Option<String>,
    pub author: Option<String>,
    pub language: Option<String>,
    pub chapters: Vec<CorpusChapter>,
}

#[derive(Deserialize, Clone)]
pub struct CorpusChapter {
    pub heading: Option<String>,
    pub paragraphs: Vec<String>,
    pub number: Option<u64>,
}

pub fn tokenize_corpus_text(text: &str, vocab: &[(u32, String)]) -> Vec<crate::weaver::pattern_matcher::TokenInfo> {
    use std::collections::HashMap;
    use crate::weaver::pattern_matcher::TokenInfo;

    let word_id_map: HashMap<&str, u32> = vocab.iter()
        .map(|(id, t)| (t.as_str(), *id))
        .collect();

    text.split_whitespace().enumerate().map(|(_, word)| {
        let id = word_id_map.get(word).copied().unwrap_or_else(|| {
            let h = word.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
            100000 + (h % 90000)
        });
        TokenInfo { id, text: word.to_string() }
    }).collect()
}

pub fn load_corpus(path: &str) -> Result<Vec<CorpusDoc>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read corpus: {}", e))?;
    Ok(content.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect())
}

