pub mod ai;
pub mod anomaly;
pub mod assembly;
pub mod autofix;
pub mod core;
pub mod engine;
pub mod layers;
pub mod multi;
pub mod multi_engine;
pub mod physics;
pub mod render;
pub mod reporters;
pub mod sandbox;
pub mod sim;
pub mod spatial;
pub mod weaver;

pub mod fisig_formatter;
pub mod gguf;
pub mod gpu;
pub mod microwave;
pub mod omni;
pub mod patcher;
pub mod quality_filter;
pub mod safety;
pub mod speech;
pub mod text_quality;
pub mod vsa;

pub use ai::latent_jepa::LATENT_DIM;
pub use core::fuga_synthesizer::{BugLocation, FugaResult, FugaSynthesizer};
pub use core::hypervector::Hypervector;
pub use core::information_triangle::InformationTriangle;
pub use core::pentagon_storage::PentagonStorage;
pub use core::wave_cube::WaveCube;

pub use layers::chaos_layer::{
    AttackKind, AttackMetadata, ChaosAnalysis, ChaosAttack, ChaosMutationLayer, ChaosStats,
};
pub use layers::semantic_layer::{AnomalyKind, SemanticAnalysis, SemanticAnomaly, SemanticLayer};
pub use layers::syntax_layer::{
    CodeStats, Severity, SyntaxAnalysisResult, SyntaxInvariantLayer, SyntaxViolation, ViolationKind,
};

pub use engine::{
    AnalysisResult, AttackInfo, FugaEngine, FugaEngineResult, FugaError, LayerResults, SourceStats,
    ViolationInfo,
};

pub use autofix::{
    FixGenerator, FixProposal, FixStrategy, FixValidator, PatchGenerator, UnifiedDiff,
};

pub use reporters::{
    FileAnalysisResult, HtmlReporter, JsonReporter, MarkdownReporter, OutputFormat, Reporter,
    ScanMode, WorkspaceScanner, WorkspaceStats,
};

pub use multi_engine::{MultiEngine, MultiEngineResult};

pub use multi::{
    CodeTranslator, LanguageId, MultiChaosAttack, MultiChaosLayer, MultiChaosResult,
    MultiFixGenerator, MultiSemanticAnomaly, MultiSemanticLayer, MultiSemanticResult,
    MultiSyntaxLayer, MultiSyntaxResult, MultiSyntaxViolation, ViolationPattern,
};

pub use weaver::{
    UnweaveResult, WeaverEngine, WeaverResult,
    explorer::TokenExplorer,
    pattern_matcher::TokenInfo,
    super_token::{SuperToken, TokenRole},
    token_builder::TokenBuilder,
    vocabulary::TokenVocabulary,
};

pub use sim::{Boiler, CubicController, Heater, Phase, Pipe, Valve};

pub use ai::{
    AIOutput, AnomalyDetector, AnomalyEvent,
    AnomalyReflector, AnswerEngine, AnswerHit, AnswerResult, AttentionCell, CodegenResult,
    CorrectionSignal, DecoderOut, DendriteSegment, DynamicRouter, ExpertConfig, FugaAI,
    HierarchicalJEPA, JepaPredictor, LatentPredictor, LatentVector, MemoryEntry, MemoryStore,
    MoEStore,
    PromptVectors, ResonanceAttention, SDR_DENSITY, SDR_DIM, SDR_WORDS, SdrIndex,
    SdrStore, SdrVector, StyloProfile, Synapse, TargetExpert, TemporalCell, TemporalMemory,
    logit_lens,
    crystal::{
        ANCHOR_FLOOR, ANCHOR_RESONANCE_MIN, ANCHOR_TOTAL_MIN, ANCHOR_WEIGHT_FACT,
        ANCHOR_WEIGHT_INTENT, ANCHOR_WEIGHT_LOGIC, CrystalEntry, CrystalHit, DEFAULT_DIM,
        DEFAULT_RESONANCE_THRESHOLD, DIM_L0, DIM_L1, DIM_L2, KIND_L0, KIND_L1, KIND_L2,
        L2_THRESHOLD_SCALE, PhaseCrystal, QueryConfig, ReasoningFoundations, bind_phase,
        dim_for_kind, fnv1a, permute_phase, project_phase,
    },
    domain_sdr, encode_text, byte_basis, encode_bytes_sdr,
    self_mirror::{
        AutoCorrectEngine, AutoCorrectSuggestion, InspectReport, PhaseNode, RawChunk, SelfMirror,
    },
    sparsify,
    temporal_predictor::{TemporalPredictor, sdr_to_hypervector},
    tm_generate::{
        tm_generate, tm_generate_latent, tm_generate_latent_bytes, tm_generate_recurrent,
        tm_generate_two_speed, tm_generate_two_speed_entropy, tm_generate_hybrid,
        tm_generate_megabyte,
        tm_generate_cosine_gate,
        tm_generate_cosine_gate_inner,
    },
    transpile::{
        Dtype, ROUTE_CAP, ShardSource, StTensor, TranspileAccumulator, TranspileConfig,
        WeightSketch, binarize_tensor, hf_resolve_url, is_repo_id, kind_for_name, list_hf_shards,
        list_ms_shards, ms_resolve_url, parse_safetensors_header, shard_label, transpile_shard,
    },
};

pub use quality_filter::{CodeQualityFilter, QualityScore, summarize_quality};
pub use text_quality::{
    TextQualityFilter, TextQualityScore, TextSourceType, extract_dialogue_pairs,
    summarize_text_quality,
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

pub fn tokenize_corpus_text(
    text: &str,
    vocab: &[(u32, String)],
) -> Vec<crate::weaver::pattern_matcher::TokenInfo> {
    use crate::weaver::pattern_matcher::TokenInfo;
    use std::collections::HashMap;

    let word_id_map: HashMap<&str, u32> = vocab.iter().map(|(id, t)| (t.as_str(), *id)).collect();

    text.split_whitespace()
        .enumerate()
        .map(|(_, word)| {
            let id = word_id_map.get(word).copied().unwrap_or_else(|| {
                let h = word
                    .bytes()
                    .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
                100000 + (h % 90000)
            });
            TokenInfo {
                id,
                text: word.to_string(),
            }
        })
        .collect()
}

pub fn load_corpus(path: &str) -> Result<Vec<CorpusDoc>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read corpus: {}", e))?;
    Ok(content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect())
}

#[cfg(test)]
mod zz_reexport_check {
    #[test]
    fn reexport_works() {
        let _: crate::LatentVector;
        let _: crate::LatentPredictor;
    }
}
