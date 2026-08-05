pub mod agent;
pub mod anomaly;
pub mod answer_engine;
pub mod autonomous_mind;
pub mod codegen;
pub mod core;
pub mod decoder;
pub mod hierarchical_jepa;
pub mod hnsw;
pub mod htm_temporal;
pub mod jepa;
pub mod memory_store;
pub mod moe;
pub mod mentalese;
pub mod latent_jepa;
pub mod personas;
pub mod predictive_coder;
pub mod prompts;
pub mod resonance_attention;
pub mod router;
pub mod sdr;
pub mod sdr_store;
pub mod state_loader;
pub mod unified_mind;
pub mod unity_mind;
pub mod wave_mesh;
pub mod world;

pub use agent::{
    ActOptions, AgentBrain, AgentPaths, AgentTick, Episode, Outcome, PainAvoidance,
    PredictiveAgentTick, SafeOutcome, ThinkOutcome, act_and_observe, act_and_observe_at,
    act_and_observe_with_prediction, generate_code, generate_safe, inject_noise_sdr, observe,
};
pub use answer_engine::{AnswerEngine, AnswerHit, AnswerResult};
pub use codegen::CodegenResult;
pub use core::{AIOutput, FugaAI};
pub use decoder::{DecoderOut, logit_lens};
pub use hierarchical_jepa::HierarchicalJEPA;
pub use htm_temporal::{DendriteSegment, Synapse, TemporalCell, TemporalMemory, TrainStats};
pub use latent_jepa::LATENT_DIM;
pub use jepa::JepaPredictor;
pub use latent_jepa::{LatentPredictor, LatentVector};
pub use memory_store::{AttractorIndex, MemoryEntry, MemoryStore, NUM_ATTRACTORS};
pub use moe::MoEStore;
pub use prompts::PromptVectors;
pub use resonance_attention::{AttentionCell, ResonanceAttention};
pub use router::{DynamicRouter, ExpertConfig, TargetExpert};
pub use sdr::{
    SDR_DENSITY, SDR_DIM, SDR_WORDS, SdrIndex, SdrVector, domain_sdr, encode_text, sparsify,
    structure_sdr,
};
pub use sdr_store::SdrStore;
pub mod crystal;
pub mod self_mirror;
pub mod temporal_predictor;
pub mod transpile;
pub use anomaly::{
    AnomalyDetector, AnomalyEvent, AnomalyReflector, CorrectionSignal, StyloProfile,
};
pub use autonomous_mind::{AutonomousMind, MindPaths};
pub use crystal::{
    ANCHOR_FLOOR, ANCHOR_RESONANCE_MIN, ANCHOR_TOTAL_MIN, ANCHOR_WEIGHT_FACT, ANCHOR_WEIGHT_INTENT,
    ANCHOR_WEIGHT_LOGIC, CRYSTAL_MAGIC, CRYSTAL_VERSION, CrystalEntry, CrystalHit, DEFAULT_DIM,
    DEFAULT_RESONANCE_THRESHOLD, DIM_L0, DIM_L1, DIM_L2, KIND_L0, KIND_L1, KIND_L2,
    L2_THRESHOLD_SCALE, PhaseCrystal, QueryConfig, ReasoningFoundations, bind_phase, dim_for_kind,
    fnv1a, permute_phase, project_phase,
};
pub use predictive_coder::PredictiveCoder;
pub use state_loader::{MindState, MindStateLoader};
pub use transpile::{
    CONCEPT_L0, CONCEPT_L1, CONCEPT_L2, Dtype, ROUTE_CAP, ShardSource, StTensor,
    TranspileAccumulator, TranspileConfig, TranspileStats, WeightSketch, binarize_tensor,
    hf_resolve_url, kind_for_name, list_hf_shards, parse_safetensors_header, transpile_shard,
};
pub use unified_mind::{UnifiedMind, UnifiedResult};
pub use unity_mind::UnityMind;

pub mod soft_sdr;
