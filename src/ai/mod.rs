pub mod resonance_attention;
pub mod router;
pub mod core;
pub mod memory_store;
pub mod answer_engine;
pub mod codegen;
pub mod moe;
pub mod hnsw;
pub mod jepa;
pub mod hierarchical_jepa;
pub mod prompts;
pub mod world;
pub mod wave_mesh;
pub mod sdr;
pub mod sdr_store;
pub mod htm_temporal;
pub mod anomaly;

pub use codegen::CodegenResult;
pub use resonance_attention::{ResonanceAttention, AttentionCell};
pub use router::{DynamicRouter, TargetExpert, ExpertConfig};
pub use core::{FugaAI, AIOutput};
pub use memory_store::{MemoryStore, MemoryEntry, AttractorIndex, NUM_ATTRACTORS};
pub use moe::MoEStore;
pub use answer_engine::{AnswerEngine, AnswerResult, AnswerHit};
pub use jepa::JepaPredictor;
pub use hierarchical_jepa::HierarchicalJEPA;
pub use prompts::PromptVectors;
pub use sdr::{SdrVector, SdrIndex, sparsify, encode_text, domain_sdr, SDR_DIM, SDR_DENSITY, SDR_WORDS};
pub use sdr_store::SdrStore;
pub use htm_temporal::{TemporalMemory, TemporalCell, DendriteSegment, Synapse};
pub mod temporal_predictor;
pub mod self_mirror;
pub mod crystal;
pub mod transpile;
pub use anomaly::{AnomalyDetector, AnomalyEvent, AnomalyReflector, CorrectionSignal, StyloProfile};
pub use crystal::{PhaseCrystal, CrystalHit, CrystalEntry, CRYSTAL_MAGIC, CRYSTAL_VERSION, DEFAULT_DIM, DEFAULT_RESONANCE_THRESHOLD, KIND_L0, KIND_L1, KIND_L2, fnv1a};
pub use transpile::{
    TranspileAccumulator, TranspileConfig, TranspileStats, ShardSource, StTensor, Dtype,
    WeightSketch, parse_safetensors_header, binarize_tensor, kind_for_name, transpile_shard,
    list_hf_shards, hf_resolve_url, ROUTE_CAP, CONCEPT_L0, CONCEPT_L1, CONCEPT_L2,
};
