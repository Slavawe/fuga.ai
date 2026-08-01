pub mod morris_sandbox;
pub mod self_optimizer;
pub mod legion;
pub mod phase_shield;
pub mod unified_pipeline;

pub use morris_sandbox::{MorrisSandbox, SandboxOutcome};
pub use self_optimizer::{SelfOptimizer, OptimizerConfig, SystemDiagnosis};
pub use legion::{LegionCoordinator, LegionCommand, LegionReport};
pub use phase_shield::{PhaseShield, ShieldAction};
pub use unified_pipeline::{UnifiedPipeline, PipelineConfig, PipelineResult};
