pub mod legion;
pub mod morris_sandbox;
pub mod phase_shield;
pub mod self_optimizer;
pub mod unified_pipeline;

pub use legion::{LegionCommand, LegionCoordinator, LegionReport};
pub use morris_sandbox::{MorrisSandbox, SandboxOutcome};
pub use phase_shield::{PhaseShield, ShieldAction};
pub use self_optimizer::{OptimizerConfig, SelfOptimizer, SystemDiagnosis};
pub use unified_pipeline::{PipelineConfig, PipelineResult, UnifiedPipeline};
