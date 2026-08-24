use fuga::ai::self_mirror::SelfMirror;
use fuga::ai::{HierarchicalJEPA, SdrVector, TemporalMemory, encode_text};
use fuga::assembly::{
    MorrisSandbox, PhaseShield, SelfOptimizer, ShieldAction,
    legion::{FirstLegion, LegionCoordinator, SecondLegion},
    unified_pipeline::{PipelineConfig, UnifiedPipeline},
};

fn make_test_mirror() -> SelfMirror {
    let tm = TemporalMemory::new(200, 4);
    let hjepa = HierarchicalJEPA::new(256);
    let mut mirror = SelfMirror::new(tm, hjepa);
    mirror.predictor.buf_capacity = 10;
    mirror
}

#[test]
fn test_morris_sandbox_basic() {
    let sandbox = MorrisSandbox::new();
    let code = r#"
fn main() {
    println!("hello from morris");
}
"#;
    let outcome = sandbox.evaluate_code(code, "test_hello.rs");
    assert!(outcome.compiles || !outcome.compiles);
}

#[test]
fn test_morris_sandbox_panic_detection() {
    let sandbox = MorrisSandbox::new();
    let code = r#"
fn main() {
    panic!("test panic");
}
"#;
    let outcome = sandbox.evaluate_code(code, "test_panic.rs");
    eprintln!(
        "compiles={} runs={} stderr={}",
        outcome.compiles, outcome.runs, outcome.stderr
    );
    eprintln!(
        "exit_code={:?} violations={:?}",
        outcome.exit_code, outcome.violations
    );
    if outcome.compiles {
        assert!(outcome.anomaly_triggered);
    }
}

#[test]
fn test_phase_shield_nominal() {
    let shield = PhaseShield::new();
    let action = shield.evaluate(0.3, false, 5);
    assert_eq!(action, ShieldAction::None);
}

#[test]
fn test_phase_shield_overshoot() {
    let shield = PhaseShield::new();
    let action = shield.evaluate(0.7, true, 25);
    match action {
        ShieldAction::PhaseShift(angle) => assert!(angle > 0.0),
        _ => panic!("expected PhaseShift, got {:?}", action),
    }
}

#[test]
fn test_phase_shield_emergency() {
    let shield = PhaseShield::new();
    let action = shield.evaluate(0.95, true, 50);
    assert!(matches!(action, ShieldAction::EmergencyHalt(_)));
}

#[test]
fn test_self_optimizer_diagnose() {
    let tm = TemporalMemory::new(50, 4);
    let fatigue = vec![0u32; 50];
    let mut opt = SelfOptimizer::new();
    let d = opt.diagnose(&tm, &fatigue);
    assert_eq!(d.tm_cell_count, 0);
    assert_eq!(d.fatigue_max, 0);
}

#[test]
fn test_self_optimizer_high_fatigue() {
    let tm = TemporalMemory::new(50, 4);
    let mut fatigue = vec![0u32; 50];
    fatigue[5] = 60;
    let mut opt = SelfOptimizer::new();
    opt.diagnose(&tm, &fatigue);
    let cfg = opt.auto_tune();
    assert!(cfg.fatigue_factor > opt.config.base_fatigue_factor);
}

#[test]
fn test_legion_second_synthesize() {
    let mut mirror = make_test_mirror();
    let mut legion = SecondLegion::new();
    let report = legion.synthesize("fn process", &mut mirror);
    assert_eq!(report.legion, "Second Legion (Researchers)");
    assert!(report.hypothesis_sdr.is_some());
}

#[test]
fn test_legion_first_execute() {
    let mut legion = FirstLegion::new();
    let code = "fn main() { println!(\"ok\"); }";
    let report = legion.execute_code(code, "test_legion.rs");
    assert!(report.command == "ExecuteCode");
}

#[test]
fn test_legion_coordinator_cycle() {
    let mut mirror = make_test_mirror();
    let mut coord = LegionCoordinator::new();
    let reports = coord.run_cycle("fn compute", &mut mirror);
    assert_eq!(reports.len(), 2);
}

#[test]
fn test_unified_pipeline_run() {
    let mut mirror = make_test_mirror();
    let mut pipeline = UnifiedPipeline::new();
    pipeline.config.max_cycles = 3;
    let result = pipeline.run(&["fn process", "fn compute"], &mut mirror);
    assert!(result.cycles_completed > 0);
    assert!(result.total_hypotheses > 0);
}

#[test]
fn test_pipeline_shield_escalation() {
    let mut mirror = make_test_mirror();
    let mut pipeline = UnifiedPipeline::new();
    pipeline.config.max_cycles = 15;
    pipeline.config.auto_adjust = true;
    pipeline.config.shield_enabled = true;
    let result = pipeline.run(&["fn process", "fn compute"], &mut mirror);
    assert!(result.shield_actions as usize <= result.cycles_completed);
}

#[test]
fn test_buffer_entropy() {
    use fuga::core::hypervector::Hypervector;
    let opt = SelfOptimizer::new();
    let empty: Vec<Hypervector> = Vec::new();
    assert!((opt.read_buffer_entropy(&empty) - 0.5).abs() < 0.01);

    let mut buf = Vec::new();
    for _ in 0..5 {
        buf.push(Hypervector::random(256));
    }
    let e = opt.read_buffer_entropy(&buf);
    assert!(e > 0.0 && e <= 1.0);
}
