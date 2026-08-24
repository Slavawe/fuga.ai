use crate::ai::self_mirror::SelfMirror;
use crate::assembly::legion::LegionCoordinator;
use crate::assembly::phase_shield::{PhaseShield, ShieldAction};
use crate::assembly::self_optimizer::{OptimizerConfig, SelfOptimizer};

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub max_cycles: usize,
    pub auto_adjust: bool,
    pub shield_enabled: bool,
    pub emit_report: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            max_cycles: 10,
            auto_adjust: true,
            shield_enabled: true,
            emit_report: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PipelineResult {
    pub cycles_completed: usize,
    pub total_hypotheses: usize,
    pub total_executions: usize,
    pub total_violations: usize,
    pub convergence: bool,
    pub final_stability: f64,
    pub shield_actions: u32,
    pub optimizer_adjustments: u32,
    pub summary: String,
}

pub struct UnifiedPipeline {
    pub config: PipelineConfig,
    pub optimizer: SelfOptimizer,
    pub shield: PhaseShield,
    pub coordinator: LegionCoordinator,
    pub cycle: usize,
    pub cell_fatigue: Vec<u32>,
}

impl UnifiedPipeline {
    pub fn new() -> Self {
        UnifiedPipeline {
            config: PipelineConfig::default(),
            optimizer: SelfOptimizer::new(),
            shield: PhaseShield::new(),
            coordinator: LegionCoordinator::new(),
            cycle: 0,
            cell_fatigue: Vec::new(),
        }
    }

    pub fn run(&mut self, seed_inputs: &[&str], mirror: &mut SelfMirror) -> PipelineResult {
        let mut total_hyp = 0usize;
        let mut total_exec = 0usize;
        let mut total_vio = 0usize;

        for _ in 0..self.config.max_cycles {
            self.cycle += 1;
            if self.cycle > self.config.max_cycles {
                break;
            }

            let input = seed_inputs[(self.cycle - 1) % seed_inputs.len()];
            let reports = self.coordinator.run_cycle(input, mirror);

            for r in &reports {
                match r.command.as_str() {
                    "SynthesizeHypothesis" => total_hyp += 1,
                    "ExecuteCode" => {
                        total_exec += 1;
                        if let Some(ref a) = r.anomaly {
                            if a.overshoot {
                                total_vio += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }

            if self.config.auto_adjust && mirror.predictor.tm.cells.len() > self.cell_fatigue.len()
            {
                self.cell_fatigue.resize(mirror.predictor.tm.cells.len(), 0);
            }

            self.optimizer
                .diagnose(&mirror.predictor.tm, &self.cell_fatigue);
            if self.config.auto_adjust {
                let new_cfg = self.optimizer.auto_tune();
                self.apply_optimization(&new_cfg, mirror);
            }

            if self.config.shield_enabled {
                let stable = self.coordinator.latest_stability();
                let overshoot = total_vio > self.cycle;
                let action = self.shield.evaluate(
                    1.0 - stable,
                    overshoot,
                    self.optimizer.diagnosis.fatigue_max,
                );
                self.apply_shield_action(&action, mirror);
                self.shield.step(&action);

                if self.shield.is_critical() {
                    let result = PipelineResult {
                        cycles_completed: self.cycle,
                        total_hypotheses: total_hyp,
                        total_executions: total_exec,
                        total_violations: total_vio,
                        convergence: false,
                        final_stability: self.coordinator.latest_stability(),
                        shield_actions: self.shield.shift_count,
                        optimizer_adjustments: self.optimizer.adjustment_count,
                        summary: format!(
                            "EMERGENCY_HALT: shield triggered after {} cycles",
                            self.cycle
                        ),
                    };
                    return result;
                }
            }
        }

        let final_stability = self.coordinator.latest_stability();
        PipelineResult {
            cycles_completed: self.cycle,
            total_hypotheses: total_hyp,
            total_executions: total_exec,
            total_violations: total_vio,
            convergence: total_vio == 0 && final_stability > 0.5,
            final_stability,
            shield_actions: self.shield.shift_count,
            optimizer_adjustments: self.optimizer.adjustment_count,
            summary: format!(
                "cycles={} hyp={} exec={} vio={} stability={:.3} shield={} opt={}",
                self.cycle,
                total_hyp,
                total_exec,
                total_vio,
                final_stability,
                self.shield.shift_count,
                self.optimizer.adjustment_count
            ),
        }
    }

    fn apply_optimization(&mut self, cfg: &OptimizerConfig, _mirror: &mut SelfMirror) {
        let _ = cfg.fatigue_factor;
    }

    fn apply_shield_action(&self, action: &ShieldAction, mirror: &mut SelfMirror) {
        match *action {
            ShieldAction::PhaseShift(angle) => {
                let _n = mirror.predictor.buffer.len();
                let shift_count = (angle / std::f64::consts::FRAC_PI_2) as usize;
                if shift_count > 0 {
                    for hv in mirror.predictor.buffer.iter_mut() {
                        for _ in 0..shift_count.min(3) {
                            let rotated = hv.permute(1);
                            *hv = crate::vsa::topology::ls_bind(hv, &rotated, 32);
                        }
                    }
                }
            }
            ShieldAction::ClearBuffer => {
                mirror.predictor.buffer.clear();
            }
            ShieldAction::ResetTM => {
                mirror.predictor.tm.cells.clear();
                mirror.predictor.tm.window.clear();
                mirror.predictor.tm.step = 0;
            }
            ShieldAction::EmergencyHalt(_) => {}
            ShieldAction::None => {}
        }
    }

    pub fn full_report(&self) -> String {
        let mut out = String::new();
        out.push_str("═══ UNIFIED PIPELINE REPORT ═══\n");
        out.push_str(&format!(
            "Cycle: {}/{}\n",
            self.cycle, self.config.max_cycles
        ));
        out.push_str(&format!("{}\n", self.optimizer.summary()));
        out.push_str(&format!("{}\n", self.coordinator.summary()));
        out.push_str(&format!(
            "PhaseShield: shifts={} consec_violations={} critical={}\n",
            self.shield.shift_count,
            self.shield.consecutive_violations,
            self.shield.is_critical()
        ));
        out.push_str(&format!(
            "OptimizerConfig: fatigue_factor={:.1} threshold={} angle={:.3}\n",
            self.optimizer.config.fatigue_factor,
            self.optimizer.config.wta_overlap_threshold,
            self.optimizer.config.phase_shift_angle
        ));
        out
    }
}
