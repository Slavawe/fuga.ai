use crate::ai::sdr::{SdrVector, encode_text, sparsify};
use crate::ai::self_mirror::SelfMirror;
use crate::anomaly::AnomalyEvent;
use crate::assembly::morris_sandbox::{MorrisSandbox, SandboxOutcome};
use crate::core::hypervector::Hypervector;

#[derive(Clone, Debug)]
pub enum LegionCommand {
    SynthesizeHypothesis(String),
    ExecuteCode { code: String, file_name: String },
    MonitorAndReport,
    AdjustPhase(f64),
    SaveSnapshot,
}

#[derive(Clone, Debug)]
pub struct LegionReport {
    pub legion: String,
    pub command: String,
    pub success: bool,
    pub output: String,
    pub anomaly: Option<AnomalyEvent>,
    pub phase_stability: f64,
    pub hypothesis_sdr: Option<SdrVector>,
}

pub struct SecondLegion {
    pub name: String,
    pub hypothesis_history: Vec<String>,
}

impl SecondLegion {
    pub fn new() -> Self {
        SecondLegion {
            name: "Second Legion (Researchers)".into(),
            hypothesis_history: Vec::new(),
        }
    }

    pub fn synthesize(&mut self, input: &str, mirror: &mut SelfMirror) -> LegionReport {
        let sdr = encode_text(input);
        let (_tm_pred, tm_match) = mirror.predictor.tm.feed(&sdr);

        let hv =
            crate::ai::temporal_predictor::sdr_to_hypervector(&sdr, mirror.predictor.hjepa.dim);
        mirror.predictor.buffer.push(hv);
        if mirror.predictor.buffer.len() > mirror.predictor.buf_capacity {
            mirror.predictor.buffer.remove(0);
        }

        let hypothesis =
            if mirror.predictor.buffer.len() >= mirror.predictor.hjepa.levels[0].context_len {
                let ctx: Vec<&Hypervector> = mirror.predictor.buffer.iter().collect();
                let temps = [0.8, 1.0, 1.2];
                let (preds, _) = mirror.predictor.hjepa.predict_refined(&ctx, &temps);
                let pred_sdr = sparsify(&preds[2]);
                let mut best = String::new();
                let mut best_o = 0u32;
                for (tok, tsdr) in &mirror.token_vocab {
                    let o = pred_sdr.overlap(tsdr);
                    if o > best_o {
                        best_o = o;
                        best = tok.clone();
                    }
                }
                best
            } else {
                input.split_whitespace().next().unwrap_or("").to_string()
            };

        self.hypothesis_history.push(hypothesis.clone());
        if self.hypothesis_history.len() > 50 {
            self.hypothesis_history.remove(0);
        }

        let stability = if tm_match > 0.5 {
            1.0 - tm_match
        } else {
            tm_match
        };

        LegionReport {
            legion: self.name.clone(),
            command: "SynthesizeHypothesis".into(),
            success: !hypothesis.is_empty(),
            output: format!("hypothesis={} tm_match={:.4}", hypothesis, tm_match),
            anomaly: None,
            phase_stability: stability,
            hypothesis_sdr: Some(encode_text(&hypothesis)),
        }
    }

    pub fn reflect(&self, mirror: &SelfMirror) -> String {
        format!(
            "[{}] nodes={} token_vocab={} tm_cells={} hyp_history={}",
            self.name,
            mirror.nodes.len(),
            mirror.token_vocab.len(),
            mirror.predictor.tm.cells.len(),
            self.hypothesis_history.len()
        )
    }
}

pub struct FirstLegion {
    pub name: String,
    pub sandbox: MorrisSandbox,
    pub execution_history: Vec<SandboxOutcome>,
}

impl FirstLegion {
    pub fn new() -> Self {
        FirstLegion {
            name: "First Legion (Machines)".into(),
            sandbox: MorrisSandbox::new(),
            execution_history: Vec::new(),
        }
    }

    pub fn execute_code(&mut self, code: &str, file_name: &str) -> LegionReport {
        let outcome = self.sandbox.evaluate_code(code, file_name);
        self.execution_history.push(outcome.clone());
        if self.execution_history.len() > 50 {
            self.execution_history.remove(0);
        }

        let anomaly = if outcome.anomaly_triggered {
            let pred_count = outcome.violations.len() as f32 * 50.0;
            let power = outcome.execution_time_us as f32 / 1000.0;
            Some(AnomalyEvent::new(
                pred_count,
                power,
                if outcome.violations.is_empty() {
                    "None".to_string()
                } else {
                    format!("{:?}", outcome.violations[0])
                },
            ))
        } else {
            None
        };

        let stability = if outcome.compiles && outcome.runs {
            1.0
        } else if outcome.compiles {
            0.5
        } else {
            0.1
        };

        LegionReport {
            legion: self.name.clone(),
            command: "ExecuteCode".into(),
            success: outcome.compiles && outcome.runs,
            output: format!(
                "compiles={} runs={} violations={} time={}us stderr={}",
                outcome.compiles,
                outcome.runs,
                outcome.violations.len(),
                outcome.execution_time_us,
                outcome.stderr.chars().take(80).collect::<String>()
            ),
            anomaly,
            phase_stability: stability,
            hypothesis_sdr: None,
        }
    }

    pub fn reflect(&self) -> String {
        let ok = self
            .execution_history
            .iter()
            .filter(|o| o.compiles && o.runs)
            .count();
        let total = self.execution_history.len();
        format!(
            "[{}] executions={} success={} rate={:.1}%",
            self.name,
            total,
            ok,
            if total > 0 {
                ok as f64 / total as f64 * 100.0
            } else {
                0.0
            }
        )
    }
}

pub struct LegionCoordinator {
    pub second: SecondLegion,
    pub first: FirstLegion,
    pub reports: Vec<LegionReport>,
}

impl LegionCoordinator {
    pub fn new() -> Self {
        LegionCoordinator {
            second: SecondLegion::new(),
            first: FirstLegion::new(),
            reports: Vec::new(),
        }
    }

    pub fn run_cycle(&mut self, input: &str, mirror: &mut SelfMirror) -> Vec<LegionReport> {
        let mut cycle = Vec::new();

        let r1 = self.second.synthesize(input, mirror);
        let hyp_sdr = r1.hypothesis_sdr.clone();
        cycle.push(r1.clone());
        self.reports.push(r1);

        if hyp_sdr.is_some() {
            let seed = format!(
                "fn auto_generated_{}() {{",
                self.second.hypothesis_history.len()
            );
            let code = format!(
                "{}\n    println!(\"hypothesis_ok\");\n}}\nfn main() {{ {}(); }}",
                seed,
                "auto_generated_".to_string() + &self.second.hypothesis_history.len().to_string()
            );

            let r2 = self.first.execute_code(&code, "auto.rs");
            cycle.push(r2.clone());
            self.reports.push(r2);
        }

        if self.reports.len() > 200 {
            self.reports.drain(0..50);
        }
        cycle
    }

    pub fn latest_stability(&self) -> f64 {
        self.reports
            .iter()
            .rev()
            .take(5)
            .map(|r| r.phase_stability)
            .sum::<f64>()
            / 5.0f64.max(1.0)
    }

    pub fn summary(&self) -> String {
        let last = self
            .reports
            .last()
            .map(|r| &r.output)
            .unwrap_or(&"".to_string())
            .clone();
        format!(
            "[LegionCoord] stability={:.3} reports={} last={}",
            self.latest_stability(),
            self.reports.len(),
            last.chars().take(60).collect::<String>()
        )
    }
}
