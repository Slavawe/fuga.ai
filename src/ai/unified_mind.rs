use super::agent::{Episode, Outcome, PainAvoidance};
use super::crystal::PhaseCrystal;
use super::hierarchical_jepa::HierarchicalJEPA;
use super::htm_temporal::TemporalMemory;
use super::personas::vlad::Vlad;
use super::predictive_coder::PredictiveCoder;
use super::sdr::{SdrVector, encode_text};
use super::state_loader::MindState;
use std::process::Command;

pub struct UnifiedMind {
    pub axiom_sdr: SdrVector,
    pub hippocampus: Option<PhaseCrystal>,
    pub tm: TemporalMemory,
    pub cortex: PainAvoidance,
    pub coder: PredictiveCoder,
    pub hjepa: HierarchicalJEPA,
}

pub struct UnifiedResult {
    pub code: String,
    pub outcome: Outcome,
    pub stderr: String,
    pub prediction_error: f32,
    pub hjepa_prediction_popcount: u32,
}

impl UnifiedMind {
    pub fn from_state(state: MindState, axiom_sdr: SdrVector) -> Self {
        Self {
            axiom_sdr,
            hippocampus: state.hippocampus,
            tm: state
                .temporal_memory
                .unwrap_or_else(|| TemporalMemory::new(512, 4)),
            cortex: state.cortex,
            coder: PredictiveCoder::new(),
            hjepa: HierarchicalJEPA::new(8192),
        }
    }

    pub fn new(hippocampus_path: Option<&str>, tm: TemporalMemory) -> Self {
        Self {
            axiom_sdr: encode_text("созидание безопасность помощь закон"),
            hippocampus: hippocampus_path.and_then(|path| PhaseCrystal::load(path).ok()),
            tm,
            cortex: PainAvoidance::new(8192, 0.35),
            coder: PredictiveCoder::new(),
            hjepa: HierarchicalJEPA::new(8192),
        }
    }

    pub fn generate_unified(&self, prompt: &str) -> String {
        let action = Vlad::new().generate_ast_with_tm(&self.tm, prompt);
        match action {
            super::personas::Action::GenerateCode(code) if !code.is_empty() => code,
            _ => "fn main() {}".to_string(),
        }
    }

    pub fn think_and_act(&mut self, prompt: &str) -> UnifiedResult {
        let prompt_hv = encode_text(prompt).to_hypervector(8192);
        let top_down_prediction = self.hjepa.predict(&[&prompt_hv]);
        let mut code = self.generate_unified(prompt);
        let (mut outcome, mut stderr) = compile_rust(&code);

        // Self-healing pass: a successful compile with only dead-code warnings
        // is repairable without changing semantics. Retry once with an explicit
        // allowance, then learn from the final compiler verdict.
        if outcome == Outcome::Warn
            && (stderr.contains("dead_code") || stderr.contains("is never used"))
            && !code.contains("allow(dead_code)")
        {
            code = format!("#![allow(dead_code)]\n{}", code);
            (outcome, stderr) = compile_rust(&code);
        }

        let actual = encode_text(match outcome {
            Outcome::Success => "SUCCESS",
            Outcome::Warn => "WARN",
            Outcome::Pain => "PAIN",
        });
        let expected = self
            .hippocampus
            .as_ref()
            .and_then(|h| h.query(prompt).map(|_| encode_text(prompt)))
            .unwrap_or_else(|| self.axiom_sdr.clone());
        let prediction_error = self.coder.compute_error(&actual, &expected);
        self.cortex.learn(&Episode {
            code: code.clone(),
            outcome,
            stderr: stderr.clone(),
        });
        UnifiedResult {
            code,
            outcome,
            stderr,
            prediction_error,
            hjepa_prediction_popcount: top_down_prediction
                .first()
                .map(|prediction| prediction.words.iter().map(|word| word.count_ones()).sum())
                .unwrap_or(0),
        }
    }
}

fn compile_rust(code: &str) -> (Outcome, String) {
    // Unique temp file per call: parallel tests (and concurrent agent runs)
    // would otherwise write the same {pid}.rs and corrupt each other's builds.
    let salt = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let source = std::env::temp_dir().join(format!(
        "fuga-unified-{}-{x}.rs",
        std::process::id(),
        x = salt
    ));
    let binary = source.with_extension("bin");
    if std::fs::write(&source, code).is_err() {
        return (Outcome::Pain, "failed to write source".into());
    }
    let mut command = Command::new("rustc");
    command.arg("--edition=2021");
    if !code.contains("fn main") {
        command.arg("--crate-type").arg("lib");
    }
    let result = command.arg(&source).arg("-o").arg(&binary).output();
    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&binary);
    match result {
        Ok(output) if output.status.success() => {
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            if stderr.trim().is_empty() {
                (Outcome::Success, stderr)
            } else {
                (Outcome::Warn, stderr)
            }
        }
        Ok(output) => (
            Outcome::Pain,
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ),
        Err(error) => (Outcome::Pain, error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_mind_compiles_generated_main() {
        let mut mind = UnifiedMind::new(None, TemporalMemory::new(32, 4));
        let result = mind.think_and_act("fn main");
        assert_eq!(result.outcome, Outcome::Success);
        assert!(result.stderr.is_empty());
        assert!(result.code.starts_with("fn main()"));
    }

    #[test]
    fn unified_generation_uses_temporal_memory_path() {
        let mut tm = TemporalMemory::new(64, 4);
        let context = encode_text("fn main() {");
        let next = encode_text("let");
        tm.learn_sequence(&context, &next);
        let mind = UnifiedMind::new(None, tm);
        assert!(mind.generate_unified("fn main").contains("let _ = 0;"));
    }

    #[test]
    fn self_healing_clears_dead_code_warning() {
        let mut mind = UnifiedMind::new(None, TemporalMemory::new(32, 4));
        let result = mind.think_and_act("fn sum(a: i32, b: i32) -> i32");
        assert_eq!(result.outcome, Outcome::Success);
        assert!(result.code.contains("allow(dead_code)"));
        assert!(result.stderr.is_empty());
    }
}
