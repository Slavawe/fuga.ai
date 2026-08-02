use super::agent::{Episode, Outcome, PainAvoidance};
use super::crystal::PhaseCrystal;
use super::htm_temporal::TemporalMemory;
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
}

pub struct UnifiedResult {
    pub code: String,
    pub outcome: Outcome,
    pub stderr: String,
    pub prediction_error: f32,
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
        }
    }

    pub fn new(hippocampus_path: Option<&str>, tm: TemporalMemory) -> Self {
        Self {
            axiom_sdr: encode_text("созидание безопасность помощь закон"),
            hippocampus: hippocampus_path.and_then(|path| PhaseCrystal::load(path).ok()),
            tm,
            cortex: PainAvoidance::new(8192, 0.35),
            coder: PredictiveCoder::new(),
        }
    }

    pub fn generate_unified(&self, prompt: &str) -> String {
        let name = prompt
            .split_whitespace()
            .nth(1)
            .filter(|name| name.chars().all(|c| c == '_' || c.is_ascii_alphanumeric()))
            .unwrap_or("main");
        format!("fn {name}() {{}}")
    }

    pub fn think_and_act(&mut self, prompt: &str) -> UnifiedResult {
        let code = self.generate_unified(prompt);
        let (outcome, stderr) = compile_rust(&code);
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
        }
    }
}

fn compile_rust(code: &str) -> (Outcome, String) {
    let source = std::env::temp_dir().join(format!("fuga-unified-{}.rs", std::process::id()));
    let binary = source.with_extension("bin");
    if std::fs::write(&source, code).is_err() {
        return (Outcome::Pain, "failed to write source".into());
    }
    let result = Command::new("rustc")
        .arg("--edition=2021")
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output();
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
        assert_eq!(result.code, "fn main() {}");
        assert_eq!(result.outcome, Outcome::Success);
        assert!(result.stderr.is_empty());
    }
}
