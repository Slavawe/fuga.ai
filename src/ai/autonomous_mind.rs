use super::agent::{Outcome, PainAvoidance};
use super::crystal::PhaseCrystal;
use super::personas::Action;
use super::sdr::{SdrVector, encode_text};

#[derive(Clone, Debug)]
pub struct MindPaths {
    pub hippocampus: String,
    pub tm: String,
    pub cortex: String,
}

pub struct AutonomousMind {
    pub name: String,
    pub soul: SdrVector,
    pub hippocampus_path: String,
    pub tm_path: String,
    pub cortex_path: String,
    pub hippocampus: Option<PhaseCrystal>,
    pub tm: Option<super::htm_temporal::TemporalMemory>,
    pub cortex: PainAvoidance,
}

impl AutonomousMind {
    pub fn incarnate(name: &str, soul: SdrVector, paths: MindPaths) -> Self {
        let hippocampus = PhaseCrystal::load(&paths.hippocampus).ok();
        let cortex =
            PainAvoidance::load(&paths.cortex).unwrap_or_else(|_| PainAvoidance::new(8192, 0.35));
        Self {
            name: name.to_string(),
            soul,
            hippocampus_path: paths.hippocampus,
            tm_path: paths.tm,
            cortex_path: paths.cortex,
            hippocampus,
            tm: None,
            cortex,
        }
    }

    pub fn perceive(&self, reality: &SdrVector) -> SdrVector {
        reality.bind(&self.soul)
    }

    pub fn perceive_and_act(&mut self, reality: &SdrVector, prompt: &str) -> Action {
        let perceived = self.perceive(reality);
        let query = format!("{}:resonance={}", prompt, perceived.popcount());
        if let Some(hippo) = &self.hippocampus {
            if hippo.query(&query).is_some() {
                if let Some((outcome, _)) = self.cortex.probe(prompt) {
                    if outcome != Outcome::Success {
                        return Action::Reject(format!("{}: past {:?}", self.name, outcome));
                    }
                }
                return Action::GenerateCode(prompt.to_string());
            }
        }
        Action::Reject(format!("{}: knowledge unavailable", self.name))
    }

    pub fn identity_signal(&self, text: &str) -> SdrVector {
        encode_text(text).bind(&self.soul)
    }
}
