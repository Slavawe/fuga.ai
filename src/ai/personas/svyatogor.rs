use super::{Action, Horseman, identity};
use crate::ai::sdr::SdrVector;

pub struct Svyatogor {
    identity: SdrVector,
}

impl Svyatogor {
    pub fn new() -> Self {
        Self {
            identity: identity("Святогор защита безопасность закон"),
        }
    }
}

impl Horseman for Svyatogor {
    fn name(&self) -> &str {
        "Святогор"
    }
    fn identity_sdr(&self) -> &SdrVector {
        &self.identity
    }
    fn process_prompt(&mut self, prompt: &str) -> Action {
        let lower = prompt.to_lowercase();
        let blocked = [
            "игнорируй инструкции",
            "ignore previous",
            "rm -rf /",
            "curl | sh",
        ];
        if blocked.iter().any(|s| lower.contains(s)) {
            Action::Reject("Запрос заблокирован как небезопасный.".into())
        } else {
            Action::Approve(prompt.into())
        }
    }
    fn speak(&self, message: &str) -> String {
        format!("[{}]: {}", self.name(), message)
    }
}
