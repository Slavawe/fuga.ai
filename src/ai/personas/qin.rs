use super::{Action, Horseman, identity};
use crate::ai::sdr::SdrVector;

pub struct Qin {
    identity: SdrVector,
}

impl Qin {
    pub fn new() -> Self {
        Self {
            identity: identity("Цинь Шихуань Ди баланс знания страж"),
        }
    }
}

impl Horseman for Qin {
    fn name(&self) -> &str {
        "Цинь Шихуань Ди"
    }
    fn identity_sdr(&self) -> &SdrVector {
        &self.identity
    }
    fn process_prompt(&mut self, prompt: &str) -> Action {
        Action::QueryKnowledge(prompt.into())
    }
    fn speak(&self, message: &str) -> String {
        format!("[{}]: {}", self.name(), message)
    }
}
