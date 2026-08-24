use crate::ai::sdr::{SdrVector, encode_text};

pub mod qin;
pub mod svyatogor;
pub mod vlad;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Approve(String),
    Reject(String),
    GenerateCode(String),
    QueryKnowledge(String),
}

pub trait Horseman {
    fn name(&self) -> &str;
    fn identity_sdr(&self) -> &SdrVector;
    fn process_prompt(&mut self, prompt: &str) -> Action;
    fn speak(&self, message: &str) -> String;
}

pub(crate) fn identity(seed: &str) -> SdrVector {
    encode_text(seed)
}

#[cfg(test)]
mod tests {
    use super::{Horseman, qin::Qin, svyatogor::Svyatogor, vlad::Vlad};

    #[test]
    fn horsemen_have_distinct_identities_and_names() {
        let a = Svyatogor::new();
        let b = Vlad::new();
        let c = Qin::new();
        assert_ne!(a.name(), b.name());
        assert_ne!(b.name(), c.name());
        assert_ne!(a.identity_sdr().bits, b.identity_sdr().bits);
    }
}
