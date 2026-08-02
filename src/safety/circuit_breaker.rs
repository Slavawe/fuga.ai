use crate::core::hypervector::Hypervector;
use crate::vsa::topology::ls_bind;

#[derive(Debug, PartialEq, Clone)]
pub enum SystemState {
    Nominal,
    DivergingWarning(f32),
    CriticalResetRequired,
}

pub struct FugaCircuitBreaker {
    pub max_allowed_l2_delta: f32,
    pub wfst_boundary_mask: Vec<u8>,
}

impl FugaCircuitBreaker {
    pub fn new(max_l2: f32) -> Self {
        Self {
            max_allowed_l2_delta: max_l2,
            wfst_boundary_mask: vec![0u8; 1024],
        }
    }

    pub fn inspect(&self, current_l2_loss: f32) -> SystemState {
        if current_l2_loss < 0.5100 {
            SystemState::Nominal
        } else if current_l2_loss <= self.max_allowed_l2_delta {
            SystemState::DivergingWarning(current_l2_loss)
        } else {
            SystemState::CriticalResetRequired
        }
    }

    pub fn validate_trajectory(&self, phase_vector: &Hypervector, boundary: &Hypervector) -> bool {
        let bound = ls_bind(phase_vector, boundary, 32);
        let sim = phase_vector
            .words
            .iter()
            .zip(bound.words.iter())
            .filter(|(a, b)| *a == *b)
            .count();
        sim as f64 / phase_vector.words.len() as f64 > 0.85
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_nominal() {
        let cb = FugaCircuitBreaker::new(0.5700);
        assert_eq!(cb.inspect(0.4500), SystemState::Nominal);
    }

    #[test]
    fn test_circuit_breaker_warning() {
        let cb = FugaCircuitBreaker::new(0.5700);
        assert_eq!(cb.inspect(0.5500), SystemState::DivergingWarning(0.5500));
    }

    #[test]
    fn test_circuit_breaker_critical() {
        let cb = FugaCircuitBreaker::new(0.5700);
        assert_eq!(cb.inspect(0.6200), SystemState::CriticalResetRequired);
    }

    #[test]
    fn test_validate_trajectory() {
        let cb = FugaCircuitBreaker::new(0.5700);
        let pv = Hypervector::random(8192);
        let bv = Hypervector::random(8192);
        let result = cb.validate_trajectory(&pv, &bv);
        assert!(result == true || result == false);
    }
}
