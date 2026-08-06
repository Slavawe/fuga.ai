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
        // CALIBRATED 2026-08: здоровый L2 при согласованном таргете живёт в
        // диапазоне 0.78-1.03 (EMA ~0.93). Порог 0.57 из предыдущей версии
        // убивал здоровый режим мгновенным reset'ом на каждом шаге.
        // Граница Nominal/Warning — 75% от критического порога.
        let warning_boundary = self.max_allowed_l2_delta * 0.75;
        if current_l2_loss < warning_boundary {
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
        // 0.85 = 0.75 * 1.15 (порог по умолчанию)
        let cb = FugaCircuitBreaker::new(1.1500);
        assert_eq!(cb.inspect(0.8000), SystemState::Nominal);
    }

    #[test]
    fn test_circuit_breaker_warning() {
        let cb = FugaCircuitBreaker::new(1.1500);
        assert_eq!(cb.inspect(1.0000), SystemState::DivergingWarning(1.0000));
    }

    #[test]
    fn test_circuit_breaker_critical() {
        let cb = FugaCircuitBreaker::new(1.1500);
        assert_eq!(cb.inspect(1.2000), SystemState::CriticalResetRequired);
    }

    #[test]
    fn test_validate_trajectory() {
        let cb = FugaCircuitBreaker::new(1.1500);
        let pv = Hypervector::random(8192);
        let bv = Hypervector::random(8192);
        let result = cb.validate_trajectory(&pv, &bv);
        assert!(result == true || result == false);
    }
}
