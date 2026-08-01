use crate::vsa::topology::ls_bind;
use crate::core::hypervector::Hypervector;

#[derive(Clone, Debug, PartialEq)]
pub enum ShieldAction {
    None,
    PhaseShift(f64),
    ClearBuffer,
    ResetTM,
    EmergencyHalt(String),
}

pub struct PhaseShield {
    pub consecutive_violations: u32,
    pub last_shift_angle: f64,
    pub shift_count: u32,
}

impl PhaseShield {
    pub fn new() -> Self {
        PhaseShield {
            consecutive_violations: 0,
            last_shift_angle: std::f64::consts::FRAC_PI_2,
            shift_count: 0,
        }
    }

    pub fn evaluate(&self, l2_err: f64, overshoot: bool, fatigue_max: u32) -> ShieldAction {
        let mut severity = 0u32;
        if overshoot { severity += 2; }
        if l2_err > 0.6 { severity += 1; }
        if fatigue_max > 20 { severity += 2; }
        if l2_err > 0.85 { severity += 2; }

        match severity {
            0 => ShieldAction::None,
            1..=2 => {
                let angle = std::f64::consts::FRAC_PI_4;
                ShieldAction::PhaseShift(angle)
            }
            3..=4 => {
                let angle = std::f64::consts::FRAC_PI_2;
                ShieldAction::PhaseShift(angle)
            }
            5..=6 => {
                ShieldAction::PhaseShift(std::f64::consts::PI)
            }
            _ => ShieldAction::EmergencyHalt("Phase collapse detected".into()),
        }
    }

    pub fn apply_shift(&self, hv: &Hypervector, angle: f64) -> Hypervector {
        let _n_blocks = hv.dim / 32;
        let phase_steps = ((angle / std::f64::consts::FRAC_PI_2) as usize).max(1) % 4;
        if phase_steps == 0 { return hv.clone(); }
        let mut permuted = hv.clone();
        for _ in 0..phase_steps {
            let rotated = permuted.permute(1);
            permuted = ls_bind(&permuted, &rotated, 32);
        }
        permuted
    }

    pub fn step(&mut self, action: &ShieldAction) {
        match action {
            ShieldAction::None => {
                self.consecutive_violations = self.consecutive_violations.saturating_sub(1);
            }
            ShieldAction::PhaseShift(angle) => {
                self.last_shift_angle = *angle;
                self.shift_count += 1;
                self.consecutive_violations += 1;
            }
            ShieldAction::ClearBuffer | ShieldAction::ResetTM => {
                self.consecutive_violations += 2;
            }
            ShieldAction::EmergencyHalt(_) => {
                self.consecutive_violations = 100;
            }
        }
    }

    pub fn is_critical(&self) -> bool {
        self.consecutive_violations >= 10 || self.shift_count > 20
    }
}
