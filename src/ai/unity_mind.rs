use super::autonomous_mind::AutonomousMind;
use super::personas::Action;
use super::sdr::{SdrVector, encode_text};

/// Short-lived composite state created only when the three minds agree.
#[derive(Clone, Debug)]
pub struct UnityMind {
    pub composite_soul: SdrVector,
    pub sync_score: f32,
}

impl UnityMind {
    pub fn try_awaken(
        svyatogor: &Action,
        vlad: &Action,
        qin: &Action,
        souls: [&SdrVector; 3],
        threshold: f32,
    ) -> Option<Self> {
        let thoughts = [action_sdr(svyatogor), action_sdr(vlad), action_sdr(qin)];
        let sync_score = pairwise_sync(&thoughts);
        if sync_score < threshold {
            return None;
        }

        let composite_soul = souls[0].bundle(&[souls[1], souls[2]]);
        Some(Self {
            composite_soul,
            sync_score,
        })
    }

    /// Returns the composite identity signal for a prompt. Actual code
    /// manifestation remains delegated to the existing generation pipeline.
    pub fn manifest_signal(&self, prompt: &str) -> SdrVector {
        encode_text(prompt).bind(&self.composite_soul)
    }

    pub fn manifest(&self, _brains: &mut [AutonomousMind; 3]) -> Option<String> {
        None
    }
}

fn action_sdr(action: &Action) -> SdrVector {
    encode_text(&format!("{:?}", action))
}

fn pairwise_sync(thoughts: &[SdrVector; 3]) -> f32 {
    let ab = thoughts[0].soft_overlap(&thoughts[1]);
    let ac = thoughts[0].soft_overlap(&thoughts[2]);
    let bc = thoughts[1].soft_overlap(&thoughts[2]);
    ((ab + ac + bc) / 3.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::personas::{Horseman, qin::Qin, svyatogor::Svyatogor, vlad::Vlad};

    #[test]
    fn unity_sleeps_when_actions_disagree() {
        let mut s = Svyatogor::new();
        let mut v = Vlad::new();
        let mut q = Qin::new();
        let a = s.process_prompt("safe task");
        let b = v.process_prompt("safe task");
        let c = q.process_prompt("safe task");
        assert!(
            UnityMind::try_awaken(
                &a,
                &b,
                &c,
                [s.identity_sdr(), v.identity_sdr(), q.identity_sdr()],
                0.85,
            )
            .is_none()
        );
    }

    #[test]
    fn unity_awakes_for_identical_consensus() {
        let s = Svyatogor::new();
        let v = Vlad::new();
        let q = Qin::new();
        let action = Action::Approve("consensus".into());
        let unity = UnityMind::try_awaken(
            &action,
            &action,
            &action,
            [s.identity_sdr(), v.identity_sdr(), q.identity_sdr()],
            0.85,
        );
        assert!(unity.is_some());
        assert!(unity.unwrap().composite_soul.popcount() > 0);
    }
}
