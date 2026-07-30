use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnomalyEvent {
    pub pred_count: f32,
    pub power_mw: f32,
    pub query_memory: bool,
    pub overshoot: bool,
    pub violation_kind: String,
}

impl AnomalyEvent {
    pub fn new(pred_count: f32, power_mw: f32, violation_kind: impl Into<String>) -> Self {
        let overshoot = pred_count > 100.0 || power_mw > 500.0;
        Self {
            pred_count,
            power_mw,
            query_memory: true,
            overshoot,
            violation_kind: violation_kind.into(),
        }
    }

    pub fn is_critical(&self) -> bool {
        self.overshoot && self.power_mw > 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_normal_state() {
        let normal = AnomalyEvent::new(12.5, 45.0, "None");
        assert!(!normal.overshoot);
        assert!(!normal.is_critical());
        assert_eq!(normal.violation_kind, "None");
    }

    #[test]
    fn test_anomaly_critical_overshoot() {
        let critical = AnomalyEvent::new(150.0, 1200.0, "InfiniteReplicationLoop");
        assert!(critical.overshoot);
        assert!(critical.is_critical());
        assert_eq!(critical.violation_kind, "InfiniteReplicationLoop");
    }

    #[test]
    fn test_fatigue_resets_trigger() {
        let mut pred_count = 120.0f32;
        let event_before = AnomalyEvent::new(pred_count, 300.0, "WTA_Lock");
        assert!(event_before.overshoot);
        pred_count -= 10.0 * 3.0;
        let event_after = AnomalyEvent::new(pred_count, 300.0, "WTA_Lock");
        assert!(!event_after.overshoot);
    }
}
