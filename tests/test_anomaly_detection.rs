use fuga::anomaly::AnomalyEvent;

#[test]
fn test_anomaly_event_creation_and_overshoot() {
    let normal_event = AnomalyEvent::new(12.5, 45.0, "None");
    assert!(!normal_event.overshoot);
    assert!(!normal_event.is_critical());

    let critical_event = AnomalyEvent::new(150.0, 1200.0, "InfiniteReplicationLoop");
    assert!(critical_event.overshoot);
    assert!(critical_event.is_critical());
    assert_eq!(critical_event.violation_kind, "InfiniteReplicationLoop");
}

#[test]
fn test_vsa_fatigue_resets_anomaly_trigger() {
    let mut simulated_pred_count = 120.0f32;
    let event_before = AnomalyEvent::new(simulated_pred_count, 300.0, "WTA_Lock");
    assert!(event_before.overshoot);

    simulated_pred_count -= 10.0 * 3.0;
    let event_after = AnomalyEvent::new(simulated_pred_count, 300.0, "WTA_Lock");
    assert!(!event_after.overshoot);
}
