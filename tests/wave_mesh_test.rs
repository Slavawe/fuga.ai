use fuga::ai::wave_mesh::{self, hypervector_to_phasors, phasors_to_hypervector, spectral_spread};
use fuga::Hypervector;

#[test]
fn test_phasor_roundtrip() {
    let hv = Hypervector::random(8192);
    let phasors = hypervector_to_phasors(&hv);
    let recovered = phasors_to_hypervector(&phasors);
    let sim = hv.similarity(&recovered);
    assert!(sim > 0.999, "Roundtrip should be near-perfect, got sim={}", sim);
}

#[test]
fn test_spectral_spread() {
    let bits = vec![1i8; 8192];
    let sp = spectral_spread(&bits);
    assert!((sp - 1.0).abs() < 0.001);
    let bits2 = vec![-1i8; 8192];
    let sp2 = spectral_spread(&bits2);
    assert!(sp2.abs() < 0.001);
}

#[test]
fn test_radiotap_frame() {
    let hv = Hypervector::random(8192);
    let payload = hv.to_bytes();
    let frame = wave_mesh::build_radiotap_frame(&payload);
    assert!(frame.len() > 100);
    let decoded = wave_mesh::decode_radiotap_frame(&frame);
    assert!(decoded.is_some());
    assert_eq!(decoded.unwrap(), payload);
}
