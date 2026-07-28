use crate::core::hypervector::Hypervector;
use core::f32::consts::PI;

pub const VSA_DIM: usize = 8192;
pub const SUBCARRIERS: usize = 64;
pub const PHASOR_SYMBOLS: usize = VSA_DIM / SUBCARRIERS;

#[derive(Debug, Clone, Copy)]
pub struct ComplexPhasor {
    pub re: f32,
    pub im: f32,
}

pub fn hypervector_to_phasors(hv: &Hypervector) -> Vec<[ComplexPhasor; SUBCARRIERS]> {
    let bits = hv.to_i8_bits();
    let mut symbols = Vec::with_capacity(PHASOR_SYMBOLS);

    for chunk in bits.chunks_exact(SUBCARRIERS) {
        let mut symbol = [ComplexPhasor { re: 0.0, im: 0.0 }; SUBCARRIERS];
        for (k, &bit) in chunk.iter().enumerate() {
            let phase = if bit > 0 { 0.0 } else { PI };
            symbol[k] = ComplexPhasor {
                re: phase.cos(),
                im: phase.sin(),
            };
        }
        symbols.push(symbol);
    }
    symbols
}

pub fn phasors_to_hypervector(symbols: &[[ComplexPhasor; SUBCARRIERS]]) -> Hypervector {
    let dim = symbols.len() * SUBCARRIERS;
    let mut bits = Vec::with_capacity(dim);
    for symbol in symbols {
        for phasor in symbol {
            let angle = phasor.im.atan2(phasor.re);
            bits.push(if angle.abs() > PI / 2.0 { -1i8 } else { 1i8 });
        }
    }
    Hypervector::from_i8_bits(dim, &bits)
}

pub fn hypervector_to_byte_payload(hv: &Hypervector) -> Vec<u8> {
    hv.to_bytes()
}

pub fn build_radiotap_frame(payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(128 + payload.len());

    frame.extend_from_slice(&[
        0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
    ]);

    frame.extend_from_slice(&[
        0x08, 0x00,
        0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0x46, 0x75, 0x67, 0x61, 0x00, 0x01,
        0x46, 0x75, 0x67, 0x61, 0x00, 0x01,
        0x00, 0x00,
    ]);

    frame.extend_from_slice(payload);

    frame
}

pub fn decode_radiotap_frame(frame: &[u8]) -> Option<Vec<u8>> {
    if frame.len() < 30 { return None; }
    let radiotap_len = u16::from_le_bytes([frame[2], frame[3]]) as usize;
    let data_start = radiotap_len + 24;
    if data_start >= frame.len() { return None; }
    Some(frame[data_start..].to_vec())
}

pub fn cosine_similarity(a: &Hypervector, b: &Hypervector) -> f64 {
    a.similarity(b)
}

pub fn spectral_spread(bits: &[i8]) -> f64 {
    let n = bits.len() as f64;
    let ones = bits.iter().filter(|&&b| b > 0).count() as f64;
    ones / n
}
