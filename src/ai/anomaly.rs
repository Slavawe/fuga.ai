use crate::ai::htm_temporal::TemporalMemory;
use crate::ai::sdr::encode_text;

const Z_WINDOW: usize = 20;
const ANOMALY_Z_THRESHOLD: f64 = -1.5;
const ENTROPY_SHIFT_THRESHOLD: f64 = 0.15;

#[derive(Clone, Debug)]
pub struct AnomalyEvent {
    pub timestamp: usize,
    pub token: String,
    pub match_score: f64,
    pub z_score: f64,
    pub entropy_token: f64,
    pub entropy_shift: f64,
}

#[derive(Clone, Debug)]
pub struct StyloProfile {
    pub token_entropy: f64,
    pub unique_ratio: f64,
    pub avg_line_len: f64,
    pub structural_entropy: f64,
}

impl StyloProfile {
    pub fn compute(text: &str) -> Self {
        let chars: Vec<char> = text.chars().collect();
        let lines: Vec<&str> = text.lines().collect();
        let words: Vec<&str> = text.split_whitespace().collect();
        let n = words.len();

        let mut freq: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for w in &words {
            *freq.entry(w).or_insert(0) += 1;
        }
        let token_entropy = freq.values().fold(0.0, |acc, &c| {
            let p = c as f64 / n.max(1) as f64;
            if p > 0.0 { acc - p * p.log2() } else { acc }
        });
        let unique_ratio = freq.len() as f64 / n.max(1) as f64;
        let avg_line_len = if lines.is_empty() {
            0.0
        } else {
            lines.iter().map(|l| l.len() as f64).sum::<f64>() / lines.len() as f64
        };

        let mut struct_freq: Vec<usize> = Vec::new();
        let mut cur = 0usize;
        for c in chars {
            if c == '\n' {
                struct_freq.push(cur);
                cur = 0;
            } else if c.is_whitespace() {
                cur += 1;
            } else {
                cur += 1;
            }
        }
        if cur > 0 {
            struct_freq.push(cur);
        }
        let total_w = struct_freq.iter().sum::<usize>().max(1);
        let structural_entropy = struct_freq.iter().fold(0.0, |acc, &c| {
            let p = c as f64 / total_w as f64;
            if p > 0.0 { acc - p * p.log2() } else { acc }
        });

        StyloProfile {
            token_entropy,
            unique_ratio,
            avg_line_len,
            structural_entropy,
        }
    }
}

pub struct AnomalyDetector {
    pub tm: TemporalMemory,
    pub match_history: Vec<f64>,
    pub anomalies: Vec<AnomalyEvent>,
    pub step: usize,
    pub baseline_profile: Option<StyloProfile>,
}

impl AnomalyDetector {
    pub fn new(tm: TemporalMemory) -> Self {
        AnomalyDetector {
            tm,
            match_history: Vec::with_capacity(Z_WINDOW),
            anomalies: Vec::new(),
            step: 0,
            baseline_profile: None,
        }
    }

    pub fn feed_text(&mut self, text: &str) -> Vec<AnomalyEvent> {
        self.step += 1;
        let mut events = Vec::new();
        let sdr = encode_text(text);
        let (_pred, match_score) = self.tm.feed(&sdr);

        let profile = StyloProfile::compute(text);
        if self.baseline_profile.is_none() && self.step < 5 {
            return events;
        }
        if self.baseline_profile.is_none() {
            self.baseline_profile = Some(profile.clone());
        }

        self.match_history.push(match_score);
        if self.match_history.len() > Z_WINDOW {
            self.match_history.remove(0);
        }

        let z_score = self.compute_z_score(match_score);

        let entropy_shift = if let Some(ref base) = self.baseline_profile {
            (profile.token_entropy - base.token_entropy).abs()
        } else {
            0.0
        };

        if z_score < ANOMALY_Z_THRESHOLD || entropy_shift > ENTROPY_SHIFT_THRESHOLD {
            let event = AnomalyEvent {
                timestamp: self.step,
                token: text.chars().take(60).collect(),
                match_score,
                z_score,
                entropy_token: profile.token_entropy,
                entropy_shift,
            };
            self.anomalies.push(event.clone());
            events.push(event);
        }

        events
    }

    pub fn compute_z_score(&self, score: f64) -> f64 {
        let n = self.match_history.len();
        if n < 3 {
            return 0.0;
        }
        let mean = self.match_history.iter().sum::<f64>() / n as f64;
        let variance = self
            .match_history
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let std = variance.sqrt();
        if std < 1e-9 {
            0.0
        } else {
            (score - mean) / std
        }
    }

    pub fn profile_text(&self, text: &str) -> StyloProfile {
        StyloProfile::compute(text)
    }

    pub fn stats(&self) -> String {
        format!(
            "step={} matches={} anomalies={} baseline_entropy={:.4}",
            self.step,
            self.match_history.len(),
            self.anomalies.len(),
            self.baseline_profile
                .as_ref()
                .map(|p| p.token_entropy)
                .unwrap_or(0.0)
        )
    }
}

#[derive(Clone, Debug)]
pub struct CorrectionSignal {
    pub source_token: String,
    pub z_score: f64,
    pub entropy_shift: f64,
    pub match_score: f64,
}

pub struct AnomalyReflector {
    pub detector: AnomalyDetector,
    pub correction_buffer: Vec<CorrectionSignal>,
    pub total_corrections: usize,
}

impl AnomalyReflector {
    pub fn new(tm: TemporalMemory) -> Self {
        AnomalyReflector {
            detector: AnomalyDetector::new(tm),
            correction_buffer: Vec::new(),
            total_corrections: 0,
        }
    }

    pub fn feed_text(&mut self, text: &str) -> Vec<AnomalyEvent> {
        let events = self.detector.feed_text(text);
        for ev in &events {
            self.correction_buffer.push(CorrectionSignal {
                source_token: ev.token.clone(),
                z_score: ev.z_score,
                entropy_shift: ev.entropy_shift,
                match_score: ev.match_score,
            });
            self.total_corrections += 1;
        }
        if self.correction_buffer.len() > 100 {
            self.correction_buffer.remove(0);
        }
        events
    }

    pub fn drain_corrections(&mut self) -> Vec<CorrectionSignal> {
        std::mem::take(&mut self.correction_buffer)
    }

    pub fn reflect_summary(&self) -> String {
        format!(
            "reflections={} total={} last_z={:.2} buffer={}",
            self.total_corrections,
            self.detector.step,
            self.correction_buffer
                .last()
                .map(|c| c.z_score)
                .unwrap_or(0.0),
            self.correction_buffer.len(),
        )
    }
}
