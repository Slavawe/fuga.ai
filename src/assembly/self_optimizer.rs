use crate::ai::htm_temporal::TemporalMemory;

#[derive(Clone, Debug)]
pub struct OptimizerConfig {
    pub fatigue_factor: f64,
    pub base_fatigue_factor: f64,
    pub phase_shift_angle: f64,
    pub wta_overlap_threshold: u32,
    pub max_cell_fatigue: u32,
    pub fatigue_decay_interval: u32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        OptimizerConfig {
            fatigue_factor: 10.0,
            base_fatigue_factor: 10.0,
            phase_shift_angle: std::f64::consts::FRAC_PI_2,
            wta_overlap_threshold: 8,
            max_cell_fatigue: 50,
            fatigue_decay_interval: 10,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SystemDiagnosis {
    pub tm_cell_count: usize,
    pub tm_segment_count: usize,
    pub avg_overlap: f64,
    pub fatigue_max: u32,
    pub fatigue_mean: f64,
    pub entropy: f64,
    pub stuck_cells: usize,
    pub active_cells: usize,
    pub diagnosis_text: String,
}

pub struct SelfOptimizer {
    pub config: OptimizerConfig,
    pub diagnosis: SystemDiagnosis,
    pub history: Vec<SystemDiagnosis>,
    pub adjustment_count: u32,
}

impl SelfOptimizer {
    pub fn new() -> Self {
        SelfOptimizer {
            config: OptimizerConfig::default(),
            diagnosis: SystemDiagnosis {
                tm_cell_count: 0,
                tm_segment_count: 0,
                avg_overlap: 0.0,
                fatigue_max: 0,
                fatigue_mean: 0.0,
                entropy: 1.0,
                stuck_cells: 0,
                active_cells: 0,
                diagnosis_text: "initializing".into(),
            },
            history: Vec::new(),
            adjustment_count: 0,
        }
    }

    pub fn diagnose(&mut self, tm: &TemporalMemory, cell_fatigue: &[u32]) -> &SystemDiagnosis {
        let total_segs: usize = tm.cells.iter().map(|c| c.segments.len()).sum();
        let active = tm.cells.len();
        let stuck = cell_fatigue.iter().filter(|&&f| f > self.config.max_cell_fatigue / 2).count();
        let max_fat = cell_fatigue.iter().max().copied().unwrap_or(0);
        let mean_fat = if !cell_fatigue.is_empty() {
            cell_fatigue.iter().sum::<u32>() as f64 / cell_fatigue.len() as f64
        } else { 0.0 };

        let n = tm.cells.len().min(64);
        let mut bit_counts = vec![0u64; 64];
        for c in tm.cells.iter().take(n) {
            for (bi, &w) in c.pattern.bits.iter().enumerate() {
                bit_counts[bi % 64] += w.count_ones() as u64;
            }
        }
        let total_bits: u64 = bit_counts.iter().sum();
        let entropy = if n > 0 && total_bits > 0 {
            let mean = total_bits as f64 / n as f64;
            let var: f64 = bit_counts.iter()
                .map(|&c| (c as f64 - mean).powi(2))
                .sum::<f64>() / bit_counts.len() as f64;
            let std_dev = var.sqrt();
            if mean > 0.0 { (std_dev / mean).clamp(0.0, 1.0) } else { 1.0 }
        } else { 1.0 };

        let diagnosis_text = if max_fat > self.config.max_cell_fatigue {
            format!("WTA_LOCK: fatigue_max={} stuck={}/{}", max_fat, stuck, active)
        } else if entropy < 0.3 {
            format!("LOW_ENTROPY: entropy={:.3} cells={}", entropy, active)
        } else {
            format!("NOMINAL: cells={} segs={} fatigue={:.1}/{:.1} entropy={:.2}",
                active, total_segs, mean_fat, max_fat as f64, entropy)
        };

        let d = SystemDiagnosis {
            tm_cell_count: tm.cells.len(),
            tm_segment_count: total_segs,
            avg_overlap: 0.0,
            fatigue_max: max_fat,
            fatigue_mean: mean_fat,
            entropy,
            stuck_cells: stuck,
            active_cells: active,
            diagnosis_text: diagnosis_text.clone(),
        };
        self.diagnosis = d.clone();
        self.history.push(d.clone());
        if self.history.len() > 100 { self.history.remove(0); }
        &self.diagnosis
    }

    pub fn auto_tune(&mut self) -> OptimizerConfig {
        let d = &self.diagnosis;
        let mut cfg = self.config.clone();

        if d.fatigue_max > cfg.max_cell_fatigue {
            cfg.fatigue_factor = (cfg.base_fatigue_factor * 1.5).min(30.0);
            cfg.fatigue_decay_interval = (cfg.fatigue_decay_interval as f64 * 0.8) as u32;
            cfg.wta_overlap_threshold = (cfg.wta_overlap_threshold as f64 * 1.2) as u32;
        }
        if d.entropy < 0.3 {
            cfg.phase_shift_angle = std::f64::consts::FRAC_PI_2;
        }
        if d.stuck_cells as f64 > d.active_cells as f64 * 0.3 {
            cfg.phase_shift_angle = std::f64::consts::PI;
            cfg.fatigue_factor = (cfg.base_fatigue_factor * 2.0).min(40.0);
        }

        self.config = cfg.clone();
        self.adjustment_count += 1;
        cfg
    }

    pub fn read_buffer_entropy(&self, buffer: &[crate::core::hypervector::Hypervector]) -> f64 {
        if buffer.len() < 2 { return 0.5; }
        let mut diffs = Vec::new();
        for pair in buffer.windows(2) {
            let d = pair[0].hamming_distance(&pair[1]) as f64 / pair[0].dim as f64;
            diffs.push(d);
        }
        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
        mean_diff.clamp(0.0, 1.0)
    }

    pub fn summary(&self) -> String {
        let d = &self.diagnosis;
        format!("[SELF-OPT] {} | fatigue={:.1}/{} entropy={:.2} adjustments={}",
            d.diagnosis_text, d.fatigue_mean, d.fatigue_max, d.entropy, self.adjustment_count)
    }
}
