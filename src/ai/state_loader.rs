use super::agent::PainAvoidance;
use super::crystal::PhaseCrystal;
use super::htm_temporal::TemporalMemory;
use std::path::Path;

pub struct MindState {
    pub hippocampus: Option<PhaseCrystal>,
    pub temporal_memory: Option<TemporalMemory>,
    pub cortex: PainAvoidance,
}

pub struct MindStateLoader;

impl MindStateLoader {
    pub fn load(hippo_path: &str, tm_path: &str, cortex_path: &str) -> MindState {
        let hippocampus = if Path::new(hippo_path).exists() {
            println!("Loading Hippocampus from {}", hippo_path);
            PhaseCrystal::load(hippo_path).ok()
        } else {
            None
        };

        let temporal_memory = if Path::new(tm_path).exists() {
            println!("Loading Temporal Memory from {}", tm_path);
            TemporalMemory::load(tm_path)
        } else {
            None
        };

        let cortex = if Path::new(cortex_path).exists() {
            PainAvoidance::load(cortex_path).unwrap_or_else(|_| PainAvoidance::new(8192, 0.35))
        } else {
            PainAvoidance::new(8192, 0.35)
        };

        MindState {
            hippocampus,
            temporal_memory,
            cortex,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_files_create_empty_state() {
        let state = MindStateLoader::load(
            "/tmp/hermes-no-hippo.fgc",
            "/tmp/hermes-no-tm.bin",
            "/tmp/hermes-no-cortex.bin",
        );
        assert!(state.hippocampus.is_none());
        assert!(state.temporal_memory.is_none());
        assert!(state.cortex.is_empty());
    }
}
