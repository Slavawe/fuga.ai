//! Embodiment demo: seed a tiny "skeleton of experience", then let the agent
//! act (generate code → compile) and observe (rustc stderr → SDR).
//!
//! Run:  cargo run --release --example agent_loop

use fuga::ai::{ActOptions, AgentBrain, Episode, FugaAI, MemoryStore, PainAvoidance};

fn main() {
    fuga::gpu::init_gpu();

    let dim = 8192usize;
    let mut ai = FugaAI::<4, 4>::new(dim, 3);
    seed_memory(&mut ai.memory, dim);
    ai.memory.build_text_index();

    // Two-layer cognitive scaffold (cortex + optional frozen hippocampus).
    let mut brain = AgentBrain::new(16384, 0.30);
    // When the finalized 1 GB base brain exists, attach it:
    //   brain.attach_hippocampus("core_brain.fgc");
    if std::path::Path::new("core_brain.fgc").exists() {
        brain.attach_hippocampus("core_brain.fgc").ok();
    }

    println!("=== Embodiment: act == observe == learn ===");
    let prompts = [
        "write a rust function that returns the sum of two integers",
        "write a rust hello world that prints to stdout",
        "define a struct Point with x and y as f64",
    ];
    // Phase A: warm up episodic memory with reactive ticks, then...
    for (i, prompt) in prompts.iter().enumerate() {
        let opts = ActOptions {
            max_tokens: 40,
            temperature: 0.7,
            noise: 0.0,
        };
        let tick = fuga::ai::act_and_observe(&mut ai, prompt, &opts);
        brain.learn(&Episode::from(&tick));
        println!("warm #{} {} → {:?}", i, prompt, tick.outcome);
        println!("  code:\n{}", &tick.code);
        println!(
            "  sensation bitmask: {:?}",
            &tick.sensation.bits[..2.min(SDRW)]
        );
    }
    println!(
        "knowledge size: {} (hippocampus) + {} (cortex)",
        brain.knowledge_size(),
        brain.cortex.len()
    );

    // Phase B: PREDICTIVE generation — double resonance through the brain.
    println!("\n=== Predictive: brain.think (double resonance) ===");
    for (i, prompt) in prompts.iter().enumerate() {
        let opts = ActOptions {
            max_tokens: 40,
            temperature: 0.7,
            noise: 0.02,
        };
        let out = brain.think_with_retries(&mut ai, prompt, &opts, 2);
        let tick = fuga::ai::observe(&mut ai, &out.code, &fuga::ai::AgentPaths::default());
        brain.learn(&Episode::from(&tick));
        println!("predictive #{} {} → {:?}", i, prompt, tick.outcome);
        println!(
            "  cortex veto: {:?} (pruned {} branch(es): {:?})",
            out.veto, out.pruned, out.prune_outcomes
        );
        println!("  code:\n{}", &out.code);
    }
}

fn seed_memory(mem: &mut MemoryStore, dim: usize) {
    use fuga::core::hypervector::Hypervector;
    fn hvec(dim: usize, s: &str) -> Hypervector {
        let mut words = vec![0u64; dim / 64];
        let mut x: u64 = 14695981039346656037;
        for b in s.as_bytes() {
            x ^= *b as u64;
            x = x.wrapping_mul(1099511628211);
        }
        for (i, w) in words.iter_mut().enumerate() {
            let mut v = x
                .wrapping_add((i * 31) as u64)
                .wrapping_mul(0x9E3779B97F4A7C15);
            v ^= v >> 30;
            v = v.wrapping_mul(0xbf58476d1ce4e5b9);
            v ^= v >> 27;
            *w = v;
        }
        Hypervector { dim, words }
    }
    let d = dim;
    mem.store_raw(
        &hvec(d, "fn main println hello"),
        "fn main() { println!(\"hello\"); }",
        "seed.rs",
        "code",
    );
    mem.store_raw(
        &hvec(d, "let add x y"),
        "fn add(a: i32, b: i32) -> i32 { a + b }",
        "seed.rs",
        "code",
    );
    mem.store_raw(
        &hvec(d, "struct x y"),
        "struct Point { x: f64, y: f64 }",
        "seed.rs",
        "code",
    );
    mem.store_raw(
        &hvec(d, "return sum"),
        "let s = a + b; s",
        "seed.rs",
        "code",
    );
}

const SDRW: usize = 8;
