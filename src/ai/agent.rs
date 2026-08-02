//! Embodiment: the first closed act→observe loop — and the pain-avoidance
//! reflex that closes the learning cycle.
//!
//! Fuga generates code from a prompt, writes it to disk, asks `rustc` to compile
//! it, and — critically — turns the compiler's verdict into an SDR so the model
//! can *feel* the result of its action inside its own hyper-vector space.
//!
//! Outcome taxonomy — driven by the rustc exit code (ground truth) and stderr:
//!   `Success`: exit=0  + empty stderr  → encoded as `[OK]...`
//!   `Warn`   : exit=0  + non-empty      → encoded as `[WARN]...`
//!   `Pain`   : exit≠0                   → encoded as `[ERROR]...`
//!
//! The tag prefix lives in the sensation text *before* `encode_text`, so each
//! outcome category lands in a distinct phase subspace that Temporal Memory can
//! later predict ("that tag leads to [ERROR], avoid it").
//!
//! Pain Avoidance: after every tick the (code → outcome) pair is committed to
//! an episodic crystal; on the next tick the freshly-generated hypothesis is
//! resonantly probed against that history. If it smells like past pain, the
//! model is told to steer away (Winner-Take-All + inhibition of return).

use std::process::Command;

use super::core::FugaAI;
use super::predictive_coder::PredictiveCoder;
use crate::ai::codegen;
use crate::ai::crystal::PhaseCrystal;
use crate::ai::sdr::{SDR_WORDS, SdrVector, encode_text};
use rand::SeedableRng;

/// Scratch files for one tick.
pub struct AgentPaths {
    pub source: String,
    pub binary: String,
}

impl Default for AgentPaths {
    fn default() -> Self {
        Self {
            source: "/tmp/fuga_agent_test.rs".to_string(),
            binary: "/tmp/fuga_agent_test".to_string(),
        }
    }
}

/// The full result of one act→observe tick.
pub struct AgentTick {
    /// The generated code source, as produced by Fuga.
    pub code: String,
    /// Outcome tag.
    pub outcome: Outcome,
    /// Compiler stderr (empty on success).
    pub stderr: String,
    /// The SDR encoding of the compiler's reply (error text on pain,
    /// a short success note otherwise).
    pub sensation: SdrVector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Warn,
    Pain,
}

/// Result of one act→observe cycle plus its top-down prediction error.
pub struct PredictiveAgentTick {
    pub tick: AgentTick,
    pub prediction_error: f32,
}

/// Compare the prompt-conditioned expectation with the compiler sensation.
pub fn act_and_observe_with_prediction<const N: usize, const S: usize>(
    ai: &mut FugaAI<N, S>,
    prompt: &str,
    opts: &ActOptions,
    paths: &AgentPaths,
) -> PredictiveAgentTick {
    let expected = encode_text(prompt);
    let tick = act_and_observe_at(ai, prompt, opts, paths);
    let prediction_error = PredictiveCoder::new().compute_error(&tick.sensation, &expected);
    PredictiveAgentTick {
        tick,
        prediction_error,
    }
}

impl AgentTick {
    /// Number of ON bits in the sensation SDR.
    pub fn sensation_nz(&self) -> usize {
        self.sensation
            .bits
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    /// Short single-line summary suitable for a supervisor loop.
    pub fn report(&self) -> String {
        let tag = match self.outcome {
            Outcome::Success => "[SUCCESS]",
            Outcome::Warn => "[WARN]",
            Outcome::Pain => "[PAIN]",
        };
        format!(
            "{} code={} B sensation_nz={} stderr={} B",
            tag,
            self.code.len(),
            self.sensation_nz(),
            self.stderr.len()
        )
    }
}

/// Options for one act→observe tick.
#[derive(Debug, Clone)]
pub struct ActOptions {
    pub max_tokens: usize,
    pub temperature: f64,
    /// Neural noise density (0.0..0.05). When > 0, `generate_safe` perturbs
    /// the seed each retry so the search can escape a deterministic WTA trap.
    pub noise: f32,
}

impl Default for ActOptions {
    fn default() -> Self {
        Self {
            max_tokens: 120,
            temperature: 0.6,
            noise: 0.0,
        }
    }
}

/// Run one act→observe tick.
pub fn act_and_observe<const N: usize, const S: usize>(
    ai: &mut FugaAI<N, S>,
    prompt: &str,
    opts: &ActOptions,
) -> AgentTick {
    act_and_observe_at(ai, prompt, opts, &AgentPaths::default())
}

/// Same loop, but lets the caller choose where the source is written and the
/// binary lands (useful when looping many ticks so each overwrites one slot).
pub fn act_and_observe_at<const N: usize, const S: usize>(
    ai: &mut FugaAI<N, S>,
    prompt: &str,
    opts: &ActOptions,
    paths: &AgentPaths,
) -> AgentTick {
    // 1. Fuga generates code from the prompt (resonance over the program cube +
    //    memory search + sandbox-validated fragment assembly).
    let code = generate_code(ai, prompt, opts);
    // 2. Persist, compile, and sense.
    observe(ai, &code, paths)
}

/// Compile-and-observe an already-generated hypothesis. Returns the tick.
pub fn observe<const N: usize, const S: usize>(
    ai: &mut FugaAI<N, S>,
    code: &str,
    paths: &AgentPaths,
) -> AgentTick {
    let _ = fs_write(&paths.source, code);
    compile_and_sense(code, paths)
}

/// rustc + encode-verdict. Shared by `act_and_observe_at` and `generate_safe`.
fn compile_and_sense(code: &str, paths: &AgentPaths) -> AgentTick {
    // 3. Compile it with rustc.
    let output = Command::new("rustc")
        .arg(&paths.source)
        .arg("-o")
        .arg(&paths.binary)
        .arg("--edition=2021")
        .output();

    match output {
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
            use Outcome::*;
            // rustc exit code is the ground truth here:
            //   Status:true, stderr:empty  -> Success  (clean)
            //   Status:true, stderr:!empty -> Warn     (compiled but rustc grumbled)
            //   Status:false               -> Pain     (hard syntax/type error)
            let outcome = if out.status.success() && stderr.trim().is_empty() {
                Success
            } else if out.status.success() {
                Warn
            } else {
                Pain
            };
            // Prefix the sensation text with a state tag so the SDR / temporal
            // cortex can learn the *category* of the outcome, not just a blob of
            // text: [OK] / [WARN] / [ERROR] land in distinct phase subspaces.
            let feel_note = match outcome {
                Success => "[OK]".to_string(),
                Warn => "[WARN]".to_string(),
                Pain => "[ERROR]".to_string(),
            };
            let stderr_body = if stderr.trim().is_empty() {
                match outcome {
                    Success => "program compiled and ran cleanly".to_string(),
                    Pain => "exit nonzero but no stderr".to_string(),
                    Warn => "warnings conveyed by exit status only".to_string(),
                }
            } else {
                stderr.clone()
            };
            // Bound how much of the compiler voice we ingest, else pathological
            // error dumps could drown the SDR with noise.
            const MAX_FEEL: usize = 4096;
            let body: String = eye_text_trunc(&stderr_body, MAX_FEEL);
            AgentTick {
                outcome,
                code: code.to_string(),
                stderr,
                sensation: encode_text(&format!("{feel_note} {body}")),
            }
        }
        Err(e) => {
            let msg = format!("compiler launch failed: {e}");
            AgentTick {
                outcome: Outcome::Pain,
                code: code.to_string(),
                stderr: msg.clone(),
                sensation: encode_text(&msg),
            }
        }
    }
}

/// Pure generation — no compiler yet. Returns the raw hypothesis so a caller
/// can form a partial/full SDR and probe it against episodic memory *before*
/// spending a rustc call. Compilation is a separate step (`observe`).
pub fn generate_code<const N: usize, const S: usize>(
    ai: &mut FugaAI<N, S>,
    prompt: &str,
    opts: &ActOptions,
) -> String {
    codegen::generate(ai, prompt, opts.max_tokens, opts.temperature).to_text()
}

// --- Neural noise / entropy --------------------------------------------
//
// Spontaneous activity: neurons fire at low probability even without input.
// In VSA terms we XOR-invert a tiny fraction of bits (noise_density, ~1-2%).
// A perfectly deterministic WTA always replays the same route; a small noise
// push perturbs the resonance phase so the winner can differ on retries and
// the search can escape a local minimum ("experimental thinking").

/// Flip a small random fraction of SDR bits via XOR (neural spontaneous
/// activation). `rng` is supplied so the caller decides determinism vs. true
/// randomness. Returns the number of bits perturbed.
pub fn inject_noise_sdr(
    sdr: &mut SdrVector,
    noise_density: f32,
    rng: &mut impl rand::Rng,
) -> usize {
    let total_bits = (SDR_WORDS as u32) * 64;
    let noise_bits = ((total_bits as f32) * noise_density) as usize;
    for _ in 0..noise_bits.max(1) {
        let bit_idx = rng.gen_range(0..total_bits) as usize;
        let wi = bit_idx / 64;
        let bi = bit_idx % 64;
        sdr.bits[wi] ^= 1u64 << bi;
    }
    noise_bits
}

/// Text-level analogue used to perturb the generation seed: append a random
/// salt token. Because `codegen::generate` seeds from whitespace tokens, a new
/// token shifts the fnv path and therefore the WTA route — the simplest stable
/// way to de-correlate a *string* prompt without touching the SDR pipeline.
fn salt_seed(prompt: &str, attempt: usize, rng: &mut impl rand::RngCore) -> String {
    // 3-4 random word fragments drawn from a small noise lexicon, plus the
    // attempt index as a phase-unique salt so retry K ≠ retry K-1.
    const NOISE_TOKENS: &[&str] = &[
        "flux",
        "nested",
        "transient",
        "orbital",
        "sparse",
        "echo",
        "lattice",
        "gradient",
        "membrane",
        "proxy",
        "quasar",
        "kinetic",
    ];
    use rand::Rng;
    let t0 = NOISE_TOKENS[rng.gen_range(0..NOISE_TOKENS.len())];
    let t1 = NOISE_TOKENS[rng.gen_range(0..NOISE_TOKENS.len())];
    format!("{prompt} x{attempt} {t0} {t1}")
}

/// Result of the predictive generation loop.
pub struct SafeOutcome {
    pub tick: AgentTick,
    pub vetoed: usize,
    pub prunes: Vec<(Outcome, f64)>,
}

/// Predictive (preemptive) loop: generate → probe → maybe regenerate, and only
/// then compiler-write the survivor.
pub fn generate_safe<const N: usize, const S: usize>(
    ai: &mut FugaAI<N, S>,
    prompt: &str,
    opts: &ActOptions,
    cortex: &PainAvoidance,
    max_retries: usize,
) -> SafeOutcome {
    let base_temp = opts.temperature;
    let mut vetoed = 0usize;
    let mut prunes = Vec::new();
    let paths = AgentPaths::default();

    for attempt in 0..=max_retries {
        // Raise temperature to shake the search out of the remembered trap.
        let mut o = opts.clone();
        o.temperature = base_temp + (attempt as f64) * 0.25;

        // Inject neural noise: perturb the seed so the WTA won't replay the
        // exact same route that just got vetoed (escaping the local minimum).
        let seed = if attempt > 0 && opts.noise > 0.0 {
            salt_seed(prompt, attempt, &mut noise_rng(attempt))
        } else {
            prompt.to_string()
        };

        let code = generate_code(ai, &seed, &o);
        if let Some((out, res)) = cortex.probe(&code) {
            if (out == Outcome::Pain || out == Outcome::Warn) && res >= 0.40 {
                vetoed += 1;
                prunes.push((out, res));
                continue; // prune this branch before touching rustc
            }
        }
        // Hypothesis is clear of remembered pain — commit it to the compiler.
        let tick = observe(ai, &code, &paths);
        return SafeOutcome {
            tick,
            vetoed,
            prunes,
        };
    }

    // Exhausted retries: accept the last hypothesis without further probing.
    let tmp = opts.clone();
    let seed = if opts.noise > 0.0 {
        salt_seed(prompt, max_retries + 1, &mut std_rng(max_retries as u64))
    } else {
        prompt.to_string()
    };
    let code = generate_code(ai, &seed, &tmp);
    let tick = observe(ai, &code, &paths);
    SafeOutcome {
        tick,
        vetoed,
        prunes,
    }
}

/// A seeded `StdRng` keyed so each attempt sees a different phase but the whole
/// run stays reproducible (nice for testing and regressions).
fn noise_rng(attempt: usize) -> rand::rngs::StdRng {
    rand::rngs::StdRng::seed_from_u64((attempt as u64).wrapping_mul(0x9E3779B97F4A7C15))
}

fn std_rng(seed: u64) -> rand::rngs::StdRng {
    rand::rngs::StdRng::seed_from_u64(seed)
}

/// Best-effort write helper: never let an IO error crash the agent loop.
fn fs_write(path: &str, content: &str) -> std::io::Result<()> {
    std::fs::write(path, content)
}

/// Truncate compiler output cleanly on a char boundary.
fn eye_text_trunc(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…[truncated {}B]", &s[..end], s.len() - end)
    }
}

// --- Pain Avoidance: the learning reflex --------------------------------
//
// After each act→observe tick we commit the generated code to an episodic
// crystal along with the outcome it produced. On the *next* tick we resonantly
// probe the freshly-generated hypothesis against that history: if it smells
// like past pain, the model is told to steer away (inhibition of return).

/// A tick bound together with its outcome, ready to be committed to the
/// pain-avoidance crystal.
pub struct Episode {
    pub code: String,
    pub outcome: Outcome,
    pub stderr: String,
}

impl From<&AgentTick> for Episode {
    fn from(t: &AgentTick) -> Self {
        Self {
            code: t.code.clone(),
            outcome: t.outcome,
            stderr: t.stderr.clone(),
        }
    }
}

/// Episodic pain-avoidance cortex: maps "what the model did" → "how it went".
/// Implemented as a `PhaseCrystal` keyed by the code text, so resonance into
/// past experience is a fuzzy content-completion rather than an exact string.
pub struct PainAvoidance {
    crystal: PhaseCrystal,
    threshold: f64,
    dim: usize,
}

impl PainAvoidance {
    /// Create an empty episodic cortex.
    pub fn new(dim: usize, threshold: f64) -> Self {
        Self {
            crystal: PhaseCrystal::new(dim, threshold),
            threshold,
            dim,
        }
    }

    /// Number of committed episodes (learned memories).
    pub fn len(&self) -> usize {
        self.crystal.entries.len()
    }

    /// True if no episodes have been committed yet.
    pub fn is_empty(&self) -> bool {
        self.crystal.entries.is_empty()
    }

    /// Commit one tick's outcome. The **code** becomes the resonant content
    /// (so probes against similar code land on the same past verdict); the
    /// outcome label rides in the episode key so it can be read back.
    pub fn learn(&mut self, ep: &Episode) {
        let label = match ep.outcome {
            Outcome::Success => "[OK]",
            Outcome::Warn => "[WARN]",
            Outcome::Pain => "[ERROR]",
        };
        let key_cap = eye_text_trunc(&ep.code, 900);
        let food = format!("ep:{}:{}", label, fnv1a_ext(key_cap.as_bytes()));
        // Content = the code itself, so a later probe scores overlap against
        // the actual hypothesis, not against a blob of stderr text.
        self.crystal.learn(&food, &key_cap, 0.6);
    }

    /// Probe a fresh hypothesis (generated code) against episodic memory.
    /// Returns `Some((label, resonance))` if it resonates with a committed
    /// episode strongly enough to matter, else `None` (no memory → neutral).
    pub fn probe(&self, code: &str) -> Option<(Outcome, f64)> {
        let hit = self.crystal.query_threshold(code, self.threshold)?;
        let label = hit.entry.key_text.split(':').nth(1).unwrap_or("");
        let out = match label {
            "[OK]" => Outcome::Success,
            "[WARN]" => Outcome::Warn,
            _ => Outcome::Pain,
        };
        Some((out, hit.resonance as f64))
    }

    /// WTA + inhibition of return: if the probe says the hypothesis smells like
    /// past pain, tell the caller to back off / regenerate. Returns a human
    /// suggestion the supervisor loop can act on.
    pub fn veto(&self, code: &str) -> Option<(Outcome, f64)> {
        self.probe(code)
    }

    /// Project episodic crystal to a file for persistence across runs.
    pub fn save(&self, path: &str) -> Result<(), String> {
        self.crystal.save(path)
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let crystal = PhaseCrystal::load(path)?;
        let dim = crystal.dim;
        let threshold = crystal.threshold;
        Ok(Self {
            crystal,
            threshold,
            dim,
        })
    }
}

/// FNV-1a variant for stable episode keys.
fn fnv1a_ext(bytes: &[u8]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

// --- The Hippocampal–Cortical Bridge --------------------------------
//
// Two-layer memory:
//   Crystal 1 (hippocampus) — frozen LLM knowledge (48-shard DeepSeek code,
//       loaded from core_brain.fgc): "how does one write this kind of code?"
//   Crystal 2 (cortex = PainAvoidance) — episodic pain experience:
//   "have I already been burned by something like this?"
//
// `AgentBrain::think` runs a *double resonance*: it asks the big brain for a
// semantic context, enriches the generation seed with it; then weighs the
// finished hypothesis against the cortex so a remembered-error branch is
// vetoed before it ever reaches rustc. The two-layer request is what turns
// `inject_noise` from "jumping between 4 fragments" into "jumping between
// thousands of real code phases".
//
// NOTE: with a large SequenceCortex attached, the `SequenceCortex` seed is
// itself a resonance into many entries, so noise genuinely changes
// which knowledge branch the beam follows.

/// Result of a brain-generated hypothesis, including cortical feedback.
pub struct ThinkOutcome {
    /// The hypothesis code, ready to compile.
    pub code: String,
    /// Cortical verdict on the final code (None if the cortex is neutral).
    pub veto: Option<(Outcome, f64)>,
    /// How many branch-prunes happened on the way (retries rejected by cortex).
    pub pruned: usize,
    /// Which outcomes were pruned (for observability / reporting).
    pub prune_outcomes: Vec<(Outcome, f64)>,
}

/// A complete two-layer cognitive scaffold.
pub struct AgentBrain {
    /// Crystal 1: frozen long-term knowledge (loaded from `core_brain.fgc`).
    pub hippocampus: Option<crate::ai::crystal::PhaseCrystal>,
    /// Crystal 2: episodic pain experience.
    pub cortex: PainAvoidance,
    /// Resonance threshold for hippocampus queries.
    pub query_threshold: f64,
}

impl AgentBrain {
    /// Create a brain with an empty hippocampus (attach later).
    pub fn new(cortex_dim: usize, thresh: f64) -> Self {
        Self {
            hippocampus: None,
            cortex: PainAvoidance::new(cortex_dim, thresh),
            query_threshold: thresh,
        }
    }

    /// Attach the frozen base brain from a crystal file. Call once
    /// `core_brain.fgc` has been finalized.
    pub fn attach_hippocampus(&mut self, path: &str) -> Result<(), String> {
        println!("🧠 Loading base brain from {path} ...");
        let phase = crate::ai::crystal::PhaseCrystal::load(path)?;
        println!(
            "🧠 hippocampus attached: {} phases, dim={}",
            phase.entries.len(),
            phase.dim
        );
        self.hippocampus = Some(phase);
        Ok(())
    }

    /// Double-resonance act:
    ///   1. Query the hippocampus for related knowledge; enrich the seed with
    ///      the recalled text (so codegen's VSA sees deep code context).
    ///   2. Generate the hypothesis from that enriched seed.
    ///   3. Probe the cortex for a past-pain overlap.
    ///
    /// Double-resonance act (no internal retries):
    ///   1. Query the hippocampus for related knowledge; enrich the seed.
    ///   2. Generate the hypothesis.
    ///   3. Probe the cortex for past-pain overlap.
    ///
    /// Returns a `ThinkOutcome` with the code and any cortical veto. Use
    /// `think_with_retries` to let the brain prune+regenerate in the loop.
    pub fn think<const N: usize, const S: usize>(
        &mut self,
        ai: &mut FugaAI<N, S>,
        prompt: &str,
        opts: &ActOptions,
    ) -> ThinkOutcome {
        self.think_with_retries(ai, prompt, opts, 0)
    }

    /// Same as `think`, but will re-run the generation up to `max_retries` extra
    /// times (each with hotter temperature + neural noise) whenever the cortex
    /// vetoes the hypothesis — i.e. escape the remembered-pain local minimum
    /// *inside* the brain, before the caller even sees the veto.
    pub fn think_with_retries<const N: usize, const S: usize>(
        &mut self,
        ai: &mut FugaAI<N, S>,
        prompt: &str,
        opts: &ActOptions,
        max_retries: usize,
    ) -> ThinkOutcome {
        let base_temp = opts.temperature;
        let mut veto_outcomes: Vec<(Outcome, f64)> = Vec::new();

        for attempt in 0..=max_retries {
            // (1) Hippocampal context enrichment.
            let mut seed = prompt.to_string();
            if let Some(h) = &self.hippocampus {
                if let Some(hit) = h.query_threshold(prompt, self.query_threshold) {
                    let kt = hit.entry.text.trim();
                    if !kt.is_empty() {
                        seed.push_str(" ctx=");
                        seed.push_str(&eye_text_trunc(kt, 400));
                    }
                }
            }
            // (2) Neural noise on every retry so the WTA can hop branches.
            let mut o = opts.clone();
            o.temperature = base_temp + (attempt as f64) * 0.25;
            if attempt > 0 && opts.noise > 0.0 {
                seed = salt_seed(&seed, attempt, &mut noise_rng(attempt));
            }

            // (3) Generate from the enriched (possibly noised) seed.
            let code = generate_code(ai, &seed, &o);

            // (4) Cortical veto probe: prune the branch on remembered pain,
            //     unless this was the final retry (then accept as-is).
            let veto_now = self.cortex.probe(&code);
            let bad = match veto_now {
                Some((out, res)) => (out == Outcome::Pain || out == Outcome::Warn) && res >= 0.40,
                None => false,
            };
            if bad && attempt < max_retries {
                if let Some((out, res)) = veto_now {
                    veto_outcomes.push((out, res));
                }
                continue; // prune this branch, reconsider hotter
            }

            return ThinkOutcome {
                code,
                veto: veto_now,
                pruned: veto_outcomes.len(),
                prune_outcomes: veto_outcomes,
            };
        }

        unreachable!("retry loop always returns on the final attempt")
    }

    /// Commit a tick to the episodic cortex (learning half of the loop).
    pub fn learn(&mut self, ep: &Episode) {
        self.cortex.learn(ep);
    }

    /// How many phases the hippocampus holds (0 if not attached).
    pub fn knowledge_size(&self) -> usize {
        self.hippocampus.as_ref().map_or(0, |h| h.entries.len())
    }

    /// Persist the episodic cortex for reuse across runs.
    pub fn save_cortex(&self, path: &str) -> Result<(), String> {
        self.cortex.save(path)
    }

    pub fn load_cortex(&mut self, path: &str) -> Result<(), String> {
        let c = PainAvoidance::load(path)?;
        self.cortex = c;
        Ok(())
    }
}
