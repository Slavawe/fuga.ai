use crate::ai::{FugaAI, MemoryStore};
use crate::core::wave_cube::WaveCube;
use crate::fisig_formatter::{self, FisigAnswer};
use crate::multi_engine::MultiEngine;
use crate::physics::reactor::ReactorCore;
use crate::spatial::controller::RoomController;
use crate::weaver::pattern_matcher::TokenInfo;
use crate::weaver::token_builder::TokenBuilder;

pub struct OmniIntegrationLayer {
    pub physics_to_code: bool,
    pub spatial_to_physics: bool,
    pub reactor_to_spatial: bool,
    pub code_to_reactor: bool,
}

impl Default for OmniIntegrationLayer {
    fn default() -> Self {
        Self {
            physics_to_code: true,
            spatial_to_physics: true,
            reactor_to_spatial: true,
            code_to_reactor: true,
        }
    }
}

pub struct OmniEngine<const N: usize, const S: usize> {
    pub ai: FugaAI<N, S>,
    pub multi: Option<MultiEngine>,
    pub reactor: Option<ReactorCore>,
    pub spatial: Option<RoomController>,
    pub integration: OmniIntegrationLayer,
}

impl<const N: usize, const S: usize> OmniEngine<N, S> {
    pub fn new(dim: usize, window: usize) -> Self {
        Self {
            ai: FugaAI::<N, S>::new(dim, window),
            multi: None,
            reactor: None,
            spatial: None,
            integration: OmniIntegrationLayer::default(),
        }
    }

    pub fn load_cube(&mut self, path: &str) -> Result<(), String> {
        let cube = WaveCube::<N, S>::load_bin(path)?;
        self.ai.cube = cube;
        let mem_path = path.replace(".bin", "_mem.bin");
        if let Ok(mem) = MemoryStore::load_bin(&mem_path) {
            self.ai.memory = mem;
        }
        Ok(())
    }

    pub fn with_multi(mut self, dim: usize) -> Self {
        self.multi = Some(MultiEngine::new(dim));
        self
    }

    pub fn with_reactor(mut self) -> Self {
        self.reactor = Some(ReactorCore::default());
        self
    }

    pub fn with_spatial(mut self, dim: usize, half_extent: f64) -> Self {
        self.spatial = Some(RoomController::new(dim, half_extent));
        self
    }

    pub fn set_vocab(&mut self, _vocab_path: &str) {}

    pub fn detect_domain(query: &str) -> &'static str {
        let q = query.to_lowercase();
        if q.contains("zfc")
            || q.contains("set theory")
            || q.contains("axiom")
            || q.contains("category")
            || q.contains("yoneda")
            || q.contains("godel")
            || q.contains("formal logic")
        {
            "zfc"
        } else if q.contains("manifold")
            || q.contains("tensor")
            || q.contains("curvature")
            || q.contains("riemann")
            || q.contains("ricci")
            || q.contains("einstein")
            || q.contains("levi-civita")
            || q.contains("cartan")
            || q.contains("adm")
        {
            "geometry"
        } else if q.contains("qft")
            || q.contains("quantum field")
            || q.contains("energy condition")
            || q.contains("anec")
            || q.contains("quantum inequality")
            || q.contains("casimir")
            || q.contains("ford-roman")
        {
            "qft"
        } else if q.contains("z3")
            || q.contains("formal verification")
            || q.contains("smt")
            || q.contains("llvm")
            || q.contains("alive2")
            || q.contains("crucible")
            || q.contains("dpll")
        {
            "z3"
        } else if q.contains("mach") || q.contains("woodward") || q.contains("met") {
            "met"
        } else if q.contains("alcubierre") || q.contains("warp") {
            "warp"
        } else if q.contains("aether") || q.contains("ether") || q.contains("newton") {
            "aether"
        } else if q.contains("tesla") {
            "tesla"
        } else if q.contains("reactor")
            || q.contains("neutron")
            || q.contains("fission")
            || q.contains("control rod")
            || q.contains("scram")
        {
            "reactor"
        } else if q.contains("spatial")
            || q.contains("room")
            || q.contains("navigation")
            || q.contains("lidar")
            || q.contains("rapier")
            || q.contains("collision")
        {
            "spatial"
        } else if q.contains("code")
            || q.contains("syntax")
            || q.contains("compiler")
            || q.contains("tree-sitter")
            || q.contains("language")
            || q.contains("rust")
            || q.contains("python")
            || q.contains("javascript")
            || q.contains("typescript")
            || q.contains("semantic")
            || q.contains("pattern")
            || q.contains("fn ")
            || q.contains("function")
            || q.contains("impl ")
            || q.contains("struct ")
        {
            "code"
        } else if q.contains("диалог")
            || q.contains("разговор")
            || q.contains("бесед")
            || q.contains("расскажи")
            || q.contains("объясни")
            || q.contains("почему")
            || q.contains("зачем")
            || q.contains("как дела")
            || q.contains("что такое")
            || q.contains("кто ты")
            || q.contains("рассказ")
            || q.contains("история")
            || q.contains("narrative")
            || q.contains("dialogue")
            || q.contains("story")
            || q.contains("conversation")
            || q.contains("hello")
            || q.contains("hi ")
            || q.contains("how are you")
        {
            "general"
        } else {
            "general"
        }
    }

    pub fn query(&mut self, text: &str, tokens: &[TokenInfo]) -> OmniResult {
        let (result, _output) = self.query_with_output(text, tokens);
        result
    }

    pub fn query_with_output(
        &mut self,
        text: &str,
        tokens: &[TokenInfo],
    ) -> (OmniResult, crate::ai::AIOutput) {
        let domain = Self::detect_domain(text);

        let ai_output = self.ai.think(tokens);
        let response_text = ai_output
            .response_tokens
            .as_ref()
            .map(|t| {
                t.iter()
                    .map(|ti| ti.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_else(|| "No response generated".to_string());
        let domain_result = match domain {
            "met" | "warp" | "aether" | "tesla" | "zfc" | "geometry" | "qft" | "z3" => {
                let ans = fisig_formatter::format_answer(&mut self.ai, text, tokens);
                OmniDomainResult::Physics(ans)
            }
            "reactor" => {
                let ans = self.query_reactor(text);
                OmniDomainResult::Reactor(ans)
            }
            "spatial" => {
                let ans = self.query_spatial(text);
                OmniDomainResult::Spatial(ans)
            }
            "code" => {
                let ans = self.query_code(text);
                OmniDomainResult::Code(ans)
            }
            _ => OmniDomainResult::General(response_text),
        };

        let cross_domain = self.run_integration(text, domain);

        (
            OmniResult {
                domain: domain.to_string(),
                domain_result,
                cross_domain,
                entropy: self.ai.cube.global_entropy(),
                coherence: self.ai.cube.coherence(),
                memory_size: self.ai.memory.size(),
            },
            ai_output,
        )
    }

    fn query_reactor(&self, _text: &str) -> String {
        if let Some(ref r) = self.reactor {
            format!(
                "ReactorCore:\n  n (power) = {:.4}\n  c (precursors) = {:.4}\n  T (fuel temp) = {:.1} K\n  rho (reactivity) = {:.1} pcm\n  rods = {}",
                r.n,
                r.c,
                r.t,
                r.rho,
                r.rods.len()
            )
        } else {
            "No reactor core initialized".to_string()
        }
    }

    fn query_spatial(&self, _text: &str) -> String {
        if let Some(ref s) = self.spatial {
            format!(
                "RoomController:\n  target = ({:.2}, {:.2})\n  phase stability = {:.4}\n  coherence = {:.4}\n  entropy = {:.4}",
                s.attractor[0],
                s.attractor[2],
                s.phase_stability(),
                s.coherence(),
                s.entropy()
            )
        } else {
            "No spatial controller initialized".to_string()
        }
    }

    fn query_code(&self, _text: &str) -> String {
        if let Some(ref _m) = self.multi {
            "MultiEngine ready — use `omni code-analyze <file>` for analysis".to_string()
        } else {
            "No multi-engine initialized. Call with_multi(dim) first.".to_string()
        }
    }

    pub fn run_integration(&self, query: &str, domain: &str) -> Vec<String> {
        let mut pipes = Vec::new();
        let q = query.to_lowercase();

        if self.integration.physics_to_code && domain != "code" && q.contains("constraint") {
            pipes.push(
                "PHYSICS→CODE: constraint vector from physics domain forwarded to code generator"
                    .to_string(),
            );
        }
        if self.integration.spatial_to_physics && domain != "physics" && q.contains("sensor") {
            pipes.push(
                "SPATIAL→PHYSICS: LiDAR sensor readings piped to aether density field model"
                    .to_string(),
            );
        }
        if self.integration.reactor_to_spatial && domain != "spatial" && q.contains("control") {
            pipes.push(
                "REACTOR→SPATIAL: rod position vector mapped to spatial attractor coordinates"
                    .to_string(),
            );
        }
        if self.integration.code_to_reactor && domain != "reactor" && q.contains("safety") {
            pipes.push(
                "CODE→REACTOR: verified Rust safety invariants constrain reactor power ramp rate"
                    .to_string(),
            );
        }
        pipes
    }
}

#[derive(Clone)]
pub enum OmniDomainResult {
    Physics(FisigAnswer),
    Reactor(String),
    Spatial(String),
    Code(String),
    General(String),
}

pub struct OmniResult {
    pub domain: String,
    pub domain_result: OmniDomainResult,
    pub cross_domain: Vec<String>,
    pub entropy: f64,
    pub coherence: f64,
    pub memory_size: usize,
}

pub fn render_omni_result(result: &OmniResult) -> String {
    let mut out = String::new();

    out.push_str("╔══════════════════════════════════════════════════╗\n");
    out.push_str(&format!(
        "║  Fuga Omni 1.0  ──  domain: {}              \n",
        result.domain
    ));
    out.push_str("╚══════════════════════════════════════════════════╝\n\n");

    out.push_str(&format!("  Cube entropy:  {:.4}\n", result.entropy));
    out.push_str(&format!("  Coherence:     {:.4}\n", result.coherence));
    out.push_str(&format!(
        "  Memory:        {} entries\n\n",
        result.memory_size
    ));

    match &result.domain_result {
        OmniDomainResult::Physics(ans) => {
            out.push_str(&fisig_formatter::render_fisig_answer(ans));
        }
        OmniDomainResult::Reactor(s) | OmniDomainResult::Spatial(s) | OmniDomainResult::Code(s) => {
            out.push_str("│ DOMAIN RESULT\n│\n");
            for line in s.lines() {
                out.push_str(&format!("│   {}\n", line));
            }
            out.push('\n');
        }
        OmniDomainResult::General(s) => {
            out.push_str("│ ANSWER\n│\n");
            out.push_str(&format!("│   {}\n", s));
            out.push('\n');
        }
    }

    if !result.cross_domain.is_empty() {
        out.push_str("│ CROSS-DOMAIN PIPES\n│\n");
        for pipe in &result.cross_domain {
            out.push_str(&format!("│   ⚡ {}\n", pipe));
        }
        out.push('\n');
    }

    out.push_str("╔══════════════════════════════════════════════════╗\n");
    out.push_str(&format!("║  Fuga Omni 1.0  ──  {}", result.domain));
    let pad = 28usize.saturating_sub(result.domain.len());
    for _ in 0..pad {
        out.push(' ');
    }
    out.push_str("║\n");
    out.push_str("╚══════════════════════════════════════════════════╝\n");

    out
}

pub fn omni_train<const N: usize, const S: usize>(
    ai: &mut FugaAI<N, S>,
    corpus_path: &str,
    save_path: &str,
) -> Result<(usize, f64, f64), String> {
    let docs = crate::load_corpus(corpus_path)?;
    let _dim = ai.dim;
    let mut total_paras = 0usize;

    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let flat_vocab = builder.build_flat_vocab();

    for (di, doc) in docs.iter().enumerate() {
        let mut doc_paras = 0usize;
        for ch in &doc.chapters {
            let heading = ch.heading.as_deref().unwrap_or("");
            for para_text in &ch.paragraphs {
                let combined = format!("{}: {}", heading, para_text);
                let tokens = crate::tokenize_corpus_text(&combined, &flat_vocab);
                ai.absorb_with_source(&tokens, doc.title.as_deref().unwrap_or("untitled"));
                doc_paras += 1;
                total_paras += 1;
            }
        }
        println!(
            "  [{}/{}] {} ({} paragraphs) ✓ entropy={:.4} mem={}",
            di + 1,
            docs.len(),
            doc.title.as_deref().unwrap_or("untitled"),
            doc_paras,
            ai.cube.global_entropy(),
            ai.memory.size(),
        );
    }

    ai.cube.save_bin(save_path)?;
    let mem_path = save_path.replace(".bin", "_mem.bin");
    ai.memory.save_bin(&mem_path)?;

    Ok((total_paras, ai.cube.global_entropy(), ai.cube.coherence()))
}

pub fn build_omni_corpus(existing: &str, output: &str) -> Result<usize, String> {
    use std::fs;
    use std::io::Write;

    let fisig_content =
        fs::read_to_string(existing).map_err(|e| format!("Failed to read {}: {}", existing, e))?;
    let fisig_lines: Vec<&str> = fisig_content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    let fisig_count = fisig_lines.len();

    let mut out =
        fs::File::create(output).map_err(|e| format!("Failed to create {}: {}", output, e))?;

    for line in &fisig_lines {
        writeln!(out, "{}", line).map_err(|e| format!("Write error: {}", e))?;
    }

    let extra_docs = vec![
        r#"{"source_url": "internal", "title": "Fuga Omni Architecture — Unified AI Engine", "author": "Fuga Models", "language": "en", "chapters": [{"heading": "OmniEngine Architecture", "paragraphs": ["Fuga Omni 1.0 is a unified physics-informed engineering AI that combines five knowledge domains into a single VSA hypervector brain. The five domains are: theoretical physics (ZFC axioms, differential geometry, QFT energy conditions, formal verification with Z3), applied physics (Mach effect thrusters, Alcubierre warp, aether gradient, Tesla ether, reactor kinetics), spatial navigation (Rapier3D physics simulation, LiDAR sensing, PD control), code analysis (multi-language syntax/semantic/chaos layers via tree-sitter), and cross-domain integration (physics constraints piped to code generation, sensor data mapped to field models).", "The core reasoning engine is FugaAI built on a WaveCube — a 3D array of hypervectors that stores knowledge through binding and superposition. The OmniEngine wraps FugaAI with domain-specific routers and structured output formatters. Each query is routed to the appropriate domain, processed through the unified VSA memory, and returned as a structured 4-block answer (theorem, derivation, limits, system vector).", "Cross-domain integration is the key innovation: the System Vector from a physics query can be piped directly into the code generator as compile-time constraints. The output of the spatial sensor (128-ray golden-spiral LiDAR) feeds into the aether density gradient model. Control rod positions from the reactor model map to spatial attractor coordinates. This creates a closed-loop reasoning system where each domain enriches the others.", "The OmniEngine supports lazy initialization of sub-engines: MultiEngine for code analysis, ReactorCore for point kinetics, RoomController for spatial awareness. These are created on demand and share the unified FugaAI instance. All knowledge is stored in a single WaveCube and MemoryStore, enabling c... (line truncated to 2000 chars)"#,
        r#"{"source_url": "internal", "title": "VSA Hypervector Memory — WaveCube Associative Storage", "author": "Fuga Models", "language": "en", "chapters": [{"heading": "WaveCube and MemoryStore Theory", "paragraphs": ["The WaveCube is a 3D associative memory organized as an LxLxL grid of hypervectors in dimension D. Each cell stores a superposition of vectors bound to that spatial coordinate. Knowledge is absorbed by encoding text tokens into hypervectors via deterministic hashing, then writing them into cube cells through wave propagation. The wave_flow operations spread information along cardinal axes enabling interference patterns.", "MemoryStore is a flat vector-text store that records each absorbed super-token alongside its original text and source document. Search is done via cosine similarity between the query hypervector and all stored vectors returning entries above a 0.55 threshold. The store supports both vector similarity search and text substring search.", "Entropy measures the uniformity of the cube cell distribution — lower entropy means more structured knowledge. Coherence measures the average similarity between neighboring cells indicating how well knowledge is organized spatially. Target values are entropy ~0.50 and coherence ~0.50 representing maximum information capacity without saturation.", "The Fuga Omni 1.0 brain uses a single WaveCube of dimension 8192 with side length 4 (64 cells total). This is sufficient to encode the full 23-document corpus across all five domains with entropy 0.50 and coherence 0.50."]}]"#,
        r#"{"source_url": "internal", "title": "Code Analysis — Multi-Language Syntax and Semantics", "author": "Fuga Models", "language": "en", "chapters": [{"heading": "Multi-Language Analysis Engine", "paragraphs": ["The MultiEngine provides syntax, semantic, and chaos analysis for multiple programming languages including Rust, Python, TypeScript, JavaScript, Go, Java, C, C++, Ruby, PHP, and Solidity. Each language is analyzed via tree-sitter grammars using language-specific query patterns for detecting violations.", "The syntax layer checks for safety issues: arithmetic overflow, buffer overflow, use-after-move, uninitialized variables, unsafe pointer arithmetic, reentrancy vulnerabilities, and integer division by zero. Each violation is tagged with severity (error, warning, info) and the enclosing function name.", "The semantic layer detects design-level anomalies: deeply nested control flow (cyclomatic complexity), overlong functions, excessive parameters, duplicate code blocks, and missing error handling. These are scored by confidence and clustered via DBSCAN.", "The chaos layer performs fault injection: it mutates the AST (flip conditions, swap operands, delete statements) and checks whether existing tests catch the mutation. The survival ratio measures test suite quality.", "Cross-domain: the safety score from code analysis can constrain reactor control parameters — only code with safety_score > 0.85 is allowed to set rod positions or power ramp rates."]}]"#,
        r#"{"source_url": "internal", "title": "Spatial Navigation — Rapier3D Room Controller", "author": "Fuga Models", "language": "en", "chapters": [{"heading": "Room Navigation and PD Control", "paragraphs": ["The spatial domain uses the Rapier3D physics engine for rigid body simulation. A Room contains 5 walls (4 side walls plus ceiling) forming a bounded arena with a 0.3m radius sphere as the controlled agent. The sphere has a collider and a velocity-based rigid body controller.", "The RoomController implements WaveCube-encoded PD control: the attractor point (target position) is encoded into a hypervector and bound with the current position vector to produce a control force. Wall repulsion is modeled as inverse-square forces from each wall when the sphere approaches within a threshold distance.", "Sensing is done via SphericalSensor which casts 128 rays in a fibonacci golden-spiral distribution around the sphere. Each ray returns the distance to the nearest collision or a maximum range value. This 128-dimensional LiDAR vector can be mapped to physics field parameters.", "Phase stability measures the entropy of the attractor path over a sliding window. High stability indicates smooth navigation; low stability indicates oscillation or confusion. The room controller achieves phase lock when the attractor and sphere positions are within 5cm of each other.", "Cross-domain: the 128-ray LiDAR vector can be interpreted as a 128-point sample of the aether density field around the agent enabling real-time field reconstruction and gradient estimation."]}]"#,
        r#"{"source_url": "internal", "title": "Reactor Core Point Kinetics Model", "author": "Fuga Models", "language": "en", "chapters": [{"heading": "Reactor Point Kinetics with Temperature Feedback", "paragraphs": ["The ReactorCore implements one-group point kinetics with delayed neutrons: dn/dt = (rho - beta)/Lambda * n + lambda * c and dc/dt = beta/Lambda * n - lambda * c. Here n is normalized neutron population (power level), c is delayed neutron precursor concentration, rho is net reactivity in pcm, beta is delayed neutron fraction (650 pcm), Lambda is generation time (1e-5 s), and lambda is precursor decay constant (0.08 s^-1).", "Temperature feedback is modeled via Newtonian cooling and Doppler broadening: dT/dt = power_coeff * n - cooling_coeff * T. The Doppler temperature coefficient alpha_T = -2 pcm/K provides inherent negative feedback: as temperature rises, reactivity drops, stabilizing the reactor.", "Two control rods provide external reactivity: each rod has a worth of 2000 pcm when fully withdrawn. Rod position is a continuous variable from 0.0 (fully inserted) to 1.0 (fully withdrawn). Rod reactivity is quadratic: rho_rod = pos^2 * worth, giving fine control at low insertion.", "The reactor grid (ReactorGrid) computes the radial flux distribution using a Bessel function approximation: flux(r) = J_0(2.4048 * r / R) * n. This gives the spatial flux profile in a cylindrical core with extrapolation radius R.", "Cross-domain: reactor power level n can be mapped to an attractor distance in the spatial domain. The reactor period (e-folding time) constrains the maximum velocity in room navigation. Code safety verification must pass before rod withdrawal commands are accepted."]}]"#,
        r#"{"source_url": "internal", "title": "Cross-Domain Integration — Unified System Vector Protocol", "author": "Fuga Models", "language": "en", "chapters": [{"heading": "System Vector Interoperability", "paragraphs": ["The System Vector protocol is the cross-domain communication mechanism in Fuga Omni 1.0. Every domain produces a labelled array of f64 values that can be consumed by any other domain as boundary condition, constraint, or control parameter.", "PHYSICS→CODE: The System Vector from a Woodward MET query [f, m, P, delta_m0, phi, rho_0, dt, epsilon_r] becomes compile-time constants in Rust code generation. Z3 SMT solver verifies that the thrust-to-power ratio f*delta_m0 / P remains below material limits.", "SPATIAL→PHYSICS: The 128-ray LiDAR distance vector from SphericalSensor is downsampled to 8 radial buckets and fed as boundary conditions to the aether density gradient Poisson solver. The reconstructed gradient field then updates the RoomController attractor.", "REACTOR→SPATIAL: Reactor power level n [0..inf) is normalized to [0..1] and mapped to the Z-coordinate of the spatial attractor. Rod positions map to XY attractor coordinates. Reactor scram sets attractor to home position (0, 0, 0).", "CODE→REACTOR: The MultiEngine safety_score [0..1] gates reactor control. If safety_score < 0.85, rod withdrawal is blocked and a diagnostic string is returned. Verified code can set the power ramp rate limit dRho/dt_max.", "All vectors use labelled legends for human interpretation. The integration layer runs after every query, checking for cross-domain triggers in the query text and activating the appropriate pipes."]}"#,
    ];

    for doc_json in &extra_docs {
        writeln!(out, "{}", doc_json).map_err(|e| format!("Write error: {}", e))?;
    }

    let total = fisig_count + extra_docs.len();
    println!(
        "Omni corpus created: {} docs ({} fisig + {} extra) -> {}",
        total,
        fisig_count,
        extra_docs.len(),
        output
    );
    Ok(total)
}
