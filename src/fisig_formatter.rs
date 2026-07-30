use std::collections::HashSet;
use crate::weaver::pattern_matcher::TokenInfo;
use crate::ai::core::FugaAI;

#[derive(Clone)]
pub struct FisigAnswer {
    pub core_theorem: String,
    pub derivation: Vec<String>,
    pub limits: Vec<String>,
    pub system_vector: Vec<f64>,
    pub source_docs: Vec<(String, f64)>,
    pub context_tag: String,
}

pub fn format_answer<const N: usize, const S: usize>(ai: &mut FugaAI<N, S>, query: &str, tokens: &[TokenInfo]) -> FisigAnswer {
    let output = ai.think(tokens);
    let mut seen = HashSet::new();
    let mut results: Vec<(String, f64, String)> = Vec::new();

    for st in &output.super_tokens {
        let mem_results = ai.memory.search(&st.vector, 8);
        for (_idx, sim, entry) in &mem_results {
            if seen.insert(entry.text.clone()) {
                results.push((entry.source_doc.clone(), *sim, entry.text.clone()));
            }
        }
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let source_docs: Vec<(String, f64)> = results.iter().map(|(d, s, _)| (d.clone(), *s)).collect();

    let q = query.to_lowercase();
    let context_tag = if q.contains("mach") || q.contains("woodward") {
        "met".to_string()
    } else if q.contains("alcubierre") || q.contains("warp") {
        "warp".to_string()
    } else if q.contains("aether") || q.contains("ether") || q.contains("newton") {
        "aether".to_string()
    } else if q.contains("tesla") {
        "tesla".to_string()
    } else if q.contains("zfc") || q.contains("set theory") || q.contains("axiom") || q.contains("formal logic") || q.contains("category") {
        "zfc".to_string()
    } else if q.contains("manifold") || q.contains("tensor") || q.contains("curvature") || q.contains("riemann") || q.contains("differential form") {
        "geometry".to_string()
    } else if q.contains("qft") || q.contains("quantum field") || q.contains("energy condition") || q.contains("anec") || q.contains("quantum inequality") {
        "qft".to_string()
    } else if q.contains("z3") || q.contains("formal verification") || q.contains("smt") || q.contains("llvm") || q.contains("rust verification") || q.contains("crucible") {
        "z3".to_string()
    } else if q.contains("physicoders") || q.contains("dual-core") || q.contains("resonant") {
        "physicoders".to_string()
    } else {
        "general".to_string()
    };

    let core_theorem = extract_core_theorem(&results, &q);

    let derivation = extract_derivation_steps(&results, &q);

    let limits = extract_limits(&results, &q);

    let system_vector = derive_system_vector(&results, &q);

    FisigAnswer {
        core_theorem,
        derivation,
        limits,
        system_vector,
        source_docs,
        context_tag,
    }
}

fn extract_core_theorem(results: &[(String, f64, String)], query: &str) -> String {
    let mut theorem = String::new();

    if let Some((_, _, best)) = results.first() {
        for line in best.lines() {
            let tl = line.trim().to_lowercase();
            if tl.contains("equation") || tl.contains("formula") || tl.contains("principle")
                || tl.contains("law") || tl.contains("theorem")
                || tl.contains("delta rho_0") || tl.contains("ds^2")
                || tl.contains("partial") || tl.contains("nabla")
                || tl.contains("integral") || tl.contains("epsilon")
            {
                theorem.push_str(line);
                theorem.push('\n');
            }
        }
    }

    let q = query.to_lowercase();
    if q.contains("mach") || q.contains("woodward") {
        if theorem.is_empty() {
            theorem.push_str(
                "Core Theorem: Transient Mass Fluctuation (Woodward Effect)\n\n\
                δρ₀ = 1/(4πGρ₀c²) · ∂²P/∂t² − 1/(4πGρ₀²c⁴) · (∂P/∂t)²\n\n\
                Variables:\n\
                δρ₀ — transient rest-mass density fluctuation [kg/m³]\n\
                ρ₀ — baseline mass density of dielectric [kg/m³]\n\
                P — power density deposited into element [W/m³]\n\
                G — Newtonian gravitational constant [6.674×10⁻¹¹ m³/kg·s²]\n\
                c — speed of light in vacuum [2.998×10⁸ m/s]\n\
                ∂/∂t — partial time derivative operator\n"
            );
        }
    } else if q.contains("alcubierre") || q.contains("warp") {
        if theorem.is_empty() {
            theorem.push_str(
                "Core Theorem: Alcubierre Warp Metric\n\n\
                ds² = −c²dt² + (dx − v_s·f(r_s)·dt)² + dy² + dz²\n\n\
                Variables:\n\
                ds² — spacetime interval [m²]\n\
                v_s — bubble velocity [m/s]\n\
                f(r_s) — shape function: [tanh(σ(r_s+R)) − tanh(σ(r_s−R))] / 2·tanh(σR)\n\
                r_s — radial distance from bubble center [(x−x_s)²+y²+z²]^½ [m]\n\
                σ — wall thickness parameter [m⁻¹]\n\
                R — bubble radius [m]\n"
            );
        }
    } else if q.contains("aether") || q.contains("ether") || q.contains("newton") && q.contains("gradient") {
        if theorem.is_empty() {
            theorem.push_str(
                "Core Theorem: Newtonian Aether Density Gradient\n\n\
                F_grav = −∇P_aether ∝ −∇ρ_aether\n\n\
                ρ_aether(r) = ρ₀ · exp(−GM / c²r)\n\n\
                Variables:\n\
                F_grav — gravitational force [N]\n\
                P_aether — aether pressure [Pa]\n\
                ρ_aether(r) — aether density at radius r [kg/m³]\n\
                ρ₀ — aether density at infinity [kg/m³]\n\
                M — central mass [kg]\n\
                G — gravitational constant [m³/kg·s²]\n\
                c — speed of light [m/s]\n\
                r — radial distance from mass center [m]\n"
            );
        }
    } else if q.contains("zfc") || q.contains("set theory") || q.contains("axiom") || q.contains("category") {
        if theorem.is_empty() {
            theorem.push_str(
                "Core Theorem: ZFC Axioms and Yoneda Lemma\n\n\
                ZFC = {Extensionality, Foundation, Specification, Pairing, Union,\n\
                       Replacement, Infinity, Power Set, Choice}\n\n\
                Yoneda Lemma: Nat(Hom_A(−,−), F) ≅ F(A)\n\n\
                Variables:\n\
                ZFC — Zermelo-Fraenkel set theory with Choice\n\
                Hom_A(−,−) — representable functor h_A = Hom(A, −)\n\
                F: C → Set — arbitrary functor to the category of sets\n\
                Nat(—,—) — set of natural transformations\n\
                ≅ — natural isomorphism\n"
            );
        }
    } else if q.contains("manifold") || q.contains("tensor") || q.contains("curvature") || q.contains("riemann") {
        if theorem.is_empty() {
            theorem.push_str(
                "Core Theorem: Riemann Curvature Tensor\n\n\
                R^i_{jkl} = ∂_k Γ^i_{jl} − ∂_l Γ^i_{jk} + Γ^i_{mk} Γ^m_{jl} − Γ^i_{ml} Γ^m_{jk}\n\n\
                Einstein Tensor: G_{μν} = R_{μν} − ½ R g_{μν}\n\n\
                Variables:\n\
                R^i_{jkl} — Riemann curvature tensor components\n\
                Γ^i_{jk} — Christoffel symbols of the Levi-Civita connection\n\
                G_{μν} — Einstein tensor (divergence-free)\n\
                R_{μν} — Ricci tensor R_{μν} = R^λ_{μλν}\n\
                R — scalar curvature R = g^{μν}R_{μν}\n\
                g_{μν} — metric tensor\n"
            );
        }
    } else if q.contains("qft") || q.contains("quantum field") || q.contains("energy condition") || q.contains("anec") {
        if theorem.is_empty() {
            theorem.push_str(
                "Core Theorem: Quantum Inequality (Ford-Roman Bound)\n\n\
                τ₀/π ∫_{-∞}^{∞} ⟨ρ(τ)⟩ / (τ² + τ₀²) dτ ≥ −3/(32π²τ₀⁴)\n\n\
                ANEC: ∫_γ T_{μν} k^μ k^ν dλ ≥ 0\n\n\
                Variables:\n\
                τ₀ — sampling time [s]\n\
                ⟨ρ(τ)⟩ — expected energy density at proper time τ\n\
                T_{μν} — stress-energy tensor\n\
                k^μ — null tangent vector along geodesic γ\n\
                λ — affine parameter along γ\n"
            );
        }
    } else if q.contains("z3") || q.contains("formal verification") || q.contains("smt") {
        if theorem.is_empty() {
            theorem.push_str(
                "Core Theorem: SMT Satisfiability and DPLL(T)\n\n\
                φ is satisfiable iff there exists model M such that M ⊧ φ\n\n\
                Rust Lifetime Constraint: l_a ≤ l_b for all borrow paths\n\n\
                Variables:\n\
                φ — first-order formula with background theories\n\
                M — model (assignment of values to all variables)\n\
                ⊧ — satisfaction relation\n\
                l_a, l_b — lifetime variables in borrow checker\n"
            );
        }
    }

    if theorem.is_empty() {
        theorem = results.first()
            .map(|(_, _, t)| t.lines().take(2).collect::<Vec<_>>().join("\n"))
            .unwrap_or_default();
    }

    theorem
}

fn extract_derivation_steps(results: &[(String, f64, String)], query: &str) -> Vec<String> {
    let mut steps = Vec::new();
    let q = query.to_lowercase();

    if q.contains("mach") || q.contains("woodward") {
        steps.push("Step 1: Start from Einstein field equations with Machian boundary conditions".to_string());
        steps.push("Step 2: Integrate local energy fluctuation over Hubble volume".to_string());
        steps.push("Step 3: Apply time derivative of energy flux through body surface".to_string());
        steps.push("Step 4: Expand to second order in power density P(t)".to_string());
        steps.push("".to_string());
        steps.push("Gμν + Λgμν = 8πG/c⁴ · Tμν".to_string());
        steps.push("Φ_local = −G ∫_V ρ_matter / |r−r'| dV ≈ −GM_H/R_H ≈ c²".to_string());
        steps.push("δm = 1/4πGρ₀c² · d²E/dt²".to_string());
        steps.push("δρ₀ = 1/4πGρ₀c² · ∂²P/∂t² − 1/4πGρ₀²c⁴ · (∂P/∂t)²".to_string());
    } else if q.contains("tesla") || q.contains("ether") {
        steps.push("Step 1: Postulate space-filling fluid with density ρ_e".to_string());
        steps.push("Step 2: Derive pressure gradient from vortex distribution".to_string());
        steps.push("Step 3: ∇·E_grav = −4πGρ_matter + Λ·Φ_e".to_string());
        steps.push("Step 4: Couple EM field energy density to ether scalar potential".to_string());
    } else if q.contains("zfc") || q.contains("set theory") {
        steps.push("Step 1: Start from ZFC axioms as primitive undefinables".to_string());
        steps.push("Step 2: Build von Neumann ordinals: 0=∅, 1={∅}, 2={∅,{∅}}".to_string());
        steps.push("Step 3: Define cardinals as initial ordinals".to_string());
        steps.push("Step 4: Apply Replacement to construct cumulative hierarchy V_α".to_string());
        steps.push("Step 5: Yoneda embedding: C → Set^{C^op} is fully faithful".to_string());
    } else if q.contains("manifold") || q.contains("tensor") {
        steps.push("Step 1: Define smooth manifold M with atlas {φ_α: U_α → R^n}".to_string());
        steps.push("Step 2: Construct tangent bundle TM with basis ∂_i".to_string());
        steps.push("Step 3: Equip with Riemannian metric g = g_{ij} dx^i ⊗ dx^j".to_string());
        steps.push("Step 4: Derive Christoffel symbols: Γ^k_{ij} = ½g^{kl}(∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij})".to_string());
        steps.push("Step 5: Compute Riemann tensor: R^i_{jkl} = ∂_k Γ^i_{jl} − ∂_l Γ^i_{jk} + Γ^i_{mk}Γ^m_{jl} − Γ^i_{ml}Γ^m_{jk}".to_string());
        steps.push("Step 6: Contract to Ricci: R_{ij} = R^k_{ikj}. Contract again: R = g^{ij}R_{ij}".to_string());
        steps.push("Step 7: Einstein tensor G_{μν} = R_{μν} − ½Rg_{μν} is divergence-free".to_string());
    } else if q.contains("qft") || q.contains("energy condition") || q.contains("anec") || q.contains("quantum inequality") {
        steps.push("Step 1: Define stress-energy tensor T_{μν} for quantum field φ".to_string());
        steps.push("Step 2: Compute renormalized expectation ⟨T_{μν}⟩ via point-splitting".to_string());
        steps.push("Step 3: Impose WEC: T_{μν}u^μu^ν ≥ 0 for all timelike u".to_string());
        steps.push("Step 4: Derive ANEC: ∫_γ T_{μν}k^μk^ν dλ ≥ 0 for null geodesics".to_string());
        steps.push("Step 5: Apply Ford-Roman QI: τ₀/π∫⟨ρ⟩/(τ²+τ₀²)dτ ≥ −3/(32π²τ₀⁴)".to_string());
        steps.push("Step 6: For Alcubierre τ₀ ≈ R/γv ⇒ constraint ρRτ₀ ≤ 1 in Planck units".to_string());
    } else if q.contains("z3") || q.contains("formal verification") || q.contains("smt") {
        steps.push("Step 1: Encode problem in SMT-LIB 2: declare-fun + assert + check-sat".to_string());
        steps.push("Step 2: Z3 applies DPLL(T): propositional SAT modulo theory solvers".to_string());
        steps.push("Step 3: Nelson-Oppen theory combination for uninterpreted functions + arithmetic".to_string());
        steps.push("Step 4: For Rust borrows: model lifetimes as linear constraints".to_string());
        steps.push("Step 5: Verify with: (assert (forall ((f Real) (m Real)) (=> (and (> f 0) (> m 0)) (< (* f m) 1e9))))".to_string());
        steps.push("Step 6: If UNSAT → reject parameter vector at compile time with diagnostic".to_string());
    } else if let Some((_, _, best)) = results.first() {
        for line in best.lines() {
            let tl = line.to_lowercase();
            if tl.contains("equation") || tl.contains("derive") || tl.contains("therefore")
                || tl.contains("thus") || tl.contains("from") || tl.contains("substituting")
                || tl.contains("integrating") || tl.contains("differentiating")
                || tl.contains("expanding") || tl.contains("solving")
            {
                steps.push(line.to_string());
            }
        }
    }

    if steps.is_empty() {
        steps.push("(Derivation sequence not found in memory; partial trace below)".to_string());
        if let Some((_, _, t)) = results.first() {
            for line in t.lines().take(4) {
                steps.push(line.to_string());
            }
        }
    }

    steps
}

fn extract_limits(results: &[(String, f64, String)], query: &str) -> Vec<String> {
    let mut limits = Vec::new();
    let q = query.to_lowercase();

    limits.push("Asymptotic constraints:".to_string());

    if q.contains("mach") || q.contains("woodward") {
        limits.push("ω → 0: δρ₀ → 0 (static field produces no mass fluctuation)".to_string());
        limits.push("ω → ∞: quantum inequality bounds apply; δt < t_Planck required".to_string());
        limits.push("ρ₀ → 0: singularity (cannot divide by zero density)".to_string());
        limits.push("Exotic matter constraint: ANEC must be satisfied for all null geodesics".to_string());
        limits.push("Lyapunov stability: neutrally stable if ∂P/∂t periodic; unstable if power self-amplifies".to_string());
    } else if q.contains("alcubierre") || q.contains("warp") {
        limits.push("v → c: energy required → ∞ (diverges at light speed)".to_string());
        limits.push("v > c: requires negative energy density ρ < −c²/R² in Planck units".to_string());
        limits.push("R → 0: singularity at bubble center (horizon forms)".to_string());
        limits.push("σ → ∞: infinite wall stress; quantum inequalities violated".to_string());
        limits.push("Energy dominance: Tμν violates weak, strong, and null energy conditions".to_string());
    } else if q.contains("aether") || q.contains("ether") || q.contains("newton") {
        limits.push("r → ∞: ρ_aether → ρ₀ (homogeneous background restored)".to_string());
        limits.push("r → 0: ρ_aether → 0 (perfect vacuum at mass center)".to_string());
        limits.push("M → M_max: gradient exceeds critical; aether cavitation (black hole)".to_string());
        limits.push("v → c: Lorentz invariance of aether violated at O(v²/c²)".to_string());
    } else if q.contains("zfc") || q.contains("set theory") || q.contains("category") {
        limits.push("Gödel Incompleteness: ZFC cannot prove its own consistency (2nd theorem)".to_string());
        limits.push("Continuum Hypothesis: independent of ZFC (Cohen forcing, 1963)".to_string());
        limits.push("Size limit: category of all sets is a proper class, not a set".to_string());
        limits.push("Large cardinal axioms: required for certain category-theoretic constructions".to_string());
        limits.push("Yoneda Lemma fails if Hom sets are not small".to_string());
    } else if q.contains("manifold") || q.contains("tensor") || q.contains("curvature") {
        limits.push("Metric signature: (− + + +) required for Lorentzian geometry; positive-definite for Riemannian".to_string());
        limits.push("Differentiability: C^k structure; curvature undefined if < C²".to_string());
        limits.push("Topology change: forbidden if metric is globally hyperbolic (Geroch theorem)".to_string());
        limits.push("Singularity: R_μνρσ R^μνρσ → ∞ at curvature singularity".to_string());
        limits.push("ADM energy: positive if dominant energy condition holds (Witten proof)".to_string());
    } else if q.contains("qft") || q.contains("quantum field") || q.contains("energy condition") || q.contains("anec") || q.contains("quantum inequality") {
        limits.push("WEC: T_μν u^μ u^ν ≥ 0 — violated by Casimir vacuum (ρ < 0 between plates)".to_string());
        limits.push("NEC: T_μν k^μ k^ν ≥ 0 — violated by any warp drive geometry".to_string());
        limits.push("SEC: (T_μν − ½Tg_μν) u^μ u^ν ≥ 0 — violation required for inflation".to_string());
        limits.push("DEC: T_μν u^μ is causal — violated by phantom energy (w < −1)".to_string());
        limits.push("Ford-Roman bound: negative energy pulses limited in magnitude × duration".to_string());
        limits.push("Quantum inequality: ρ ≥ −3/(32π²τ₀⁴) for sampling time τ₀".to_string());
    } else if q.contains("z3") || q.contains("formal verification") || q.contains("smt") {
        limits.push("SMT decidability: NRA (nonlinear real arithmetic) is decidable; NIA is not (Hilbert 10)".to_string());
        limits.push("DPLL(T) performance: exponential in worst case; heuristic-dependent".to_string());
        limits.push("Array theory: extensionality makes decision problem harder".to_string());
        limits.push("Bitvector: decidable but exponential in width; use incremental solving".to_string());
        limits.push("Alive2: only verifies LLVM IR, not source-level Rust semantics".to_string());
        limits.push("Crucible: sound for a subset of Rust; unsafe code requires manual annotations".to_string());
    }

    if let Some((_, _, best)) = results.first() {
        for line in best.lines() {
            let tl = line.to_lowercase();
            if tl.contains("bound") || tl.contains("constraint") || tl.contains("limit")
                || tl.contains("singularity") || tl.contains("cannot") || tl.contains("forbidden")
            {
                limits.push(line.to_string());
            }
        }
    }

    limits
}

fn derive_system_vector(results: &[(String, f64, String)], query: &str) -> Vec<f64> {
    let q = query.to_lowercase();

    if q.contains("mach") || q.contains("woodward") {
        vec![
            30000.0,    // f [Hz] — drive frequency
            4.0,        // m [kg] — stack mass
            250.0,      // P [W] — input power
            1e-5,       // δm₀ [kg] — mass fluctuation amplitude
            0.0,        // φ [rad] — phase angle (0° = max thrust)
            7800.0,     // ρ₀ [kg/m³] — PZT density
            25e-6,      // Δt [s] — switching period (1/f)
            8.85e-12,   // ε_r [F/m] — relative permittivity of PZT
        ]
    } else if q.contains("alcubierre") || q.contains("warp") {
        vec![
            100.0,      // R [m] — bubble radius
            3.0e8,      // v [m/s] — bubble velocity (1c)
            1e3,        // σ [m⁻¹] — wall sharpness
            -1.0e45,    // ρ_min [kg/m³] — required negative energy density
            0.0,        // x_s [m] — initial bubble position
        ]
    } else if q.contains("aether") || q.contains("ether") || q.contains("newton") {
        vec![
            5.972e24,   // M [kg] — central mass (Earth)
            6.371e6,    // r [m] — radius from center
            1.0,        // ρ₀ [kg/m³] — background aether density
            6.674e-11,  // G [m³/kg·s²]
            2.998e8,    // c [m/s]
        ]
    } else if q.contains("tesla") {
        vec![
            1.0e6,      // f [Hz] — Tesla coil frequency
            1.0e7,      // V [V] — electrostatic potential
            1.0e-3,     // L [m] — gradient scale length
            8.85e-12,   // ε₀ [F/m] — vacuum permittivity
        ]
    } else if q.contains("zfc") || q.contains("set theory") {
        vec![
            9.0,        // n_axioms — count of ZFC axioms
            1.0,        // consistent? (1=yes, 0=no per Gödel ~= 1)
            1.0,        // has_CH? (Continuum Hypothesis independent = 0.5)
            0.0,        // consistency_proof? (0 = Gödel 2nd)
            1.0,        // yoneda_valid? (1 = yes for small categories)
        ]
    } else if q.contains("manifold") || q.contains("tensor") {
        vec![
            4.0,        // dim — spacetime dimension
            20.0,       // n_riemann — independent Riemann components
            10.0,       // n_ricci — independent Ricci components
            0.0,        // R — scalar curvature (flat)
            0.0,        // Λ — cosmological constant
        ]
    } else if q.contains("qft") || q.contains("quantum field") || q.contains("energy condition") || q.contains("anec") || q.contains("quantum inequality") {
        vec![
            1.0,        // WEC (1 = satisfied, 0 = violated)
            0.0,        // NEC (0 = violated by Casimir)
            1.0,        // SEC (1 = satisfied for standard matter)
            1.0,        // DEC (1 = satisfied)
            0.5,        // ANEC_sat (0.5 = saturated by free fields)
            1.0e-3,     // τ₀ [s] — sampling time for QI bound
        ]
    } else if q.contains("z3") || q.contains("formal verification") || q.contains("smt") {
        vec![
            0.0,        // sat? (0 = SAT, 1 = UNSAT)
            1.0,        // verifiable? (1 = Z3 can handle)
            10.0,       // n_vars — number of SMT variables
            1.0e-6,     // timeout [s] — Z3 solving budget
        ]
    } else {
        vec![results.len() as f64]
    }
}

pub fn render_fisig_answer(answer: &FisigAnswer) -> String {
    let mut out = String::new();

    out.push_str("──────────────────────────────────────────────\n");
    out.push_str(&format!(" SOURCES: {}", answer.source_docs.iter()
        .map(|(d, s)| format!("{} [{:.3}]", d, s))
        .collect::<Vec<_>>().join(" · ")));
    out.push_str("\n──────────────────────────────────────────────\n\n");

    out.push_str("│ 1. CORE THEOREM / EQUATION\n");
    out.push_str("│\n");
    for line in answer.core_theorem.lines() {
        out.push_str(&format!("│   {}\n", line));
    }
    out.push('\n');

    out.push_str("│ 2. DERIVATION & PROOF\n");
    out.push_str("│\n");
    for (_i, step) in answer.derivation.iter().enumerate() {
        if step.is_empty() {
            out.push('\n');
        } else {
            out.push_str(&format!("│   {}\n", step));
        }
    }
    out.push('\n');

    out.push_str("│ 3. LIMITS & SINGULARITY CONSTRAINTS\n");
    out.push_str("│\n");
    for limit in &answer.limits {
        out.push_str(&format!("│   ∎ {}\n", limit));
    }
    out.push('\n');

    out.push_str("│ 4. SYSTEM VECTOR / TENSOR\n");
    out.push_str("│\n");
    out.push_str("│   [");
    for (i, v) in answer.system_vector.iter().enumerate() {
        if i > 0 { out.push_str(", "); }
        if *v == v.floor() && v.is_finite() {
            out.push_str(&format!("{:e}", v));
        } else {
            out.push_str(&format!("{:.6e}", v));
        }
    }
    out.push_str("]\n");
    out.push('\n');

    out.push_str("│   Parameter legend:\n");
    let labels: Vec<String> = match answer.context_tag.as_str() {
        "met" => vec!["f [Hz]", "m [kg]", "P [W]", "δm₀ [kg]", "φ [rad]", "ρ₀ [kg/m³]", "Δt [s]", "ε_r [F/m]"].into_iter().map(String::from).collect(),
        "warp" => vec!["R [m]", "v [m/s]", "σ [m⁻¹]", "ρ_min [kg/m³]", "x_s [m]"].into_iter().map(String::from).collect(),
        "aether" => vec!["M [kg]", "r [m]", "ρ₀ [kg/m³]", "G [m³/kg·s²]", "c [m/s]"].into_iter().map(String::from).collect(),
        "tesla" => vec!["f [Hz]", "V [V]", "L [m]", "ε₀ [F/m]"].into_iter().map(String::from).collect(),
        "zfc" => vec!["n_axioms", "consistent?", "has_CH?", "consistency_proof?", "yoneda_valid?"].into_iter().map(String::from).collect(),
        "geometry" => vec!["dim", "n_riemann", "n_ricci", "R [scalar curvature]", "Λ"].into_iter().map(String::from).collect(),
        "qft" => vec!["WEC", "NEC", "SEC", "DEC", "ANEC_sat", "τ₀ [s]"].into_iter().map(String::from).collect(),
        "z3" => vec!["sat?", "verifiable?", "n_vars", "timeout [s]"].into_iter().map(String::from).collect(),
        _ => (0..answer.system_vector.len()).map(|i| format!("p{}", i)).collect(),
    };
    for (i, lbl) in labels.iter().enumerate() {
        out.push_str(&format!("│   [{}]  {}\n", i, lbl));
    }

    out.push_str("──────────────────────────────────────────────\n");
    out
}
