#!/usr/bin/env python3
"""Fuga self-recursive agent loop — powered by own mask (necli -> fuga-web).
Task -> necli (self-brain via Fuga mask) -> code -> compile/run -> PASS/FAIL
-> absorb into agent_lessons.jsonl -> incremental JEPA retrain -> loop."""

import json, os, re, subprocess, sys, time, tempfile
from pathlib import Path

NECLI_DIR = Path(__file__).parent / "necli"
PYTHON_BIN = str(NECLI_DIR / ".venv" / "bin" / "python3") if (NECLI_DIR / ".venv" / "bin" / "python3").exists() else sys.executable
CWD = Path(__file__).parent.resolve()
MASK_CMD = ["cargo", "run", "--release", "--bin", "fuga-web"]
FUGA_WEB_PORT = "8080"


def mask_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8080/v1/models", timeout=1).close()
        return True
    except Exception:
        return False


def start_mask() -> subprocess.Popen:
    env = os.environ.copy()
    env["FUGA_WEB_PORT"] = FUGA_WEB_PORT
    print("[loop] starting fuga-web mask (self-powered brain) on port 8080 ...")
    proc = subprocess.Popen(
        MASK_CMD,
        cwd=CWD,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    for _ in range(30):
        if mask_running():
            print("[loop] mask ready (fuga-2.0 brain active at localhost:8080)")
            return proc
        time.sleep(1)
    print("[loop] WARNING: mask did not respond at port 8080; continuing anyway ...")
    return proc


def run_necli(task: str, workdir: Path, iter_: int) -> tuple[str, str, bool]:
    print(f"[loop iter {iter_}] task: {task}")
    result_path = Path(workdir) / f"agent_result_{iter_}.txt"
    # Use uv / python entry; headless --api fuga --model fuga-2.0
    # Write a minimal prompt file for headless run
    prompt_file = workdir / f"agent_prompt_{iter_}.txt"
    prompt_file.write_text(f"TASK: {task}\nWrite a self-contained Rust file that solves the task; include tests; output PASS/FAIL.\nGenerate only safe code.")
    cmd = [
        str(PYTHON_BIN),
        str(NECLI_DIR / "src/main.py"),
        "run",
        "--api", "fuga",
        "--model", "fuga-2.0",
        f"--workdir", str(workdir),
        str(prompt_file.read_text().strip()),
    ]
    # Actually the CLI expects a prompt argument; run from necli dir with uv
    necli_main = NECLI_DIR / "src/main.py"
    try:
        out = subprocess.run(
            [str(PYTHON_BIN), str(necli_main), "run", "--api", "fuga", "--model", "fuga-2.0", "--workdir", str(workdir), task],
            cwd=str(NECLI_DIR),
            env={**os.environ, "FUGA_WEB_PORT": FUGA_WEB_PORT},
            capture_output=True,
            text=True,
            timeout=300,
        )
        stdout_text = out.stdout
        stderr_text = out.stderr
        success = out.returncode == 0
        # Save result
        result_text = (
            f"TASK: {task}\n"
            f"SCRIPT: {stdout_text[-3000:] if len(stdout_text) > 3000 else stdout_text}\n"
            f"OUTPUT: {stdout_text[-1000:] if len(stdout_text) > 1000 else stdout_text}\n"
            f"STATUS: {'success' if success else 'failed'}\n"
            f"TIME: {time.time():.0}\n"
            f"ERROR: {stderr_text[-1000:] if len(stderr_text) > 1000 else stderr_text}\n"
        )
        result_path.write_text(result_text)
    except Exception as exc:
        stdout_text = str(exc)
        result_path.write_text(
            f"TASK: {task}\nSCRIPT: (no script)\nOUTPUT: \nSTATUS: failed\nERROR: {str(exc)[:500]}\n"
        )
    return str(result_path), stdout_text, True  # treat as process completed regardless


def absorb_result(path: str, code_override: str = "") -> bool:
    path_obj = Path(path)
    if not path_obj.exists():
        return False
    content = path_obj.read_text()
    # Append the full entry as a JSON line to agent_lessons.jsonl.
    # IMPORTANT: must match load_corpus() (src/lib.rs:157) which parses EACH
    # LINE as a JSON CorpusDoc {"title","author","language","chapters":[{heading,
    # paragraphs, number}]}. The OLD plain-text + ---END_DOC--- format made
    # every lesson fail serde_json::from_str and get silently dropped, so the
    # "incremental retrain" was a longstanding no-op.
    lesson_path = CWD / "agent_lessons.jsonl"
    task = ""
    script = ""
    output = ""
    status = ""
    error = ""
    lines = content.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("TASK: "):
            task = line[len("TASK: "):]
        elif line.startswith("SCRIPT: "):
            # SCRIPT block is multi-line in the result file (raw necli tail);
            # collect until a known trailer prefix.
            parts = [line[len("SCRIPT: "):]]
            j = i + 1
            while j < n and not any(
                lines[j].startswith(p)
                for p in ("OUTPUT: ", "STATUS: ", "ERROR: ", "TIME:")
            ):
                parts.append(lines[j])
                j += 1
            script = "\n".join(parts)
        elif line.startswith("OUTPUT: "):
            output = line[len("OUTPUT: "):]
        elif line.startswith("STATUS: "):
            status = line[len("STATUS: "):]
        elif line.startswith("ERROR: "):
            error = line[len("ERROR: "):]
        i += 1
    # Prefer the gate-clean extracted code (what actually passed compile +
    # relevance) over the raw tail which may carry banners/non-code.
    if code_override:
        script = code_override
    paragraphs = [
        f"TASK: {task}",
        f"RESULT: {status}",
        f"CODE: {script[:1500]}",
        f"OUTPUT: {output[:1500]}",
        f"ERROR: {error[:1500] if status == 'failed' else ''}",
    ]
    doc = {
        "title": task[:200],
        "author": "fuga-agent",
        "language": "rust",
        "chapters": [{
            "heading": "agent lesson",
            "number": 1,
            "paragraphs": paragraphs,
        }],
    }
    # Dedup by CODE: never re-absorb/re-inject a program already learned.
    # Anti-self-reinforcement guard — a repeatedly re-asked task (e.g. a
    # recurring test query like "factorial") must NOT append an identical lesson
    # every round (that would inflate agent_lessons.jsonl and amplify one
    # transition with no new information).
    def leaf_code(line: str) -> str:
        try:
            d = json.loads(line)
        except Exception:
            return ""
        for ch in (d.get("chapters") or []):
            for p in (ch.get("paragraphs") or []):
                if isinstance(p, str) and (p.startswith("CODE: ") or p.startswith("CODE:")):
                    return p.split(":", 1)[1].strip()
        return ""

    def norm_code(s: str) -> str:
        # Drop comment/marker lines (e.g. the mask's "// agent lesson" provenance
        # line) and collapse whitespace so semantically identical programs match.
        parts = [
            ln
            for ln in s.splitlines()
            if not (ln.strip().startswith("//") or ln.strip().startswith("#"))
        ]
        return " ".join((" ".join(parts)).split())

    code_leaf = norm_code(script or "")
    for line in lesson_path.read_text(encoding="utf-8").splitlines() if lesson_path.exists() else []:
        if leaf_code(line) and norm_code(leaf_code(line)) == code_leaf:
            print("[loop] ⚠ duplicate lesson (same code already in agent_lessons) — skip absorb+inject (anti self-reinforce)")
            return False
    with open(lesson_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"[loop] absorbed lesson -> agent_lessons.jsonl ({path_obj.stat().st_size} bytes)")
    return True


def inject_lesson_live(task: str, code: str) -> bool:
    """Push a gate-passed lesson into the LIVE searchable memory (via the mask's
    /v1/fuga/lesson endpoint) so it is immediately observable in answers/codegen
    without a restart. Persistence across restarts is handled by the mask reading
    agent_lessons.jsonl at startup.
    """
    import urllib.request
    try:
        req_body = json.dumps({"task": task, "code": code[:4000]}).encode()
        req = urllib.request.Request(
            "http://localhost:8080/v1/fuga/lesson",
            data=req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            resp = json.loads(r.read())
        print(f"[loop] lesson injected live: {resp.get('source_doc')} (mem={resp.get('memory_size')})")
        return bool(resp.get("ok"))
    except Exception as e:
        print(f"[loop] ⚠ live inject failed (mask down?): {e}")
        return False


def incremental_retrain():
    # Continue-train from existing fuga_hjepa.bin
    out_bin = CWD / "fuga_hjepa.bin"
    lesson_path = CWD / "agent_lessons.jsonl"
    if not lesson_path.exists() or lesson_path.stat().st_size < 50:
        print("[loop] no lessons yet, skip incremental retrain")
        return
    # Limit to recent 30 docs to keep it fast
    cmd = [
        "cargo", "run", "--release", "--bin", "fuga", "--",
        "jepa-train",
        "--load", str(CWD / "fuga_hjepa.bin"),
        "--corpus", str(lesson_path),
        "--out", str(out_bin),
        "--limit", "30",
        "--progress-every", "5",
    ]
    print("[loop] incremental retrain on absorbed lessons (continue from fuga_hjepa.bin) ...")
    res = subprocess.run(cmd, cwd=str(CWD), capture_output=True, text=True, timeout=600)
    # Don't block on failure; show result
    if res.returncode == 0:
        print("[loop] retrain OK -> fuga_hjepa.bin updated (self-improved)")
    else:
        print(f"[loop] retrain exit={res.returncode}; continuing (brain kept previous state).")
        # Show last lines of stderr/stdout for debugging
        last_err_lines = res.stderr.split("\n")[-8:] if res.stderr else []
        for ln in last_err_lines:
            print("    ", ln)


def write_lessons_as_rust(workdir: Path) -> Path:
    """Extract lesson code (CODE: paragraphs) into .rs files so the OLD
    generative path (TM/HTM/SDR + W via learn_structure) can learn them."""
    src_dir = Path(workdir) / "lessons_src"
    src_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    lesson_path = CWD / "agent_lessons.jsonl"
    for line in lesson_path.read_text(encoding="utf-8").splitlines() if lesson_path.exists() else []:
        try:
            d = json.loads(line)
        except Exception:
            continue
        for ch in (d.get("chapters") or []):
            for p in (ch.get("paragraphs") or []):
                if isinstance(p, str) and (p.startswith("CODE: ") or p.startswith("CODE:")):
                    code = p.split(":", 1)[1].strip()
                    if code:
                        (src_dir / f"lesson_{n:03d}.rs").write_text(code)
                        n += 1
    return src_dir


def train_tm_on_lessons(src_dir: Path) -> bool:
    """Old generative stack: learn the accepted lessons' structure into the W
    operator (train-tm --structure resumes fuga_stack_tm.bin, so prior TM/HTM
    cells, latent W and OWM projector P are preserved, not wiped). The mask's
    TemporalPredictor picks the updated W up on restart."""
    tm_path = CWD / "fuga_stack_tm.bin"
    if not src_dir.exists() or not any(src_dir.iterdir()):
        print("[loop] no lesson source files for TM/W structure learning — skip")
        return False
    cmd = [
        "cargo", "run", "--release", "--bin", "fuga", "--",
        "train-tm", str(src_dir),
        "--out", str(tm_path),
        "--cap", "8192", "--ctx", "4",
        "--structure", "--max-files", "30",
    ]
    print("[loop] training TM/W operator on lessons (resume fuga_stack_tm.bin) ...")
    res = subprocess.run(cmd, cwd=str(CWD), capture_output=True, text=True, timeout=600)
    if res.returncode == 0:
        print("[loop] TM/W OK -> fuga_stack_tm.bin updated (W learned lesson structure)")
        return True
    print(f"[loop] train-tm exit={res.returncode}; W kept previous state.")
    for ln in (res.stderr.split("\n")[-6:] if res.stderr else []):
        print("    ", ln)
    return False


def incremental_stack_retrain(workdir: Path):
    """Teach an accepted lesson through the WHOLE stack so the old technologies
    (TM/HTM/SDR + W operator, VSA via lesson-tier, H-JEPA) all internalize it:
    not only retrieval (tier/memory) but also the generative W operator."""
    write_lessons_as_rust(workdir)
    train_tm_on_lessons(Path(workdir) / "lessons_src")
    incremental_retrain()


def lesson_seed_prompt(src_dir: Path) -> str:
    """First fn signature from a lesson, used to seed the tm-gen probe.
    Carries through the return type and opening brace (`fn f(...) -> T {`),
    which the TM transition needs to depolarize a real code path."""
    for f in sorted(src_dir.glob("*.rs")):
        text = f.read_text(errors="replace")
        m = re.search(r"\bfn\s+\w+\s*\([^)]*\)\s*(?:->\s*[^{;{]+)?\s*\{", text)
        if m:
            return m.group(0).replace("{", "").strip()
        m = re.search(r"\bfn\s+\w+\s*\([^)]*\)", text)
        if m:
            return m.group(0)
    return ""


def probe_w_generation(prompt: str, cube: str = "fuga_stack.bin", steps: int = 30,
                       src_dir: Path | None = None, task: str = "") -> list[str]:
    """GENERATION probe: tm-gen is the mask's real sequential generator
    (fuga::tm_generate over the trained TM/W operator). Seeding with the lesson
    function signature and checking whether W regenerates its body tokens tells
    us if the accepted lesson is GENERATED by the old operator, not merely
    retrieved by the new lesson-tier. Returns the generated token sequence.

    With src_dir (lessons) + task (lesson body words), the two-speed bridge is
    engaged: H-JEPA task corridor (--task-sim) gates the TM autoregressor, so
    content comes from the corridor, order from the TM syntax graph."""
    if not prompt:
        return []
    cmd = [
        "cargo", "run", "--release", "--bin", "fuga", "--",
        "tm-gen", prompt,
        "--cube", cube,
        "--steps", str(steps),
    ]
    if src_dir is not None and task.strip():
        cmd += [
            "--vocab-dir", str(src_dir),
            "--task", task,
            "--task-sim", "0.15",
        ]
    try:
        res = subprocess.run(cmd, cwd=str(CWD), capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return []
    for ln in res.stdout.splitlines():
        m = re.match(r"^\s*Sequence:\s*(.*)$", ln)
        if m:
            return m.group(1).split()
    return []


def lesson_body_tokens(src_dir: Path) -> set[str]:
    toks: set[str] = set()
    for f in src_dir.glob("*.rs"):
        toks |= set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|[(){}[\],;:.]", f.read_text(errors="replace")))
    return toks


def extract_rust(text: str) -> str:
    """Pull the first Rust-looking fenced code block out of necli output."""
    import re
    # fenced blocks first
    blocks = re.findall(r"```(?:rust)?\s*(.*?)```", text, re.S)
    for b in blocks:
        if re.search(r"\b(fn |use |struct |impl |pub fn |#!\[|main\s*\()", b):
            return b.strip()
    # fallback: scan lines for a function signature and keep a contiguous block
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(pub\s+)?fn\s+\w+\s*\(", ln) or ln.strip().startswith("#!"):
            start = i
            break
    if start is None:
        return ""
    out = []
    for ln in lines[start:]:
        if ln.strip() == "" and out and ln.strip() == "" and out[-1].strip() == "":
            break
        out.append(ln)
    return "\n".join(out).strip()


def relevance_gate(task: str, code: str) -> tuple[float, bool]:
    """Structural relevance heuristic: how much of the task's meaningful
    vocabulary (identifiers/keywords) appears in the generated source.

    Returns (relevance 0..1, boolean). This guards the second failure mode:
    a compiled-but-unrelated fragment (e.g. MemoryMapping code from elsewhere
    in the corpus) must NOT be treated as a lesson for the requested task.
    """
    STOP = {
        "the", "a", "an", "and", "or", "of", "to", "in", "that", "it", "for",
        "generate", "safe", "rust", "function", "with", "compile", "report",
        "pass", "fail", "this", "is", "output", "only", "write", "into", "as",
        "on", "by", "from", "at",
    }
    import re
    task_tokens = {t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", task.lower())
                   if len(t) >= 3 and t not in STOP}
    if not task_tokens:
        return 1.0, True  # nothing structural to match against
    code_l = code.lower()
    hits = sum(1 for t in task_tokens if re.search(r"(?<![a-z0-9_])" + re.escape(t) + r"(?![a-z0-9_])", code_l))
    score = hits / len(task_tokens)
    # Require at least one task identifier to appear in the code, and a
    # meaningful overall overlap, to treat it as a relevant lesson.
    return round(score, 3), hits >= 1 and score >= 0.25


def rustc_gate(code: str, workdir: Path, tag: str) -> bool:
    """Gate: retrain/absorb happens ONLY if the generated Rust compiles.

    Mirrors the existing `cargo check --quiet` gate in src/main.rs (run loop).
    Returns True only on a clean compile.
    """
    if not code.strip():
        return False
    src = workdir / f"gate_{tag}.rs"
    src.write_text(code)
    out = subprocess.run(
        ["rustc", "--edition", "2021", "--crate-type", "bin", str(src), "-o", str(workdir / f"gate_{tag}.bin")],
        capture_output=True,
        text=True,
    )
    ok = out.returncode == 0
    errs = [l for l in out.stderr.splitlines() if "error" in l or "aborting" in l]
    if not ok:
        print(f"[loop gate {tag}] ✗ compile FAILED ({len(errs)} errors):")
        for l in errs[:6]:
            print(f"      {l}")
    else:
        print(f"[loop gate {tag}] ✓ compiles clean -> eligible for retrain")
    return ok


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fuga self-recursive agent loop via necli + own mask")
    parser.add_argument("task", nargs="?", default="Write a safe Rust hello-world that prints its PID and exits 0", help="agent task prompt")
    parser.add_argument("--iters", type=int, default=3, help="loop iterations (default 3)")
    parser.add_argument("--workdir", type=str, default=str(CWD / ".workdir"), help="work directory")
    args = parser.parse_args()

    # Ensure mask up
    mask_proc = None
    if not mask_running():
        mask_proc = start_mask()
        # Give mask time to load
        import time
        time.sleep(3)
    else:
        print("[loop] mask already running on port 8080")

    workdir_path = Path(args.workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    n_passed = 0
    n_dup = 0
    n_compiled_but_irrelevant = 0
    n_gated = 0
    for i in range(1, args.iters + 1):
        result_path, stdout_text, completed = run_necli(args.task, workdir_path, i)
        # GATE (compile) AND GATE (relevance): only a compilable program that
        # structurally relates to the requested task is absorbed and used for
        # retrain. Anything else is recorded to the diagnostic result file but
        # NEVER reaches agent_lessons.jsonl / fuga_hjepa.bin.
        code = extract_rust(stdout_text)
        tag = f"it{i}"
        compiles = rustc_gate(code, workdir_path, tag)
        rel_score, rel_ok = relevance_gate(args.task, code)
        print(f"[loop iter {i}] relevance={rel_score} relevant={rel_ok} compiles={compiles}")
        if compiles and rel_ok:
            if absorb_result(result_path, code):
                inject_lesson_live(args.task, code)
                n_passed += 1
                incremental_stack_retrain(workdir_path)
                seed = lesson_seed_prompt(workdir_path / "lessons_src")
                if seed:
                    body = lesson_body_tokens(workdir_path / "lessons_src")
                    task = " ".join(sorted(body))
                    gen = probe_w_generation(seed, cube="fuga_stack.bin",
                                             steps=30, src_dir=workdir_path / "lessons_src", task=task)
                    if body:
                        hits = sum(1 for t in gen if t in body)
                        print(f"[loop] W-generation probe: seed={seed!r}")
                        print(f"[loop]   generated: {' '.join(gen[:14]) or '(none)'}")
                        print(f"[loop]   W internalized lesson: {hits}/{len(gen)} generated tokens overlap lesson body — {'GENERATION, not retrieval' if hits >= 2 else 'weak/absent (see day-1 core problem: single sample vs 587K-corpus W)'}")
            else:
                n_dup += 1  # same code already learned — not a new lesson
        else:
            n_gated += 1
            if compiles and not rel_ok:
                n_compiled_but_irrelevant += 1
                print(f"[loop iter {i}] ⚠ compiles BUT not relevant to task — skipped (retrain-guard: relevance)")
            else:
                print(f"[loop iter {i}] ✗ no compilable, relevant Rust produced — skipped (no retrain)")
    print(f"[loop] finished {args.iters} iterations: {n_passed} NEW lesson(s) passed gate, {n_dup} duplicate(s) skipped (anti self-reinforce), {n_gated} gated-out ({n_compiled_but_irrelevant} compiled-but-irrelevant) — brain retrained ONLY on new compilable+relevant results")
    if n_gated > 0:
        print(f"[loop] NOTE: {n_gated} iteration(s) were NOT absorbed/retrained (compile or relevance gate)")


if __name__ == "__main__":
    main()
