
"""Legacy Absorb: спутник поглощает и чинит легаси-код.

Пайплайн:
  1. Клонирование легаси-репозитория (git clone --depth 1)
  2. Rust CodeIndexer индексирует его -> VSA-память (символы/сигнатуры)
  3. BIM регистрирует репозиторий как узел (legacy_module)
  4. Self-spawn рождает спутник на якорях поглощённого кода
  5. Sandbox «починка»: L1 Tree-Sitter (0 ERROR) + L2 py_compile/исполнение

Цель: механизм «поглощение легаси -> общая память -> автономная
починка/перегенерация» на реальном историческом коде (minGPT).
"""

from __future__ import annotations

from __future__ import annotations

import os
import shutil
import subprocess


import fuga_core
from fuga_core import CodeIndexer
from astral.bim_engine import BIMEngine, BIMNode
from astral.self_spawn import SelfSpawnEngine
from astral import sandbox


def clone_repo(url: str, dst: str) -> bool:
    if os.path.isdir(dst):
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    r = subprocess.run(["git", "clone", "--depth", "1", url, dst],
                       capture_output=True, text=True, timeout=180)
    return r.returncode == 0


def absorb(binder, repo_dir: str, repo_name: str, bim: BIMEngine,
           engine: SelfSpawnEngine) -> dict:
    """Поглощение репозитория: Rust-индекс -> память -> BIM -> спутник."""
    # 1. Rust-индексация (параллельный CodeIndexer)
    idx = CodeIndexer()
    items, lines = idx.index_dir(repo_dir, 100000)
    kinds = {}
    for kind, _name, _f in items:
        kinds[kind] = kinds.get(kind, 0) + 1
    symbols = sorted({n for _, n, _ in items if n})

    # 2. в код-память (через спутниковый CodeQueryEngine-путь: напрямую факты)
    from astral.core.memory import PersistentVSAMemory
    mem = PersistentVSAMemory(binder, directory=f"fuga_memory_{repo_name}")
    for n in symbols[:2000]:
        mem.add_fact("en", f"code:{n}", "legacy_symbol", repo_name,
                     dedupe_key=("legacy", repo_name, n))

    # 3. BIM-узел репозитория
    bim.add_component(
        BIMNode(repo_name, "legacy_module", vram_gb=0.0,
                throughput=f"{len(symbols)} symbols, {lines} lines"),
        deps=["vsa_memory", "code_ingestor"])

    # 4. спутник на якорях легаси-кода (топ-4 символа)
    anchors = symbols[:4] if len(symbols) >= 4 else (symbols + ["parse", "add"])[:4]
    task = f"работа с легаси {repo_name}"
    rec = engine.spawn_with_anchors(task, anchors)

    return {
        "repo": repo_name,
        "lines": lines,
        "symbols_total": len(symbols),
        "node_kinds": kinds,
        "anchors": anchors,
        "satellite": rec,
    }


def validate_legacy(repo_dir: str, file_rel: str) -> dict:
    """«Починка»: L1 Tree-Sitter + L2 py_compile поглощённого файла."""
    path = os.path.join(repo_dir, file_rel)
    if not os.path.exists(path):
        return {"file": file_rel, "error": "not found"}
    code = open(path, encoding="utf-8", errors="ignore").read()
    l1 = sandbox.level1_static(code, "python")
    l2 = {"ok": True, "output": ""}
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
        l2 = {"ok": True, "output": "py_compile OK"}
    except Exception as e:
        l2 = {"ok": False, "output": str(e)[:200]}
    return {"file": file_rel, "level1": l1["ok"], "errors_l1": l1["errors"],
            "level2": l2["ok"], "level2_msg": l2["output"]}


def main():
    binder = fuga_core.HybridBinder(2048)
    bim = BIMEngine(binder)
    bim.add_component(BIMNode("vsa_memory", "memory", vram_gb=0.3,
                              throughput="30125 facts"), [])
    bim.add_component(BIMNode("code_ingestor", "ingestor",
                              throughput="53-641 K-lines/s"), ["vsa_memory"])
    bim.add_component(BIMNode("fast_vsa", "rust_core",
                              throughput="156K steps/s"), ["vsa_memory"])

    engine = SelfSpawnEngine(binder, bim, None)  # ce=None: используем anchors
    # подмена: engine.spawn_with_anchors — добавим метод на лету
    import types

    def spawn_with_anchors(self, task, anchors):
        from astral.self_spawn import render_rust_satellite, _exec_satellite
        ops = [0, 1, 2, 3]
        name = "sat_" + task.replace(" ", "_")[:28]
        params_m = round((32768 * 256 * 5 + 256 * 256 * 5 +
                          256 * 32768 * 5) / 1e6, 1)
        code = render_rust_satellite(name, anchors, ops, params_m)
        path = os.path.join(self.out_dir, f"{name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        probe = b"\x5a" * 4096
        r = _exec_satellite(path, probe)
        ok = r["ok"] and len(r["stdout"]) == 4096
        self.bim.add_component(
            BIMNode(name, "satellite", params_m=params_m, vram_gb=0.0015,
                    throughput=f"ops={ops}", dims_in=32768, dims_out=32768),
            deps=[self.spawned[-1]["name"]] if self.spawned else
            ["vsa_memory", "fast_vsa"])
        rec = {"task": task, "name": name, "path": path,
               "params_m": params_m, "anchors": anchors, "ops": ops,
               "sandbox": "PASS" if ok else "FAIL"}
        self.spawned.append(rec)
        return rec

    SelfSpawnEngine.spawn_with_anchors = spawn_with_anchors

    # 1. клонируем minGPT (легаси-эталон)
    dst = "/tmp/opencode/mingpt"
    if not clone_repo("https://github.com/karpathy/minGPT.git", dst):
        print("клонирование не удалось — продолжаем на имеющихся данных")
    print(f"[legacy] репозиторий: minGPT ({'склонирован' if os.path.isdir(dst) else 'нет'})")

    # 2. поглощение
    res = absorb(binder, dst, "mingpt", bim, engine)
    print(f"[absorb] строк={res['lines']} символов={res['symbols_total']}")
    print(f"[absorb] якоря спутника: {res['anchors']}")
    print(f"[absorb] типы узлов AST: {res['node_kinds']}")
    print(f"[satellite] {res['satellite']['name']}: {res['satellite']['sandbox']}")

    # 3. починка: валидация поглощённого кода
    check = validate_legacy(dst, "mingpt/model.py")
    print(f"\n[repair-check] {check['file']}: L1={check['level1']} "
          f"L2={check['level2']} ({check.get('level2_msg','')[:40]})")

    print(f"\n[BIM] узлов после поглощения: {len(bim.nodes)}")


if __name__ == "__main__":
    main()
