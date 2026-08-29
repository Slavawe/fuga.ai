
"""Rewriter: полная самопереписка легаси-модулей с циклом самокоррекции.

Правила модернизации (заменяют экзотические аппроксимации внимания
на нативный fused F.scaled_dot_product_attention):
  - performer: FAVOR+ / ядерное внимание -> SDPA
  - minGPT:    ручной causal-attention -> SDPA

Цикл: rewrite -> L1 Tree-Sitter -> L2 исполнение (import + forward)
-> PASS: готово / FAIL: фиксируем ошибку, корректируем, повтор.
"""

from __future__ import annotations

from __future__ import annotations

import sys


import fuga_core
from astral import sandbox

# ---- правила модернизации: (маркер-блок, замена) ----
PERFORMER_FAVOR = (
    """        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out""",
    """        # modernized: native fused SDPA instead of FAVOR+ kernel approximation
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return out""",
)

MINGPT_ATTN = (
    """        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)""",
    """        # modernized: fused scaled dot-product attention (PyTorch 2.0+)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)""",
)

RULES = {
    "performer_favor": PERFORMER_FAVOR,
    "mingpt_attention": MINGPT_ATTN,
}


def rewrite_module(src: str, rule_name: str) -> tuple[str, int]:
    """Применяет правило ко ВСЕМ вхождениям; возвращает (код, число замен)."""
    old, new = RULES[rule_name]
    n = src.count(old)
    return src.replace(old, new), n


def self_rewrite_loop(src_path: str, out_path: str, rule_name: str,
                      run_check=None, max_iters: int = 3) -> dict:
    """Цикл: rewrite -> validate -> при ошибке корректируем (добавляем
    недостающий импорт F, если нужно) -> повтор."""
    code = open(src_path, encoding="utf-8").read()
    history = []
    for it in range(max_iters):
        new_code, n = rewrite_module(code, rule_name)
        if n == 0:
            return {"ok": False, "reason": f"rule {rule_name} не найден в коде",
                    "history": history}
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        l1 = sandbox.level1_static(new_code, "python")
        if not l1["ok"]:
            history.append(f"iter{it}: L1 FAIL ({l1['errors']} ERROR)")
            continue
        if run_check is not None:
            try:
                result = run_check(out_path)
                if result["ok"]:
                    return {"ok": True, "iters": it + 1, "replacements": n,
                            "run": result, "history": history}
                history.append(f"iter{it}: RUN FAIL: {result.get('error','')[:100]}")
            except Exception as e:
                history.append(f"iter{it}: RUN EXC: {str(e)[:100]}")
                # самокоррекция: гарантируем импорт F (torch.nn.functional)
                if "F." in new_code and "import torch.nn.functional as F" not in new_code \
                        and "import torch.nn.functional" not in new_code:
                    code = new_code.replace(
                        "import torch.nn as nn",
                        "import torch.nn as nn\nimport torch.nn.functional as F")
                    continue
        else:
            return {"ok": True, "iters": it + 1, "replacements": n,
                    "history": history}
    return {"ok": False, "reason": "не сошлось за итерации", "history": history}


def run_performer(out_path: str) -> dict:
    import importlib.util
    import torch
    repo_root = "/tmp/opencode/performer"
    if repo_root not in sys.path:
    spec = importlib.util.spec_from_file_location("mod_performer", out_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.Performer(dim=8, depth=1, heads=2, dim_head=4, causal=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 6, 8))
    return {"ok": True, "shape": tuple(out.shape)}


def run_mingpt(out_path: str) -> dict:
    import importlib.util
    import torch
    repo_root = "/tmp/opencode/mingpt"
    if repo_root not in sys.path:
    spec = importlib.util.spec_from_file_location("mod_gpt", out_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = type("C", (), {"n_layer": 1, "n_head": 2, "n_embd": 16,
                         "block_size": 16, "vocab_size": 64,
                         "model_type": None, "embd_pdrop": 0.1,
                         "resid_pdrop": 0.1, "attn_pdrop": 0.1,
                         "bias": True})()
    model = mod.GPT(cfg)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 8, dtype=torch.long))
    if isinstance(out, tuple):
        out = out[0]
    return {"ok": True, "shape": tuple(out.shape)}


def main():
    print("=== PERFOMER: FAVOR+ -> нативный SDPA ===")
    r1 = self_rewrite_loop(
        "/tmp/opencode/performer/performer_pytorch/performer_pytorch.py",
        "/tmp/opencode/performer_modernized.py",
        "performer_favor", run_check=run_performer)
    print(f"  {r1.get('ok')} | итераций {r1.get('iters')} | замен {r1.get('replacements')} "
          f"| run {r1.get('run', {}).get('shape')} | история {r1.get('history')}")

    print("\n=== MINGPT: полная переписка модуля -> SDPA ===")
    r2 = self_rewrite_loop(
        "/tmp/opencode/mingpt/mingpt/model.py",
        "/tmp/opencode/mingpt_model_modernized.py",
        "mingpt_attention", run_check=run_mingpt)
    print(f"  {r2.get('ok')} | итераций {r2.get('iters')} | замен {r2.get('replacements')} "
          f"| run {r2.get('run', {}).get('shape')} | история {r2.get('history')}")


if __name__ == "__main__":
    main()
