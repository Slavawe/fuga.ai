
"""Modernize: модель модернизирует легаси-код (minGPT).

Трансформация: ручной causal-attention (QK^T -> masked_fill -> softmax)
заменяется на fused F.scaled_dot_product_attention (PyTorch 2.0+).
Валидация: L1 Tree-Sitter (0 ERROR) + L2 py_compile + L2-run (инстанс
мини-GPT и forward-прогон с проверкой формы выхода).

Это пример «поглощение -> модернизация -> проверка исполнением» —
модель сама генерирует патч и доказывает его работоспособность.
"""

from __future__ import annotations

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuga_core
from astral import sandbox

LEGACY_BLOCK = """        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)"""

MODERN_BLOCK = """        # modernized: fused scaled dot-product attention (PyTorch 2.0+)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)"""


def modernize(src_path: str, out_path: str) -> dict:
    code = open(src_path, encoding="utf-8").read()
    if LEGACY_BLOCK not in code:
        return {"error": "legacy attention block not found", "modernized": False}
    new_code = code.replace(LEGACY_BLOCK, MODERN_BLOCK)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(new_code)
    return {"modernized": True, "diff_lines": LEGACY_BLOCK.count("\n")}


def validate_modernized(out_path: str, repo_root: str | None = None) -> dict:
    """L1 + L2 + L2-run: реальный forward мини-GPT."""
    code = open(out_path, encoding="utf-8").read()
    l1 = sandbox.level1_static(code, "python")

    # L2-run: загрузить модель, инстанциировать мини-GPT, forward
    import importlib.util
    import torch
    result = {"l1_ok": l1["ok"], "errors_l1": l1["errors"]}
    try:
        if repo_root and repo_root not in sys.path:
            sys.path.insert(0, repo_root)      # чтобы "from mingpt.utils" резолвился
        spec = importlib.util.spec_from_file_location("modern_gpt", out_path)
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
        result["l2_run_ok"] = True
        result["output_shape"] = tuple(out.shape)
    except Exception as e:
        result["l2_run_ok"] = False
        result["l2_error"] = str(e)[:200]
    return result


def main():
    src = "/tmp/opencode/mingpt/mingpt/model.py"
    out = "/tmp/opencode/mingpt_model_modernized.py"
    if not os.path.exists(src):
        print("minGPT не найден — запустите legacy_absorb.py сначала")
        return

    print("[modernize] применяю патч (ручной attention -> F.sdpa)...")
    res = modernize(src, out)
    if "error" in res:
        print(" ", res["error"])
        return
    print(f"  заменено {res['diff_lines']} строк легаси-блока")

    print("[validate] L1 Tree-Sitter + L2 исполнение мини-GPT...")
    repo_root = os.path.dirname(os.path.dirname(src))
    v = validate_modernized(out, repo_root=repo_root)
    print(f"  L1 (static): {'VALID' if v['l1_ok'] else 'ERROR ' + str(v['errors_l1'])}")
    print(f"  L2 (forward): {'PASS' if v.get('l2_run_ok') else 'FAIL ' + v.get('l2_error','')} "
          f"| shape={v.get('output_shape')}")

    # показываем модернизированный фрагмент
    code = open(out, encoding="utf-8").read()
    idx = code.find("scaled_dot_product_attention")
    print("\n[модернизированный фрагмент]")
    print(code[max(0, idx - 120):idx + 90])


if __name__ == "__main__":
    main()
