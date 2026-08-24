from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from antitf.data_i18n import load_tatoeba_pairs
from antitf.item_memory import SimpleWordVocab, VSAItemMemory
from antitf.jepa import HJEPAPredictor
from antitf.rust_bridge import get_binder, packed_to_torch, rust_unbinding_check, tokens_to_items


def main() -> None:
    device = "cpu"
    torch.manual_seed(0)

    binder = get_binder(2048)
    if binder is None:
        raise SystemExit("fuga_core не собран: maturin develop --release в fuga-core/")
    print(f"fuga_core HybridBinder: {binder.bits()} бит")

    pairs = load_tatoeba_pairs(max_pairs=6000)
    split = int(len(pairs) * 0.85)
    train_pairs, test_pairs = pairs[:split], pairs[split:]
    vocab = SimpleWordVocab.build(
        [p[0] for p in train_pairs] + [p[1] for p in train_pairs], max_size=50000)
    print(f"pairs: {len(train_pairs)}/{len(test_pairs)}  vocab: {len(vocab)}")

    print("\n[1] rust unbinding (packed u64, все кандидаты словаря)")
    ub = rust_unbinding_check(binder, vocab, [p[0] for p in test_pairs])
    print(f"  acc@1={ub['unbinding_acc@1']:.4f} ({ub['positions_evaluated']} позиций)"
          f" per_pos={ub['per_position_first4']}")

    print("\n[2] bench encode 2000 sentences")
    sentences = [tokens_to_items(vocab.encode(t, 32)) for t in [p[0] for p in test_pairs[:2000]]]
    py_mem = VSAItemMemory(vocab_size=len(vocab), hyper_dim=2048)
    t0 = time.perf_counter()
    hv_rust = np.asarray(binder.bind_batch(sentences))
    t_rust = time.perf_counter() - t0
    t0 = time.perf_counter()
    ids_py = torch.tensor([vocab.encode(t, 32) for t in [p[0] for p in test_pairs[:2000]]])
    hv_py = py_mem.encode_structured_sequence(ids_py)
    t_py = time.perf_counter() - t0
    print(f"  rust(packed u64)={t_rust * 1000:.1f}ms | python(f32)={t_py * 1000:.1f}ms"
          f" | x{t_py / max(t_rust, 1e-9):.1f}")

    print("\n[3] self-play: предсказание замаскированного слова из Rust-HV")
    seq = 16

    def make_ds(texts):
        rows = [vocab.encode(t, seq) for t in texts]
        hv = packed_to_torch(np.asarray(binder.bind_batch([tokens_to_items(r) for r in rows])))
        return hv, torch.tensor(rows)

    hv_tr, y_tr = make_ds([p[0] for p in train_pairs])
    hv_te, y_te = make_ds([p[0] for p in test_pairs])

    model = HJEPAPredictor(hyper_dim=2048, latent_dim=256)
    head = torch.nn.Linear(256, len(vocab))
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    mask_pos = 3
    n = hv_tr.shape[0]
    for step in range(200):
        idx = torch.randint(0, n, (128,))
        z = model.encode(hv_tr[idx])
        logits = head(z)
        gold = y_tr[idx][:, mask_pos]
        keep = gold != 0
        if keep.any():
            loss = F.cross_entropy(logits[keep], gold[keep])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step in (0, 99, 199):
            with torch.no_grad():
                zte = model.encode(hv_te)
                pred = head(zte).argmax(dim=1)
                gte = y_te[:, mask_pos]
                m = gte != 0
                acc = (pred[m] == gte[m]).float().mean().item()
            print(f"  step {step}: ce={loss.item():.4f} slot-{mask_pos} acc@1={acc:.4f}")




if __name__ == "__main__":
    main()
