from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")

from antitf.data_i18n import load_tatoeba_pairs
from antitf.selfplay import neural_alignment, selfplay_vsa_metrics
from antitf.vsa import VSAEncoder


def main() -> None:
    device = "cpu"
    pairs = load_tatoeba_pairs(max_pairs=20000)
    print(f"tatoeba ru-en pairs loaded: {len(pairs)}")
    for ru, en in pairs[:3]:
        print("  ", ru, "<->", en)

    split = int(len(pairs) * 0.8)
    train_pairs, test_pairs = pairs[:split], pairs[split:]

    vsa = VSAEncoder(vocab_size=256, hyper_dim=2048, seed=42).to(device)
    print("\n[pure-VSA self-play: role binding H_ru * R ~ H_en]")
    m = selfplay_vsa_metrics(vsa, train_pairs, test_pairs, device=device)
    for key in sorted(m):
        print(f"  {key}: {m[key]:.4f}")

    print("\n[neural alignment: H-JEPA latents RU<->EN]")
    n = neural_alignment(vsa, train_pairs, test_pairs, device=device, steps=60)
    for key in sorted(n):
        print(f"  {key}: {n[key]:.4f}")

    print("\n[item-memory structural binding + unbinding]")
    from antitf.item_memory import SimpleWordVocab, VSAItemMemory
    from antitf.selfplay import item_memory_unbinding_check, structured_retrieval

    vocab = SimpleWordVocab.build([p[0] for p in train_pairs] + [p[1] for p in train_pairs],
                                  max_size=50000)
    print(f"  vocab size: {len(vocab)}")
    mem = VSAItemMemory(vocab_size=len(vocab), hyper_dim=2048)
    ub = item_memory_unbinding_check(mem, vocab, [p[0] for p in test_pairs])
    for key in ("unbinding_acc@1", "positions_evaluated", "per_position_acc_first8"):
        print(f"  {key}: {ub[key]}")
    sr = structured_retrieval(mem, vocab, test_pairs)
    for key in sorted(sr):
        print(f"  {key}: {sr[key]:.4f}")


if __name__ == "__main__":
    main()
