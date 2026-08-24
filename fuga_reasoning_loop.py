from __future__ import annotations

import random
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

import fuga_core

MAX_SEG = 6          # шагов мысли на цепочку
SEG_TOKENS = 16


def tokenize(text: str, max_tokens: int = SEG_TOKENS) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-z0-9]+", text.lower())][:max_tokens]


def parse_solution(answer_text: str):
    """-> (segments[], final_answer_word)"""
    clean = re.sub(r"<<[^>]*>>", "", answer_text)
    parts = clean.split("####")
    steps = [ln.strip() for ln in parts[0].splitlines() if ln.strip()]
    final = parts[1].strip().lower() if len(parts) > 1 else ""
    return steps[:MAX_SEG], tokenize(final.replace(",", ""), 2)


class LatentReasoner(nn.Module):
    """z_{t+1} = KAN([z_t ; seg_t]): предсказание следующего шага мысли."""

    def __init__(self, dim=2048, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, h_state, seg_hv):
        return F.normalize(self.net(torch.cat([h_state, seg_hv], dim=-1)), dim=-1)


def unpack_rows(pk):
    w = torch.from_numpy(np.ascontiguousarray(pk)).long()
    b, W = w.shape
    return (((w.unsqueeze(-1) >> torch.arange(64)) & 1).reshape(b, W * 64)).float() * 2 - 1


@torch.no_grad()
def unbind_top_words(binder, z_batch, vocab_words, pos=1, topk=3):
    sgn = torch.sign(z_batch).cpu().numpy() > 0
    packed = np.zeros((sgn.shape[0], sgn.shape[1] // 64), dtype=np.uint64)
    for i in range(64):
        packed |= sgn[:, i::64].astype(np.uint64) << np.uint64(i)
    unb = np.asarray(binder.unbind_batch(packed, pos))
    scores = np.asarray(binder.score_items(unb, vocab_words))
    return scores.argsort(axis=1)[:, -topk:].flip(axis=1)


def main():
    random.seed(0)
    torch.manual_seed(0)

    # --- данные GSM8K ---
    tb_tr = pq.read_table("datasets/gsm8k/train.parquet").to_pylist()
    tb_te = pq.read_table("datasets/gsm8k/test.parquet").to_pylist()
    random.shuffle(tb_tr)

    def build(rows, limit):
        chains = []
        for r in rows[:limit]:
            steps, ans = parse_solution(r["answer"])
            if len(steps) < 2:
                continue
            chains.append((tokenize(r["question"]), steps, ans))
        return chains

    train_chains = build(tb_tr, 4000)
    test_chains = build(tb_te, 600)
    print(f"chains: train={len(train_chains)} heldout={len(test_chains)}")

    binder = fuga_core.HybridBinder(2048)

    def enc(text_tokens):
        pk = np.asarray(binder.bind_batch([text_tokens]))
        return unpack_rows(pk)[0]

    # энкодинг сегментов лениво по индексам
    all_segments = []
    index = []           # (chain_idx, seg_idx)
    for ci, (q, steps, ans) in enumerate(train_chains):
        for si, st in enumerate(steps):
            all_segments.append(st)
            index.append((ci, si))
    seg_pk = np.zeros((len(all_segments), 32), dtype=np.uint64)
    for s in range(0, len(all_segments), 2000):
        chunk = [tokenize(t) for t in all_segments[s:s + 2000]]
        chunk = [c if c else ["<empty>"] for c in chunk]
        seg_pk[s:s + 2000] = np.asarray(binder.bind_batch(chunk))
    print(f"segments encoded: {len(all_segments)}")

    q_hv_tr = torch.stack([enc(q) for q, _, _ in train_chains])
    vocab_words = sorted({w for (_, steps, _) in train_chains for st in steps
                          for w in tokenize(st)})
    print(f"vocab(train segments)={len(vocab_words)}")

    model = LatentReasoner()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    GAMMA = 0.5

    def chain_seg(ci):
        idxs = [k for k, (cj, _) in enumerate(index) if cj == ci]
        return seg_pk[idxs]

    # предвыборка индексов сегментов по цепочке
    from collections import defaultdict
    seg_of_chain = defaultdict(list)
    for k, (ci, si) in enumerate(index):
        seg_of_chain[ci].append(k)

    n = len(train_chains)
    for step in range(1201):
        ci = random.randrange(n)
        ks = seg_of_chain[ci]
        if len(ks) < 2:
            continue
        h = F.normalize(q_hv_tr[ci], dim=-1)
        loss_acc = 0.0
        n_steps = min(len(ks), MAX_SEG) - 1
        for t_i in range(n_steps):
            cur = unpack_rows(seg_pk[ks[t_i]:ks[t_i] + 1])
            nxt = unpack_rows(seg_pk[ks[t_i + 1]:ks[t_i + 1] + 1])
            pred = model(h, cur[0])
            loss_acc = loss_acc + (1 - F.cosine_similarity(pred, nxt[0], dim=-1).mean())
            h = GAMMA * h + (1 - GAMMA) * pred
            h = F.normalize(h, dim=-1)
        loss = loss_acc / max(n_steps, 1)
        std = torch.sqrt(pred.var(dim=0) + 1e-4)
        loss = loss + 1.0 * torch.relu(1.0 - std).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 300 == 0:
            print(f"step {step}: thought_loss={loss.item():.4f}")

    # --- оценка на held-out: teacher-forced cos + answer retrieval ---
    model.eval()
    cos_pred, cos_shuf = [], []
    ans_hits = 0
    ans_pool = []
    for q, steps, ans in test_chains[:400]:
        segs = [unpack_rows(np.asarray(binder.bind_batch(
            [tokenize(st) or ["<empty>"]])))[0] for st in steps]
        ans_pool.append(segs[-1])
    ans_pool_t = torch.stack(ans_pool)

    correct_rank = 0
    evaluated = 0
    with torch.no_grad():
        for qi, (q, steps, ans) in enumerate(test_chains[:400]):
            if len(steps) < 2:
                continue
            segs = [unpack_rows(np.asarray(binder.bind_batch(
                [tokenize(st) or ["<empty>"]])))[0] for st in steps]
            h = F.normalize(enc(q), dim=-1)
            n_steps = min(len(steps) - 1, MAX_SEG - 1)
            for t_i in range(n_steps):
                pred = model(h, segs[t_i])
                cos_pred.append(F.cosine_similarity(pred, segs[t_i + 1], dim=-1).item())
                shuf = ans_pool_t[random.randrange(len(ans_pool_t))]
                cos_shuf.append(F.cosine_similarity(pred, shuf, dim=-1).item())
                h = F.normalize(GAMMA * h + (1 - GAMMA) * pred, dim=-1)
            # финальный ответ: ранжирование среди пула ответов
            sims = ans_pool_t @ h
            gold_idx = qi % ans_pool_t.shape[0]
            if sims.argmax().item() == gold_idx:
                correct_rank += 1
            evaluated += 1

    print("\n===== HELD-OUT REASONING =====")
    print(f"cos(pred_thought, true_next): {np.mean(cos_pred):.4f}")
    print(f"cos(pred_thought, random):    {np.mean(cos_shuf):.4f}")
    print(f"answer rank acc@1 (pool={ans_pool_t.shape[0]}): {correct_rank/max(evaluated,1):.4f}")

    # --- VSA Concept Memory: факты S-R-O и O(1)-запрос ---
    print("\n===== CONCEPT MEMORY (SPO facts) =====")
    facts = [
        ("dog", "is_a", "animal"), ("cat", "is_a", "animal"),
        ("sparrow", "is_a", "bird"), ("dog", "can", "bark"),
        ("bird", "can", "fly"), ("animal", "has", "cells"),
    ]
    role_sub, role_rel, role_obj = "ROLE_subject", "ROLE_relation", "ROLE_object"
    from antitf.rust_bridge import packed_to_torch
    memory = binder.bind_batch(
        [[f"S:{s}", role_sub] for s, _, _ in facts] +
        [[f"R:{r}", role_rel] for _, r, _ in facts] +
        [[f"O:{o}", role_obj] for _, _, o in facts])
    memory_bp = packed_to_torch(np.asarray(memory))            # [n_facts*3, 2048]
    mem_bundle = torch.sign(memory_bp.sum(dim=0) + 1e-5).numpy().astype(np.float32)
    mem_bundle[mem_bundle == 0] = 1

    def query(subject, role):
        probe = binder.bind_batch([[f"S:{subject}", role_sub],
                                   [f"R:{role}", role_rel]])
        # unbind: XOR с компонентами запроса (probe packed -> биполярный)
        zb = torch.from_numpy(mem_bundle.astype(np.float32))
        qp = packed_to_torch(np.asarray(probe))[0]
        residual = (torch.sign(zb * qp + 1e-5).numpy().astype(np.uint64) > 0)[None, :]
        packed = np.zeros((1, 32), dtype=np.uint64)
        for i in range(64):
            packed |= residual[:, i::64].astype(np.uint64) << np.uint64(i)
        objects = sorted({o for _, rr, o in facts if rr == role and o})
        sc = np.asarray(binder.score_items(packed, objects))
        return objects[int(sc[0].argmax())]

    print(f"  dog is_a ? -> {query('dog', 'is_a')}   (ожидалось animal)")
    print(f"  cat is_a ? -> {query('cat', 'is_a')}   (ожидалось animal)")
    print(f"  bird can ? -> {query('bird', 'can')}   (ожидалось fly)")
    print(f"  dog can ?  -> {query('dog', 'can')}    (ожидалось bark)")


if __name__ == "__main__":
    main()
