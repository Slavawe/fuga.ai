from __future__ import annotations

import glob
import random
import sys

sys.path.insert(0, ".")

import fuga_core

from antitf.data_i18n import load_tatoeba_pairs
from antitf.item_memory import SimpleWordVocab
from antitf.linguistic_data import load_rucola, tokenize


def eval_filter(flt, valid, invalid, salads, vc, tc, label):
    av = len(flt.filter_batch(valid, vc, tc)[0]) / len(valid)
    ri = 1 - len(flt.filter_batch(invalid, vc, tc)[0]) / len(invalid)
    rs = 1 - len(flt.filter_batch(salads, vc, tc)[0]) / len(salads)
    bal = 0.5 * (av + (ri + rs) / 2)
    print(f"  [{label}] accept_valid={av:.3f} reject_bad={ri:.3f} "
          f"reject_salad={rs:.3f} balanced={bal:.3f}")


def main() -> None:
    random.seed(0)
    pairs = load_tatoeba_pairs(max_pairs=20000)
    ru_texts = [p[0] for p in pairs]

    lex = SimpleWordVocab.build(ru_texts, max_size=100000)
    base_words = list(lex.stoi.keys())
    transitions = []
    for s, a in load_rucola("in_domain_train"):
        if a == 1:
            t = tokenize(s)
            transitions += list(zip(t, t[1:]))
    for s in ru_texts:
        t = tokenize(s)
        transitions += list(zip(t, t[1:]))

    flt_base = fuga_core.RustLinguisticFilter()
    flt_base.load_wiktionary_vocab(base_words)
    flt_base.load_rucola_transitions(transitions)

    go_words = []
    with open("datasets/wiktionary/lexicon_ru_50k.txt", encoding="utf-8") as f:
        go_words = [w.strip() for w in f if w.strip()]
    flt_ext = fuga_core.RustLinguisticFilter()
    flt_ext.load_wiktionary_vocab(base_words + go_words)
    flt_ext.load_rucola_transitions(transitions)
    print(f"[lexicons] base={flt_base.vocab_size()}  +go-fetcher={flt_ext.vocab_size()}")

    dev = load_rucola("in_domain_dev") + load_rucola("out_of_domain_dev")
    valid = [s for s, a in dev if a == 1]
    invalid = [s for s, a in dev if a == 0]
    salads = flt_base.make_word_salad_negatives(valid, n_shuffles=1)

    print("[filter quality @0.6/0.3]")
    eval_filter(flt_base, valid, invalid, salads, 0.6, 0.3, "base")
    eval_filter(flt_ext, valid, invalid, salads, 0.6, 0.3, "go-lex")

    print("[wiktionary JSONL streaming (kaikki format)]")
    path = "datasets/wiktionary/sample_kaikki.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for w in go_words[:10000]:
            w2 = w.replace('"', "")
            f.write('{"word": "%s", "lang": "Russian"}\n' % w2)
    n = flt_base.load_wiktionary_dump_jsonl(path)
    print(f"  streamed {n} wordforms -> vocab={flt_base.vocab_size()}")

    print("[AST grammar filter: python code]")
    files = [pth for pth in sorted(glob.glob("antitf/*.py"))
             if len(open(pth, encoding="utf-8").read()) > 3000][:6]
    corpus = [open(pth, encoding="utf-8").read() for pth in files]
    ast = fuga_core.RustASTGrammarFilter()
    edges = ast.collect_ast_edges(corpus, lang=0)
    ast.load_ast_grammar_rules(edges)
    print(f"  learned edges from {len(corpus)} files: {len(edges)} raw, {ast.rules_size()} unique")

    good_code = corpus[0][-2500:]
    # порча уровня токенов внутри функций (структурный мусор, не строки)
    toks = good_code.split()
    random.shuffle(toks)
    bad_code = " ".join(toks)
    garbage = "".join(random.choice("(){ }=,:def return self._x1") for _ in range(1500))
    s_good = ast.score_ast_acceptability(good_code, lang=0)
    s_bad = ast.score_ast_acceptability(bad_code, lang=0)
    s_gar = ast.score_ast_acceptability(garbage, lang=0)
    print(f"  edge coverage: intact={s_good:.3f} shuffled-lines={s_bad:.3f} garbage={s_gar:.3f}")

    print("[pipeline: filter -> bind]")
    binder = fuga_core.HybridBinder(2048)
    ok, bad = flt_ext.filter_batch(valid + salads, 0.6, 0.3)
    hv = binder.bind_batch([tokenize(w) for w in ok])
    print(f"  {len(valid)+len(salads)} -> bind {len(ok)}, rejected-as-negatives {len(bad)}, hv={hv.shape}")


if __name__ == "__main__":
    main()
