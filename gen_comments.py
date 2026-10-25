import os, random

forum_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_corpus", "forum")  # кроссплатформенно (аудит 15.10)
os.makedirs(forum_dir, exist_ok=True)

comments = []
corpus_dirs = [os.path.join(os.path.dirname(os.path.abspath(__file__)), d) for d in ("src", "corpus_flat", "corpus_sources")]
for cd in corpus_dirs:
    if not os.path.exists(cd): continue
    for root, dirs, files in os.walk(cd):
        for f in files:
            if not f.endswith(('.rs', '.py', '.go', '.c', '.cpp', '.h')):
                continue
            fp = os.path.join(root, f)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        if line.startswith("// ") or line.startswith("# "):
                            clean = line.lstrip("/# ").strip()
                            if 15 < len(clean) < 250:
                                comments.append(clean)
            except:
                pass

comments = list(set(comments))
random.shuffle(comments)
comments = comments[:3000]

with open(os.path.join(forum_dir, "code_comments.txt"), "w", encoding="utf-8") as f:
    for c in comments:
        f.write(c)
        f.write("\n")
print(f"  ✓ {len(comments)} comments")
