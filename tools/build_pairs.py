import os, json, re

dirs = ["temp_repos", "text_corpus_processed", "text_batches", "text_done"]
exts = {".rs", ".py", ".js", ".ts", ".c", ".cpp", ".h", ".go", ".java"}
pairs = []

for d in dirs:
    for root, dirs_, files in os.walk(d):
        dirs_[:] = [x for x in dirs_ if not x.startswith('.') and x not in ('node_modules','target','thirdparty')]
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext not in exts:
                continue
            path = os.path.join(root, f)
            try:
                src = open(path, errors='ignore').read()
            except:
                continue
            lines = src.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                # detect docstring start
                doc = None
                if line.startswith("///") or line.startswith("//!"):
                    doc_lines = []
                    while i < len(lines) and lines[i].strip().startswith("///") or (i < len(lines) and lines[i].strip().startswith("//!")):
                        dl = lines[i].strip()
                        if dl.startswith("///"):
                            doc_lines.append(dl[3:].strip())
                        elif dl.startswith("//!"):
                            doc_lines.append(dl[3:].strip())
                        i += 1
                    doc = " ".join(doc_lines)
                elif line.startswith("/**") or line.startswith("/*"):
                    doc_lines = []
                    while i < len(lines) and "*/" not in lines[i]:
                        dl = lines[i].strip().lstrip("/**").lstrip("/*").lstrip("*").strip()
                        doc_lines.append(dl)
                        i += 1
                    if i < len(lines):
                        dl = lines[i].strip().rstrip("*/").strip()
                        doc_lines.append(dl)
                        i += 1
                    doc = " ".join(doc_lines)
                elif line.startswith("# ") or line.startswith("#("):
                    doc_lines = []
                    while i < len(lines) and (lines[i].strip().startswith("# ") or lines[i].strip().startswith("#(")):
                        dl = lines[i].strip().lstrip("#").strip()
                        doc_lines.append(dl)
                        i += 1
                    doc = " ".join(doc_lines)

                if doc and len(doc) >= 10:
                    # collect code block after docstring
                    code_lines = []
                    depth = 0
                    start = i
                    while i < len(lines) and (depth > 0 or len(code_lines) < 3):
                        cl = lines[i]
                        code_lines.append(cl)
                        depth += cl.count('{') - cl.count('}')
                        i += 1
                        if depth <= 0 and len(code_lines) >= 5:
                            break
                    code = "\n".join(code_lines).strip()
                    if 20 <= len(code) <= 2000:
                        pairs.append({"doc": doc, "code": code, "source": path})
                else:
                    i += 1

out = "corpus_doc_code_pairs.jsonl"
with open(out, "w") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"Generated {len(pairs)} docstring→code pairs → {out}")
