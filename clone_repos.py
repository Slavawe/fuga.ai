#!/usr/bin/env python3
import os, json, subprocess, sys
from pathlib import Path

REPOS = {
    "code_c": [
        "https://github.com/redis/redis.git",
        "https://github.com/git/git.git",
    ],
    "code_cpp": [
        "https://github.com/facebook/rocksdb.git",
        "https://github.com/godotengine/godot.git",
    ],
    "code_rust": [
        "https://github.com/tokio-rs/tokio.git",
        "https://github.com/dimforge/rapier.git",
    ],
    "code_go": [
        "https://github.com/etcd-io/etcd.git",
        "https://github.com/gin-gonic/gin.git",
    ],
    "code_python": [
        "https://github.com/tiangolo/fastapi.git",
        "https://github.com/pallets/flask.git",
        "https://github.com/encode/httpx.git",
        "https://github.com/psf/requests.git",
        "https://github.com/pydantic/pydantic.git",
        "https://github.com/Textualize/rich.git",
        "https://github.com/Textualize/textual.git",
        "https://github.com/pallets/click.git",
        "https://github.com/encode/uvicorn.git",
        "https://github.com/encode/starlette.git",
        "https://github.com/pola-rs/polars.git",
        "https://github.com/faif/python-patterns.git",
        "https://github.com/cosmicpython/code.git",
    ],
}

EXTENSIONS = {
    "code_c":     [".c", ".h"],
    "code_cpp":   [".cpp", ".hpp", ".cc", ".cxx", ".hh"],
    "code_rust":  [".rs"],
    "code_go":    [".go"],
    "code_python":[".py"],
}

OUTPUT = "omni_corpus_repos.jsonl"
TEMP = Path("./temp_repos")

def clone(tag, repo_url):
    name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    dst = TEMP / name
    if dst.exists():
        print(f"  ⏭ {name} already cloned")
        return str(dst)
    print(f"  📦 Cloning {name}...", end=" ", flush=True)
    r = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dst)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"FAIL: {r.stderr.strip()}")
        return None
    print("done")
    return str(dst)

def walk_files(path, exts, tag):
    print(f"  🔍 Scanning {path}...", end=" ", flush=True)
    files = []
    path = Path(path)
    for f in path.rglob("*"):
        if f.suffix in exts and f.is_file():
            sz = f.stat().st_size
            if 50 < sz < 500_000:
                files.append(str(f.relative_to(TEMP)))
    print(f"{len(files)} files")
    return files

def main():
    TEMP.mkdir(exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as out:
        total = 0
        for tag, repo_list in REPOS.items():
            exts = EXTENSIONS[tag]
            for url in repo_list:
                path = clone(tag, url)
                if not path:
                    continue
                files = walk_files(path, exts, tag)
                for f in files:
                    abs_path = TEMP / f
                    try:
                        code = abs_path.read_text(encoding="utf-8", errors="replace")
                    except:
                        continue
                    if len(code) < 100:
                        continue
                    doc = {
                        "source_url": f"https://github.com/{'/'.join(url.rstrip('/').split('/')[-2:])}/blob/main/{'/'.join(f.split('/')[1:])}",
                        "title": f,
                        "author": tag,
                        "language": tag.split("_")[1],
                        "chapters": [
                            {
                                "heading": f"Code: {tag}",
                                "paragraphs": [
                                    code[:2000],
                                    code[2000:4000] if len(code) > 2000 else "",
                                    code[4000:6000] if len(code) > 4000 else "",
                                ]
                            }
                        ]
                    }
                    if doc["chapters"][0]["paragraphs"][0]:
                        out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        total += 1
                        if total % 100 == 0:
                            print(f"  📝 {total} documents written to {OUTPUT}")
        print(f"\n✅ Done: {total} documents in {OUTPUT}")

if __name__ == "__main__":
    main()
