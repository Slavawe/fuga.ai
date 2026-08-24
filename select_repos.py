#!/usr/bin/env python3
import json, os
from pathlib import Path

TEMP = Path("./temp_repos")

# Key files per repo — most representative, medium-large, architecturally significant
SELECT = {
    "redis/src":  ["server.c", "db.c", "networking.c", "aof.c", "rdb.c", "cluster.c",
                   "replication.c", "sentinel.c", "scripting.c", "module.c",
                   "t_list.c", "t_set.c", "t_hash.c", "t_zset.c", "t_stream.c",
                   "ae.c", "anet.c", "ziplist.c", "sds.c", "adlist.c"],
    "git":        ["builtin/fetch.c", "builtin/push.c", "builtin/commit.c",
                   "builtin/log.c", "builtin/diff.c", "builtin/merge.c",
                   "diff.c", "merge-recursive.c", "sha1-file.c", "refs.c",
                   "tree.c", "commit.c", "object.c", "packfile.c",
                   "remote.c", "transport.c", "fsck.c", "revision.c"],
    "rocksdb":    ["db/db_impl.cc", "db/column_family.cc", "db/compaction.cc",
                   "db/version_set.cc", "table/block_based_table_reader.cc",
                   "table/block_based_table_builder.cc", "table/cuckoo_table_reader.cc",
                   "util/hash.cc", "util/crc32c.cc", "memtable/skiplistrep.cc",
                   "memtable/hashskiplistrep.cc", "include/rocksdb/db.h"],
    "godot":      ["core/object/object.cpp", "core/object/class_db.cpp",
                   "core/io/resource.cpp", "core/string/ustring.cpp",
                   "scene/main/viewport.cpp", "scene/main/node.cpp",
                   "scene/main/scene_tree.cpp", "scene/resources/material.cpp",
                   "scene/3d/physics/rigid_body_3d.cpp",
                   "scene/2d/physics/rigid_body_2d.cpp",
                   "modules/godot_physics_3d/godot_physics_server_3d.cpp",
                   "modules/godot_physics_2d/godot_physics_server_2d.cpp",
                   "platform/linuxbsd/os_linuxbsd.cpp",
                   "drivers/vulkan/rendering_context_driver_vulkan.cpp"],
    "tokio":      ["tokio/src/runtime/mod.rs", "tokio/src/runtime/scheduler.rs",
                   "tokio/src/io/mod.rs", "tokio/src/net/mod.rs",
                   "tokio/src/sync/mod.rs", "tokio/src/time/mod.rs",
                   "tokio/src/fs/mod.rs", "tokio/src/process/mod.rs",
                   "tokio/src/signal/mod.rs", "tokio/src/task/mod.rs",
                   "tokio-util/src/codec/mod.rs",
                   "tokio-stream/src/lib.rs"],
    "rapier":     ["src/dynamics/jacobian.rs", "src/dynamics/joint/mod.rs",
                   "src/dynamics/solver/velocity_ground.rs",
                   "src/dynamics/integrator.rs", "src/geometry/mod.rs",
                   "src/pipeline/physics_pipeline.rs",
                   "src/pipeline/query_pipeline.rs",
                   "src/control/character_controller.rs"],
    "etcd":       ["server/etcdserver/server.go", "server/etcdserver/raft.go",
                   "server/etcdserver/v3_server.go", "server/storage/backend.go",
                   "server/storage/mvcc/kv.go", "server/embed/etcd.go",
                   "client/v3/client.go", "client/v3/kv.go",
                   "raft/raft.go", "raft/node.go", "raft/rawnode.go",
                   "pkg/fileutil/purge.go"],
    "gin":        ["gin.go", "router.go", "context.go", "binding/form.go",
                   "render/json.go", "middleware/logger.go",
                   "middleware/recovery.go"],
    "fastapi":    ["fastapi/main.py", "fastapi/routing.py", "fastapi/dependencies/utils.py",
                   "fastapi/params.py", "fastapi/responses.py",
                   "fastapi/exception_handlers.py", "fastapi/middleware/cors.py",
                   "fastapi/openapi/utils.py"],
    "flask":      ["src/flask/app.py", "src/flask/routing.py", "src/flask/ctx.py",
                   "src/flask/blueprints.py", "src/flask/json/provider.py",
                   "src/flask/sessions.py", "src/flask/templating.py",
                   "src/flask/wrappers.py"],
}

OUT = "omni_corpus_repos_subset.jsonl"

def main():
    count = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for repo_prefix, files in SELECT.items():
            for f in files:
                for candidate in sorted(TEMP.rglob(f.split("/")[-1])):
                    rel = str(candidate.relative_to(TEMP))
                    if rel.startswith(repo_prefix) or f in rel:
                        try:
                            code = candidate.read_text(encoding="utf-8", errors="replace")
                        except:
                            continue
                        if len(code) < 200:
                            continue
                        parts = [code[i:i+2000] for i in range(0, min(len(code), 6000), 2000)]
                        if parts and parts[0].strip():
                            doc = {
                                "source_url": f"https://github.com/{rel.split('/')[0]}/repos",
                                "title": f"{repo_prefix}/{f}",
                                "author": repo_prefix.split("/")[0],
                                "language": repo_prefix.split("_")[0] if "_" in repo_prefix else "c",
                                "chapters": [{"heading": f"Code: {repo_prefix.split('/')[0]}",
                                              "paragraphs": parts[:3]}]
                            }
                            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                            count += 1
                        break
    print(f"Selected {count} files -> {OUT}")

if __name__ == "__main__":
    main()
