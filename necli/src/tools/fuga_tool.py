"""Fuga knowledge tools.

Fuga — это движок гиперразмерных вычислений (VSA WaveCube + MemoryStore +
TM/HTM + H-JEPA), обученный на едином корпусе (код 5 языков, физика, тексты,
исходники самой Fuga). necli-агент использует его как базу знаний:

- fuga_query — задать вопрос базе знаний Fuga: возвращает top-совпадения из
  обученной памяти (source + сниппет). Быстрый путь — HTTP /api/retrieve
  на запущенном fuga-web; фолбэк — CLI `fuga ask --cube`.

- fuga_learn — инъекция нового знания: факт дописывается в корпус Fuga
  (fuga_injections.jsonl) в формате CorpusDoc и на следующем прогоне
  train-stack сворачивается в единый стек. Так Fuga обучается через
  инъекции от агента.

Конфигурация (env):
  FUGA_API_URL    — base URL запущенного fuga-web (default http://localhost:8080)
  FUGA_BIN        — путь к бинарю fuga (default 'fuga' в PATH)
  FUGA_CUBE_PATH  — куб для CLI-фолбэка (default fuga_stack.bin)
  FUGA_DIR        — каталог Fuga для инъекций (default: рядом с CU_BE_PATH или $PWD)
  FUGA_LEARN_FILE — файл корпуса инъекций (default <FUGA_DIR>/fuga_injections.jsonl)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess

from loguru import logger

from tools.models import ToolCall, ToolResult

_DEFAULT_API = "http://localhost:8080"
_DEFAULT_CUBE = "fuga_stack.bin"
_LEARN_FILE = "fuga_injections.jsonl"
_QUERY_TIMEOUT = 15
_MAX_SNIPPET = 1200
_MAX_RESULTS = 25


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _fuga_dir() -> str:
    env = os.environ.get("FUGA_DIR", "").strip()
    if env:
        return env
    cube = os.environ.get("FUGA_CUBE_PATH", _DEFAULT_CUBE).strip()
    if os.path.sep in cube and os.path.isabs(cube):
        return os.path.dirname(cube)
    return os.getcwd()


def _query_http(query: str, top_k: int) -> list[dict] | None:
    """Быстрый запрос к запущенному fuga-web (/api/retrieve). None если недоступен."""
    base = os.environ.get("FUGA_API_URL", _DEFAULT_API).strip().rstrip("/")
    url = f"{base}/api/retrieve"
    try:
        import httpx

        resp = httpx.post(
            url,
            json={"query": query, "top_k": top_k},
            timeout=_QUERY_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("fuga /api/retrieve status {}", resp.status_code)
            return None
        data = resp.json()
        return data.get("results") or []
    except Exception as e:
        logger.warning("fuga HTTP query unavailable: {}", e)
        return None


def _query_cli(query: str, top_k: int) -> list[dict]:
    """Фолбэк: CLI `fuga ask --cube <path>`. Возвращает список совпадений."""
    bin_path = os.environ.get("FUGA_BIN", "fuga").strip()
    if not shutil.which(bin_path) and not os.path.isfile(bin_path):
        return [{"score": "", "source": "fuga-cli", "text": f"fuga binary not found: {bin_path}"}]
    cube = os.environ.get("FUGA_CUBE_PATH", _DEFAULT_CUBE).strip()
    cmd = [bin_path, "ask", query, "--cube", cube]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=_QUERY_TIMEOUT * 3)
    except Exception as e:
        return [{"score": "", "source": "fuga-cli", "text": f"fuga ask failed: {e}"}]
    text = (out.stdout or "") + ("\n" + out.stderr if out.stderr else "")
    return [{"score": "", "source": "fuga-cli", "text": text[:_MAX_SNIPPET * 4]}]


def execute_fuga_query(call: ToolCall) -> ToolResult:
    """args: {question, top_k?} — спросить базу знаний Fuga."""
    args = call.args or {}
    question = str(args.get("question", "")).strip()
    if not question:
        return ToolResult(
            name="fuga_query", status="error",
            output="fuga_query requires non-empty 'question'.",
            exit_code=1, command=call.command,
        )
    try:
        top_k = int(args.get("top_k", 8))
    except (TypeError, ValueError):
        top_k = 8
    top_k = max(1, min(top_k, _MAX_RESULTS))

    results = _query_http(question, top_k)
    if results is None:
        results = _query_cli(question, top_k)
    if not results:
        return ToolResult(
            name="fuga_query", status="ok",
            output="No matches in Fuga knowledge base.",
            exit_code=0, command=call.command,
        )

    lines = [f"Fuga knowledge matches for: {question}", ""]
    for r in results:
        src = r.get("source", "?")
        score = r.get("score", "")
        txt = (r.get("text") or "").strip().replace("\n", " ")
        if len(txt) > _MAX_SNIPPET:
            txt = txt[:_MAX_SNIPPET] + "…"
        tag = f" (sim={score})" if score else ""
        lines.append(f"[{src}]{tag}\n  {txt}\n")
    return ToolResult(
        name="fuga_query", status="ok",
        output="\n".join(lines),
        exit_code=0, command=call.command,
    )


def execute_fuga_learn(call: ToolCall) -> ToolResult:
    """args: {text, title?} — инъекция знания в корпус Fuga (CorpusDoc).

    Факт дописывается в fuga_injections.jsonl; на следующем train-stack он
    сворачивается в единый стек. title — короткое имя факта (default по времени).
    """
    args = call.args or {}
    text = str(args.get("text", "")).strip()
    if not text:
        return ToolResult(
            name="fuga_learn", status="error",
            output="fuga_learn requires non-empty 'text'.",
            exit_code=1, command=call.command,
        )
    title = str(args.get("title", "")).strip() or f"agent-injection {_now()}"
    lang = str(args.get("language", "en")).strip() or "en"

    learn_path = os.path.join(_fuga_dir(), _LEARN_FILE)
    doc = {
        "source_url": "necli://agent-injection",
        "title": title,
        "author": "necli",
        "language": lang,
        "chapters": [{"heading": f"Agent injection: {title}", "paragraphs": [text]}],
    }
    try:
        with open(learn_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    except Exception as e:
        return ToolResult(
            name="fuga_learn", status="error",
            output=f"fuga_learn failed to write {learn_path}: {type(e).__name__}: {e}",
            exit_code=1, command=call.command,
        )
    return ToolResult(
        name="fuga_learn", status="ok",
        output=(
            f"Injected into Fuga corpus: {learn_path}\n"
            f"  title: {title}\n"
            f"  text:  {text[:200]}{'…' if len(text) > 200 else ''}\n"
            f"Fold into the unified stack with: fuga train-stack {learn_path}"
        ),
        exit_code=0, command=call.command,
    )
