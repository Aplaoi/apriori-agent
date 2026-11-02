"""FastAPI entry point for the Weka Apriori agent demo."""
from __future__ import annotations

import logging
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent import analyze_with_agent
from charts import build_charts
from config import (
    DATA_DIR,
    DEFAULT_BINS,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    MAX_UPLOAD_BYTES,
    OUTPUT_DIR,
    STATIC_DIR,
    UPLOAD_DIR,
    WEKA_JAR,
)
from weka_runner import WekaError, prepare_dataset

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("apriori-agent")

app = FastAPI(title="Weka Apriori Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

ALLOWED_SUFFIXES = {".csv", ".arff"}


def _chart_url(path: Path | str) -> str:
    relative = Path(path).resolve().relative_to(OUTPUT_DIR.resolve())
    return "/outputs/" + relative.as_posix()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "engine": "Weka Apriori",
        "weka_jar": str(WEKA_JAR) if WEKA_JAR else None,
        "weka_ready": WEKA_JAR is not None,
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "deepseek_model": DEEPSEEK_MODEL,
    }


@app.get("/api/demo", include_in_schema=False)
def demo_file() -> FileResponse:
    path = DATA_DIR / "llm.arff"
    if not path.exists():
        raise HTTPException(status_code=404, detail="示例数据不存在")
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.post("/api/analyze")
async def analyze(
    prompt: str = Form(default=""),
    file: UploadFile = File(...),
) -> JSONResponse:
    started_at = time.perf_counter()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="只支持 .csv 或 .arff 文件")

    upload_id = uuid.uuid4().hex
    safe_stem = Path(file.filename or "dataset").stem
    safe_stem = "".join(ch for ch in safe_stem if ch.isalnum() or ch in "-_")[:60] or "dataset"
    upload_path = UPLOAD_DIR / f"{upload_id}_{safe_stem}{suffix}"

    size = 0
    try:
        with upload_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="文件超过 20 MB 限制")
                destination.write(chunk)
    except HTTPException:
        upload_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if size == 0:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传文件为空")

    try:
        prepared = prepare_dataset(upload_path, prompt=prompt)
        prepared["bins"] = DEFAULT_BINS
        prepared["source_filename"] = file.filename
        result = analyze_with_agent(prompt, prepared)
        mining = result["mining"]
        chart_dir = OUTPUT_DIR / prepared["work_dir"].name
        charts = build_charts(mining, chart_dir)
        profile = dict(prepared["profile"])
        profile["filename"] = file.filename
    except WekaError as exc:
        logger.warning("Weka error for %s: %s", file.filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Analysis failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"分析失败：{exc}") from exc
    finally:
        upload_path.unlink(missing_ok=True)

    response = {
        "ok": True,
        "job_id": prepared["work_dir"].name,
        "filename": file.filename,
        "dataset": profile,
        "answer": result["answer"],
        "mode": result["mode"],
        "warning": result["warning"],
        "tool_trace": result["tool_trace"],
        "parameters": mining["parameters"],
        "engine": mining["engine"],
        "total_instances": mining["total_instances"],
        "itemset_total": mining["itemset_total"],
        "rule_total": mining["rule_total"],
        "single_item_count": mining["single_item_count"],
        "reported_min_support": mining["reported_min_support"],
        "reported_min_confidence": mining["reported_min_confidence"],
        "itemsets": mining["itemsets"],
        "rules": mining["rules"],
        "charts": {
            "itemsets": _chart_url(charts["itemsets"]),
            "rules": _chart_url(charts["rules"]),
        },
        "elapsed_seconds": round(time.perf_counter() - started_at, 2),
    }
    return JSONResponse(response)


@app.exception_handler(WekaError)
async def weka_error_handler(_request, exc: WekaError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8180, reload=False)