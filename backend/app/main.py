from __future__ import annotations

import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .jobs import create_job, get_job, update_job
from .pipeline.run import SUPPORTED_EXTS, run_pipeline
from .pipeline.skill_loader import load_skill

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
FRONTEND_DIR = BASE_DIR.parents[0] / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = 300 * 1024 * 1024  # 300 MB

app = FastAPI(title="AI Squat Analysis")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/skill")
def get_skill():
    return load_skill()


@app.get("/api/interface-requirements")
def interface_requirements():
    return {
        "supported_formats": sorted(SUPPORTED_EXTS),
        "max_duration_sec": 90,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "recording_requirements": [
            "Side-view (profile) of the lifter, camera roughly perpendicular to the direction they face",
            "Full body in frame for the entire rep, including feet and head",
            "Barbell visible across the lifter's back",
            "Stable camera (not handheld-shaky) for reliable landmark tracking",
        ],
    }


def _process_job(job_id: str, input_path: str, output_path: str):
    def progress_cb(pct, message):
        update_job(job_id, progress=pct, message=message)

    try:
        update_job(job_id, status="processing", progress=1, message="Starting")
        result = run_pipeline(input_path, output_path, progress_cb=progress_cb)
        update_job(
            job_id, status="done", progress=100, message="Done",
            result={
                "rep_count": result.rep_count,
                "side_used": result.side_used,
                "fps": result.fps,
                "width": result.width,
                "height": result.height,
                "findings": result.findings,
                "overall_summary": result.overall_summary,
                "confidence_caveats": result.confidence_caveats,
                "writeup_method": result.writeup_method,
                "not_assessable": result.not_assessable,
                "warnings": result.warnings,
                "video_url": f"/api/jobs/{job_id}/video",
            },
        )
    except Exception as e:
        traceback.print_exc()
        update_job(job_id, status="error", message=str(e), error=str(e))


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTS)}")

    job = create_job()
    input_path = UPLOAD_DIR / f"{job.id}{ext}"
    size = 0
    with open(input_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                input_path.unlink(missing_ok=True)
                raise HTTPException(400, f"File too large (> {MAX_UPLOAD_BYTES // (1024*1024)} MB).")
            out.write(chunk)

    output_path = RESULTS_DIR / f"{job.id}.mp4"
    update_job(job.id, status="queued", message="Upload received, queued for processing")

    thread = threading.Thread(
        target=_process_job, args=(job.id, str(input_path), str(output_path)), daemon=True,
    )
    thread.start()

    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
        "result": job.result,
    }


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    job = get_job(job_id)
    if not job or job.status != "done":
        raise HTTPException(404, "Video not ready")
    path = RESULTS_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Video file missing")
    return FileResponse(path, media_type="video/mp4")


# Serve the frontend SPA last so /api/* routes above take priority.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
