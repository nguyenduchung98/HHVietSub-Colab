from __future__ import annotations

import os
import json
import random
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from omnivoice.models.omnivoice import OmniVoice, VoiceClonePrompt

TOKEN = os.environ.get("HHVIETSUB_TOKEN", "")
MODEL_ID = os.environ.get("HHVIETSUB_MODEL", "k2-fsa/OmniVoice")
DATA = Path(os.environ.get("HHVIETSUB_DATA", "/content/hhvietsub-data"))
DATA.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="HHVietSub Colab", version="1.0.0")
model = None
inference_lock = threading.Lock()


def authorize(authorization: str | None = Header(default=None)):
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "Token không hợp lệ")


def get_model():
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = OmniVoice.from_pretrained(MODEL_ID, device_map=device, dtype=dtype, load_asr=True)
    return model


@app.get("/health")
def health(_=Depends(authorize)):
    return {"ok": True, "name": "HHVietSub Colab", "runtime": {
        "status": "ready", "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_id": MODEL_ID, "loaded": model is not None}}


@app.post("/warmup")
def warmup(_=Depends(authorize)):
    runtime = get_model()
    return {"ok": True, "device": "cuda" if torch.cuda.is_available() else "cpu",
            "sampling_rate": runtime.sampling_rate}


@app.post("/generate")
def generate(text: str = Form(...), ref_audio: UploadFile = File(...), ref_text: str = Form(""),
                   language: str = Form("vi"), speed: float = Form(1.0), num_step: int = Form(32),
                   guidance_scale: float = Form(2.0), seed: int | None = Form(None),
                   denoise: bool = Form(False), postprocess_output: bool = Form(True),
                   output_format: str = Form("wav"), _=Depends(authorize)):
    if not text.strip(): raise HTTPException(422, "Nội dung trống")
    if output_format != "wav": raise HTTPException(422, "V1 chỉ xuất WAV")
    used_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    suffix = Path(ref_audio.filename or "ref.wav").suffix or ".wav"
    reference = DATA / f"ref-{time.time_ns()}{suffix}"
    reference.write_bytes(ref_audio.file.read())
    output = DATA / f"out-{time.time_ns()}.wav"
    try:
        torch.manual_seed(used_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(used_seed)
        runtime = get_model()
        audio = runtime.generate(text=text.strip(), language=None if language in {"auto", "Auto"} else language,
                                 ref_audio=str(reference), ref_text=ref_text.strip() or None, speed=speed,
                                 num_step=num_step, guidance_scale=guidance_scale, denoise=denoise,
                                 postprocess_output=postprocess_output)[0]
        sf.write(output, audio, runtime.sampling_rate)
        duration = len(audio) / runtime.sampling_rate
        return FileResponse(output, media_type="audio/wav", filename="voice.wav", background=None,
                            headers={"X-Seed": str(used_seed), "X-Audio-Duration": f"{duration:.3f}"})
    finally:
        reference.unlink(missing_ok=True)


@app.post("/generate-batch")
def generate_batch(entries: str = Form(...), voice_prompt: UploadFile = File(...),
                         language: str = Form("vi"), speed: float = Form(1.0),
                         num_step: int = Form(32), guidance_scale: float = Form(2.0),
                         seed: int | None = Form(None), denoise: bool = Form(False),
                         postprocess_output: bool = Form(True), _=Depends(authorize)):
    """Generate an entire SRT job while loading the cloned-voice prompt only once."""
    try:
        items = json.loads(entries)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "Danh sách câu không hợp lệ") from exc
    if not isinstance(items, list) or not items:
        raise HTTPException(422, "Không có câu phụ đề")
    if len(items) > 2000:
        raise HTTPException(422, "Tối đa 2000 câu mỗi batch")

    job_dir = Path(tempfile.mkdtemp(prefix="batch-", dir=DATA))
    prompt_path = job_dir / "voice.pt"
    prompt_path.write_bytes(voice_prompt.file.read())
    archive = job_dir / "voices.zip"
    manifest = []
    try:
        prompt_data = torch.load(prompt_path, map_location="cpu", weights_only=True)
        prompt = VoiceClonePrompt(ref_audio_tokens=prompt_data["ref_audio_tokens"],
                                  ref_text=prompt_data["ref_text"], ref_rms=prompt_data["ref_rms"])
        runtime = get_model()
        with inference_lock:
            for position, item in enumerate(items, 1):
                item_id = int(item.get("id", position))
                text = str(item.get("text", "")).strip()
                if not text:
                    manifest.append({**item, "id": item_id, "status": "failed", "error": "Nội dung trống"})
                    continue
                used_seed = int(seed if seed is not None else time.time_ns() % 2_147_483_647) + item_id
                output = job_dir / f"{item_id:04d}.wav"
                last_error = ""
                for attempt in range(1, 4):
                    try:
                        torch.manual_seed(used_seed + attempt - 1)
                        if torch.cuda.is_available(): torch.cuda.manual_seed_all(used_seed + attempt - 1)
                        audio = runtime.generate(text=text,
                            language=None if language in {"auto", "Auto"} else language,
                            voice_clone_prompt=prompt, speed=speed, num_step=num_step,
                            guidance_scale=guidance_scale, denoise=denoise,
                            postprocess_output=postprocess_output)[0]
                        sf.write(output, audio, runtime.sampling_rate)
                        manifest.append({**item, "id": item_id, "status": "completed",
                                         "duration": round(len(audio) / runtime.sampling_rate, 3),
                                         "attempts": attempt, "seed": used_seed + attempt - 1})
                        break
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                else:
                    manifest.append({**item, "id": item_id, "status": "failed",
                                     "error": last_error, "attempts": 3})
        (job_dir / "manifest.json").write_text(json.dumps({"items": manifest}, ensure_ascii=False), encoding="utf-8")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(job_dir / "manifest.json", "manifest.json")
            for wav in job_dir.glob("*.wav"):
                bundle.write(wav, wav.name)
        return FileResponse(archive, media_type="application/zip", filename="hhvietsub-voices.zip",
                            background=BackgroundTask(shutil.rmtree, job_dir, ignore_errors=True))
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"Batch generation failed: {type(exc).__name__}: {exc}") from exc
