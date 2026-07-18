from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path

import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from omnivoice.models.omnivoice import OmniVoice

TOKEN = os.environ.get("HHVIETSUB_TOKEN", "")
MODEL_ID = os.environ.get("HHVIETSUB_MODEL", "k2-fsa/OmniVoice")
DATA = Path(os.environ.get("HHVIETSUB_DATA", "/content/hhvietsub-data"))
DATA.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="HHVietSub Colab", version="1.0.0")
model = None


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


@app.post("/generate")
async def generate(text: str = Form(...), ref_audio: UploadFile = File(...), ref_text: str = Form(""),
                   language: str = Form("vi"), speed: float = Form(1.0), num_step: int = Form(32),
                   guidance_scale: float = Form(2.0), seed: int | None = Form(None),
                   denoise: bool = Form(False), postprocess_output: bool = Form(True),
                   output_format: str = Form("wav"), _=Depends(authorize)):
    if not text.strip(): raise HTTPException(422, "Nội dung trống")
    if output_format != "wav": raise HTTPException(422, "V1 chỉ xuất WAV")
    used_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    suffix = Path(ref_audio.filename or "ref.wav").suffix or ".wav"
    reference = DATA / f"ref-{time.time_ns()}{suffix}"
    reference.write_bytes(await ref_audio.read())
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

