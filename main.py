import os, io, base64, tempfile, shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
import detector as D

app = FastAPI(title="Deepfake Authenticity Platform")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

@app.on_event("startup")
def _startup(): D.preload()

@app.get("/api/status")
def status(): return D.status()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def read_meta(path):
    sz = os.path.getsize(path)
    with open(path, "rb") as f:
        b = f.read(1_500_000)
        if sz > 1_500_000:
            f.seek(max(0, sz - 700_000)); b += f.read()
    return D.meta_scan(b)

def level(p):
    c = max(p, 1 - p)
    return "High" if c >= .8 else "Medium" if c >= .65 else "Low"

def pack(kind, p, items, rows, meta, note, steps, frames=None):
    ai = p >= D.THRESH
    return {"kind": kind, "label": "ai_generated" if ai else "authentic", "ai_probability": round(p, 4),
            "confidence": round(max(p, 1 - p), 4), "level": level(p), "criteria": D.criteria(rows, meta, kind),
            "steps": steps + ([note] if note else []), "frames": frames, "engine": "FastAPI + PyTorch + Hugging Face Transformers"}

def thumb(im):
    t = im.copy(); t.thumbnail((120, 120)); b = io.BytesIO(); t.save(b, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

@app.get("/api/health")
def health(): return {"ok": True}

@app.post("/api/analyze/{kind}")
async def analyze(kind: str, file: UploadFile = File(...)):
    if kind not in ("image", "video", "audio"): raise HTTPException(404, "kind must be image | video | audio")
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
        shutil.copyfileobj(file.file, t); path = t.name
    try:
        meta = read_meta(path)
        steps = [f"Input Router: content type = {kind}", f"meta_scan(): AI signature = {meta['ai'] or 'none'}, device tag = {meta['cam'] or 'none'}"]
        if kind == "image":
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            steps.append(f"preprocess(): decoded {im.width}×{im.height}, EXIF orientation fixed, RGB")
            rows = D.run_image_models([im])
            for r in rows: steps.append(f"{r['name']}: " + (f"P(AI) = {r['ps'][0]:.3f}" if "ps" in r else "unavailable"))
            p, items, note = D.fuse(rows, meta, kind)
            steps.append("fuse(): weighted ensemble of neural models + metadata prior")
            return pack(kind, p, items, rows, meta, note, steps)
        if kind == "video":
            import cv2
            cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
            if n <= 0: raise HTTPException(400, "Cannot read video frames")
            idx = np.linspace(n * .05, n * .95, 12).astype(int); frames = []
            for i in idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
                if ok: frames.append((Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), i / fps))
            cap.release()
            if not frames: raise HTTPException(400, "No frames could be decoded")
            steps.append(f"sample_frames(): {len(frames)} frames extracted with OpenCV ({frames[0][0].width}×{frames[0][0].height})")
            rows = D.run_image_models([f for f, _ in frames])
            for r in rows: steps.append(f"{r['name']}: mean P(AI) over frames = " + (f"{np.mean(r['ps']):.3f}" if "ps" in r else "unavailable"))
            p, items, note = D.fuse(rows, meta, kind)
            steps.append("fuse(): per-frame ensemble → mean over frames + metadata prior")
            fr_out = [{"at": round(t, 1), "p": round(float(items[k]), 3), "t": thumb(f)} for k, (f, t) in enumerate(frames)]
            return pack(kind, p, items, rows, meta, note, steps, fr_out)
        import librosa
        y, _ = librosa.load(path, sr=16000, mono=True, duration=60)
        if len(y) < 16000: raise HTTPException(400, "Audio shorter than 1 second")
        win = 6 * 16000
        starts = [0] if len(y) <= win else list(np.linspace(0, len(y) - win, min(5, len(y) // win + 1)).astype(int))
        segs = [y[s:s + win] for s in starts]
        steps.append(f"preprocess(): resampled to 16 kHz mono, {len(y)/16000:.1f}s, {len(segs)} analysis window(s)")
        rows = D.run_audio_models(segs)
        for r in rows: steps.append(f"{r['name']}: mean P(AI) = " + (f"{np.mean(r['ps']):.3f}" if "ps" in r else "unavailable"))
        p, items, note = D.fuse(rows, meta, kind)
        steps.append("fuse(): weighted ensemble over windows")
        return pack(kind, p, items, rows, meta, note, steps)
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f"{type(e).__name__}: {e}")
    finally:
        try: os.remove(path)
        except OSError: pass

@app.get("/")
def home(): return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
