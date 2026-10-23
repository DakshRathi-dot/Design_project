"""Core detection logic: pretrained neural detectors + metadata forensics + fusion."""
import re, io, os, math
import numpy as np

# ---------- 1. METADATA FORENSICS (text strings inside the file) ----------
AI_RE = re.compile(rb"midjourney|dall[-\xb7\s]?e|stable[\s_-]?diffusion|comfyui|automatic1111|firefly|openai|imagen|novelai|sdxl|leonardo\.ai|ideogram|trainedAlgorithmicMedia|Negative prompt|Sampler: |Steps: \d+|synthid|made with google ai|black forest labs|\bsora\b|runway|gen-[234]\b|pika ?labs|\bkling\b|luma ?(ai|dream)|\bveo\b|hailuo|minimax|heygen|synthesia|animatediff|elevenlabs|murf\.ai|wellsaid|\bsuno\b|\budio\b|coqui|resemble\.ai|play\.ht", re.I)
CAM_RE = re.compile(rb"\b(Canon|NIKON|SONY|Apple|iPhone|SAMSUNG|Pixel|Xiaomi|OnePlus|vivo|OPPO|FUJIFILM|Panasonic|OLYMPUS|HUAWEI|DJI|GoPro|realme|Motorola|LEICA|Zoom H\d|Tascam|Zoom Corporation)\b|com\.apple\.quicktime")

def meta_scan(data: bytes) -> dict:
    """Extract printable strings (like `strings`) then look for AI-tool / camera signatures."""
    s = b"\n".join(re.findall(rb"[\x20-\x7e]{6,}", data))
    a, c = AI_RE.search(s), CAM_RE.search(s)
    return {"ai": a.group(0).decode(errors="ignore") if a else None,
            "cam": c.group(0).decode(errors="ignore") if c else None,
            "exif": (b"Exif\x00\x00" in data[:100000]) or (b"<x:xmpmeta" in data[:100000])}

# ---------- 2. LABEL PARSING (works for any HF classifier) ----------
LAB_AI = re.compile(r"\b(ai|fake|artificial|synthetic|generated|deepfake|spoof\w*)\b", re.I)
def p_ai(probs, id2label) -> float:
    idx = [i for i, l in id2label.items() if LAB_AI.search(str(l))]
    if not idx:                       # LABEL_0 / LABEL_1 style: class 1 = fake
        idx = [1] if len(probs) > 1 else [0]
    return float(sum(probs[int(i)] for i in idx))

# ---------- 3. MODEL REGISTRY ----------
BOMBEK = "Bombek1/ai-image-detector-siglip-dinov2"   # SigLIP2 + DINOv2 ensemble, trained on OpenFake (25+ generators); 97% cross-dataset
IMG_MODELS = [(BOMBEK, 0.50),
              ("Smogy/SMOGY-Ai-images-detector", 0.30),     # Swin, modern generators
              ("Organika/sdxl-detector", 0.20)]             # Swin, diffusion images
AUD_MODELS = [("garystafford/wav2vec2-deepfake-voice-detector", 0.40),
              ("MelodyMachine/Deepfake-audio-detection-V2", 0.30),
              ("Gustking/wav2vec2-large-xlsr-deepfake-audio-classification", 0.30)]
THRESH = float(os.getenv("AI_THRESHOLD", "0.5"))
_cache = {}
import threading
STATUS = {}

def _settle(name, audio):
    STATUS[name] = {"state": "loading"}
    print(f"[models] loading {name} ...", flush=True)
    try:
        _load(name, audio); STATUS[name] = {"state": "ready"}
        print(f"[models] ready   {name}", flush=True)
    except Exception as e:
        STATUS[name] = {"state": "failed", "err": f"{type(e).__name__}: {str(e)[:220]}"}
        print(f"[models] FAILED  {name}: {e}", flush=True)

def preload():
    """Load every model in a background thread at server start (first run downloads from Hugging Face)."""
    for n, _ in IMG_MODELS + AUD_MODELS: STATUS[n] = {"state": "pending"}
    def work():
        for n, _ in IMG_MODELS: _settle(n, False)
        for n, _ in AUD_MODELS: _settle(n, True)
        print("[models] all done — open http://localhost:8000", flush=True)
    threading.Thread(target=work, daemon=True).start()

def status():
    out = {}
    for kind, models in (("image", IMG_MODELS), ("audio", AUD_MODELS)):
        st = [STATUS.get(n, {"state": "pending"}) for n, _ in models]
        done = sum(s["state"] in ("ready", "failed") for s in st); ok = sum(s["state"] == "ready" for s in st)
        cur = next((n for (n, _), s in zip(models, st) if s["state"] == "loading"), None)
        out[kind] = {"total": len(st), "done": done, "ok": ok, "loading": cur, "ready": done == len(st) and ok > 0,
                     "failed_all": done == len(st) and ok == 0, "errors": [f"{n.split('/')[-1]}: {s['err']}" for (n, _), s in zip(models, st) if s["state"] == "failed"]}
    return out

def _dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

def _load(name, audio=False):
    if name not in _cache and name == BOMBEK:
        import importlib.util
        from huggingface_hub import hf_hub_download
        for extra in ("config.json",):
            try: hf_hub_download(BOMBEK, extra)
            except Exception: pass
        mp = hf_hub_download(BOMBEK, "pytorch_model.pt"); code = hf_hub_download(BOMBEK, "model.py")
        spec = importlib.util.spec_from_file_location("bombek_model", code); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        _cache[name] = (None, mod.AIImageDetector(mp))
    if name not in _cache:
        from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor, AutoModelForAudioClassification
        if audio:
            _cache[name] = (AutoFeatureExtractor.from_pretrained(name), AutoModelForAudioClassification.from_pretrained(name).eval().to(_dev()))
        else:
            _cache[name] = (AutoImageProcessor.from_pretrained(name), AutoModelForImageClassification.from_pretrained(name).eval().to(_dev()))
    return _cache[name]

def run_image_models(imgs):
    import torch
    rows = []
    for name, w in IMG_MODELS:
        try:
            if STATUS.get(name, {}).get("state") == "failed": raise RuntimeError(STATUS[name]["err"])
            proc, m = _load(name)
            ps = []
            if name == BOMBEK:
                import tempfile
                for im in imgs:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t: im.save(t, "PNG"); tp = t.name
                    try:
                        r = m.predict(tp)
                        ps.append(float(r["probability"]) if isinstance(r, dict) and "probability" in r else float(r))
                    finally: os.remove(tp)
                rows.append({"name": name, "w": w, "ps": ps}); continue
            with torch.no_grad():
                for im in imgs:
                    x = proc(images=im, return_tensors="pt").to(_dev())
                    ps.append(p_ai(m(**x).logits.softmax(-1)[0].tolist(), m.config.id2label))
            rows.append({"name": name, "w": w, "ps": ps})
        except Exception as e:
            rows.append({"name": name, "w": w, "err": str(e)[:160]})
    return rows

def run_audio_models(segs):
    import torch
    rows = []
    for name, w in AUD_MODELS:
        try:
            if STATUS.get(name, {}).get("state") == "failed": raise RuntimeError(STATUS[name]["err"])
            fe, m = _load(name, audio=True)
            ps = []
            with torch.no_grad():
                for s in segs:
                    x = fe(s, sampling_rate=16000, return_tensors="pt", padding=True).to(_dev())
                    ps.append(p_ai(m(**x).logits.softmax(-1)[0].tolist(), m.config.id2label))
            rows.append({"name": name, "w": w, "ps": ps})
        except Exception as e:
            rows.append({"name": name, "w": w, "err": str(e)[:160]})
    return rows

# ---------- 4. FUSION ----------
CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fusion.json")
def _lg(p):
    p = min(max(p, 1e-4), 1 - 1e-4); return math.log(p / (1 - p))
def _cal(kind):
    try:
        import json; return json.load(open(CAL_PATH)).get(kind)
    except Exception: return None

def fuse(rows, meta, kind):
    ok = [r for r in rows if "ps" in r]
    if not ok:
        raise RuntimeError("No model could be loaded. Check internet for first-time download. " + " | ".join(r.get("err", "") for r in rows))
    n = len(ok[0]["ps"]); note = None
    cal = _cal(kind); by = {r["name"]: r for r in ok}
    if cal and all(m in by for m in cal["models"]):
        comb = lambda v: 1 / (1 + math.exp(-(cal["b"] + sum(w * _lg(v[m]) for m, w in zip(cal["models"], cal["w"])))))
        per_item = [comb({m: by[m]["ps"][i] for m in cal["models"]}) for i in range(n)]
        p = comb({m: float(np.mean(by[m]["ps"])) for m in cal["models"]}); calibrated = True
    else:
        tw = sum(r["w"] for r in ok)
        per_item = [sum(r["w"] * r["ps"][i] for r in ok) / tw for i in range(n)]
        p = float(np.mean(per_item)); calibrated = False
    if meta["ai"]:
        p, note = max(p, 0.97), f"AI-tool signature “{meta['ai']}” found in file metadata → forced AI"
    elif meta["cam"] and kind != "audio" and not calibrated:
        p, note = p * 0.55, f"Genuine device metadata ({meta['cam']}) found → AI probability reduced"
    return min(max(p, 0.01), 0.99), per_item, note

# ---------- shared input helpers (used by calibrate.py) ----------
def sample_frames(path, n=12):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(path); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 25; out = []
    for i in np.linspace(total * .05, total * .95, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
        if ok: out.append((Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), i / fps))
    cap.release(); return out

def audio_segments(path):
    import librosa
    y, _ = librosa.load(path, sr=16000, mono=True, duration=60); win = 6 * 16000
    starts = [0] if len(y) <= win else list(np.linspace(0, len(y) - win, min(5, len(y) // win + 1)).astype(int))
    return [y[s:s + win] for s in starts]

def criteria(rows, meta, kind):
    C = []
    C.append({"k": "AI-tool signature", "fn": "meta_scan()", "val": f"Found “{meta['ai']}”" if meta["ai"] else "None found",
              "s": 99 if meta["ai"] else 45, "w": "★",
              "ai": f"File metadata names an AI tool (“{meta['ai']}”).", "re": "No AI-tool signature in the file metadata."})
    C.append({"k": "Device / camera metadata", "fn": "meta_scan()", "val": (meta["cam"] + " tags") if meta["cam"] else ("EXIF/XMP present" if meta["exif"] else "None"),
              "s": 4 if meta["cam"] else 58, "w": "★",
              "ai": f"No camera/device metadata — real {kind}s normally carry it.", "re": f"Genuine device metadata ({meta['cam']}) is embedded — AI generators do not produce this."})
    for r in rows:
        short = r["name"].split("/")[-1]
        if "ps" in r:
            v = float(np.mean(r["ps"]))
            C.append({"k": f"Neural model · {short}", "fn": "transformers.pipeline()", "val": f"P(AI) = {v:.2f}", "s": round(v * 100), "w": f"{int(r['w']*100)}%",
                      "ai": f"{short} rates this {kind} {v:.0%} likely AI-generated (learned generator artefacts).",
                      "re": f"{short} rates this {kind} only {v:.0%} likely AI — it matches real-{kind} patterns."})
        else:
            C.append({"k": f"Neural model · {short}", "fn": "transformers.pipeline()", "val": "failed to load", "s": 50, "w": "0%", "ai": "", "re": ""})
    return C
