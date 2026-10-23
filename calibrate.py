"""Tune the ensemble on YOUR files so accuracy is measured and maximised for your demo data.
   Put files in:  test/image/real  test/image/ai   test/video/real  test/video/ai   test/audio/real  test/audio/ai
   (>= 15 per folder recommended; more = better).  Run:  python calibrate.py image   (or video / audio)
   It prints honest cross-validated accuracy and saves fusion.json, which the server uses automatically."""
import sys, glob, json, os, math, numpy as np
from PIL import Image, ImageOps
import detector as D
kind = sys.argv[1] if len(sys.argv) > 1 else "image"
EXT = {"image": ("jpg", "jpeg", "png", "webp", "bmp"), "video": ("mp4", "mov", "webm", "avi", "mkv"), "audio": ("wav", "mp3", "m4a", "flac", "ogg")}[kind]
def files(lbl):
    ds = [f"test/{kind}/{lbl}"] + ([f"test/{lbl}"] if kind == "image" else [])
    return [p for d in ds for p in sorted(glob.glob(d + "/*")) if p.lower().rsplit(".", 1)[-1] in EXT]
def feats(path):
    if kind == "image": rows = D.run_image_models([ImageOps.exif_transpose(Image.open(path)).convert("RGB")])
    elif kind == "video": rows = D.run_image_models([f for f, _ in D.sample_frames(path, 8)])
    else: rows = D.run_audio_models(D.audio_segments(path))
    return {r["name"]: float(np.mean(r["ps"])) for r in rows if "ps" in r}
def fit(X, y, l2=0.05, it=3000, lr=0.2):
    w = np.ones(X.shape[1]) / X.shape[1]; b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ w + b))); g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w); b -= lr * g.mean()
    return w, b
if __name__ == "__main__":
    R, A = files("real"), files("ai"); print(f"{kind}: {len(R)} real, {len(A)} AI files")
    if min(len(R), len(A)) < 5: sys.exit("Need at least 5 files in each folder (15+ recommended).")
    F, y = [], []
    for lbl, L in ((0, R), (1, A)):
        for i, p in enumerate(L):
            try: F.append(feats(p)); y.append(lbl); print(f"  [{i+1}/{len(L)}] {'AI  ' if lbl else 'real'} {os.path.basename(p)}", flush=True)
            except Exception as e: print("  skipped", p, e)
    names = sorted(set.intersection(*[set(f) for f in F])); y = np.array(y)
    X = np.array([[D._lg(f[n]) for n in names] for f in F]); P = 1 / (1 + np.exp(-X))
    print("\\nPer-model accuracy at 0.5:")
    for j, n in enumerate(names): print(f"  {((P[:, j] >= .5) == y).mean():6.1%}  {n}")
    wts = {n: w for n, w in dict(D.IMG_MODELS + D.AUD_MODELS).items()}
    base = sum(wts.get(n, 1) * P[:, j] for j, n in enumerate(names)) / sum(wts.get(n, 1) for n in names)
    print(f"  {((base >= .5) == y).mean():6.1%}  default weighted ensemble")
    hit = 0
    for i in range(len(y)):
        m = np.arange(len(y)) != i; w, b = fit(X[m], y[m]); hit += ((X[i] @ w + b >= 0) == y[i])
    print(f"\\nCalibrated ensemble, leave-one-out accuracy (honest estimate): {hit/len(y):.1%}")
    w, b = fit(X, y); cur = json.load(open(D.CAL_PATH)) if os.path.exists(D.CAL_PATH) else {}
    cur[kind] = {"models": names, "w": [float(v) for v in w], "b": float(b)}; json.dump(cur, open(D.CAL_PATH, "w"), indent=1)
    print(f"Saved {D.CAL_PATH}. Restart the server to use it.")
