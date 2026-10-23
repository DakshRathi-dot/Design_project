"""Self-test: python check.py   -> tells you exactly what is wrong (packages, internet, models)."""
import sys, importlib
print("Python", sys.version.split()[0])
bad = False
for p in ["fastapi", "uvicorn", "multipart", "torch", "transformers", "PIL", "timm", "peft", "numpy", "cv2", "librosa", "soundfile"]:
    try: m = importlib.import_module(p); print(f"  OK   {p} {getattr(m,'__version__','')}")
    except Exception as e: bad = True; print(f"  FAIL {p}: {e}")
if bad: sys.exit("\nFix: pip install -r requirements.txt  (use Python 3.10, 3.11 or 3.12)")
import detector as D
for n, _ in D.IMG_MODELS + D.AUD_MODELS:
    D._settle(n, (n, _) in D.AUD_MODELS)
    s = D.STATUS[n]; print(f"  {'OK  ' if s['state']=='ready' else 'FAIL'} {n} {s.get('err','')}")
print("\nIf models FAIL: check internet / firewall / VPN once, then run again. If all OK, run run.bat")
