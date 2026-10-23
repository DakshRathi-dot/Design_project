# Deepfake Detection & Digital Content Authenticity Platform

## Run (2 minutes)
0. If anything fails, run `python check.py` — it prints exactly which package or model is the problem.
1. Install Python 3.10–3.12 and (for MP3/M4A audio) ffmpeg.
2. Double-click `run.bat` (Windows) or run `./run.sh` (Mac/Linux).
   First start downloads ~1 GB of models from Hugging Face — **do this once with internet, before the presentation**. After that it works offline.
3. Wait until the black window shows `[models] all done`, then open http://localhost:8000 in Chrome. (The page also shows loading progress by itself.) Upload an image, video or audio file.

## What runs
| Type | Models | Method |
|---|---|---|
| Image | Bombek1 SigLIP2 + DINOv2 (50%), SMOGY Swin (30%), Organika SDXL Swin (20%) | neural ensemble |
| Video | same image ensemble on 12 frames (OpenCV) | mean over frames |
| Audio | 3 wav2vec2 deepfake-voice detectors | up to 5 windows of 6 s @ 16 kHz |
| All | metadata forensics | AI-tool signatures force AI |
First start downloads ~4 GB (the SigLIP2 + DINOv2 model is large). Images take a few seconds each on a laptop CPU.

## Get the best accuracy: calibrate on your own files (15 minutes)
Make these folders and put files in them (15+ per folder; more is better; use original files):
`test/image/real`, `test/image/ai`, `test/video/real`, `test/video/ai`, `test/audio/real`, `test/audio/ai`
Then run, for each type:  `python calibrate.py image`   `python calibrate.py video`   `python calibrate.py audio`
It prints each model's accuracy and an honest leave-one-out accuracy, learns the best weights, and saves `fusion.json`. Restart the server and it is used automatically. **Quote the printed leave-one-out number in your report — it is the real, defensible accuracy.**

## Honest limits (mention these to your sir — they match your literature review)
No detector is 100% accurate. Public models lag new generators and accuracy drops on compressed, screenshotted or very small files (your slide cites 83.9% in-data vs 49.9% cross-data). Use original files for the demo. The image models were trained mostly on photos; paintings/illustrations may mislead them.
