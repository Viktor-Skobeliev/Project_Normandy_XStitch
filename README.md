# Project Normandy (Alpha)

A Python orchestration engine that turns a photo into a production-ready cross-stitch pattern:
neural segmentation, colour quantization, thread math and a PDF schematic — with a two-agent audit
layer that reviews the result before it ships.

`Python` · `OpenCV / Pillow` · `MobileSAM + U2Net` · `ONNX Runtime (GPU)` · `Llama 3.1 8B (local)` ·
`Qwen2-VL` · `CustomTkinter`

---

## Multi-Agent AI Audit

A dual-stage pipeline that reviews every generated pattern, running on a local Meta-Llama-3.1-8B —
no data leaves the machine.

| Agent | Input | Job |
|-------|-------|-----|
| **DataCritic** | Numerical pattern data | Anomaly detection — flags thread over- or underestimation from usage ratios |
| **VisionAnalyst** | Image metrics | Final verdict plus a human-readable pattern summary |

The pair produces a verdict and a PDF summary, so a bad pattern is caught before it reaches the
user rather than after.

## Visual Metrics Extraction

A custom PIL-based engine computes the metrics the audit runs on:

- **Edge density** — separates geometric patterns from painterly ones
- **Foreground coverage** — precise area actually being stitched
- **Brightness and colour complexity** — packed-pixel RGB analysis

## Thread Optimization

- Dynamic scale-factor adjustment
- Automatic removal of low-value colours (under 1 m) to keep the shopping list practical

## DMC Mapping and Export

- Image colours mapped to the real DMC thread palette using internal JSON palettes (see `data/`)
- Skein-count estimation for a usable shopping list
- PDF schematics with symbolic grids

---

## Installation

```bash
git clone https://github.com/Viktor-Skobeliev/Project_Normandy_XStitch.git
cd Project_Normandy_XStitch
pip install -r requirements.txt
```

The `models/` folder ships empty. Download the weights (Llama 3.1, Qwen2-VL, MobileSAM, U2Net):

```bash
python download_models.py
```

## Usage

```bash
python main.py
```
