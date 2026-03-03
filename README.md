Project Normandy (Alpha)
A Python-based orchestrator for cross-stitch pattern generation, featuring automated image segmentation and color quantization.

Core Features
Multi-Agent AI Audit: A dual-stage pipeline featuring DataCritic and VisionAnalyst (running on Meta-Llama-3.1-8B).
DataCritic: Performs numerical anomaly detection, flagging thread overestimation or underestimation based on usage ratios.
VisionAnalyst: Processes image metrics (edge density, foreground coverage, brightness) to provide a final verdict and a human-readable pattern summary.
Intelligent Thread Optimization: Dynamic Scale Factor adjustment and automatic removal of "low-value" colors (<1m) to optimize the shopping list.

Visual Metrics Extraction: Custom PIL-based engine to calculate:
Edge Density: Identifying geometric vs. painterly patterns.

Foreground Coverage: Precise calculation of active stitching areas.

Color Complexity: Unique RGB packed-pixel analysis.

DMC Hardware Mapping: Automated conversion of image data into real-world DMC thread codes with skein-count estimation.

DMC Integration: Precise color mapping using internal JSON palettes for authentic cross-stitch results.

Color Processing: Maps image colors to the DMC thread palette (see data/ for configs).

Pattern Export: Generates PDF schematics with symbolic grids.

Installation
Clone the repository:
git clone https://github.com/Viktor-Skobeliev/Project_Normandy_XStitch.git
cd Project_Normandy_XStitch

Install dependencies:
pip install -r requirements.txt

Download Models:
The models/ folder is empty by default. You must run the following script to download the required weights (Llama 3.1, Qwen2-VL, MobileSAM, and U2Net):
python download_models.py

Usage
Run the main application:
python main.py
