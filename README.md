# Deep Image Colorization Using U-Net in CIE Lab Color Space

The objective of this project is to build a deep-learning image colorization system where a grayscale image represented by the CIE Lab L* channel is provided as input to a custom U-Net, and the network predicts the a* and b* color channels. The predicted channels will later be combined with L* and converted back to RGB.

## Current Development Status
**Step 1 — Environment Setup**

## Basic Project Structure
```text
deep-image-colorization/
├── data/           # Datasets (raw, train, val, test)
├── models/         # Saved model checkpoints
├── outputs/        # Predictions and plots
├── src/            # Source code (dataset, model, training, inference)
├── scripts/        # Utility scripts (like setup verification)
├── requirements.txt# Project dependencies
└── README.md       # Project documentation
```

## Basic Environment Setup Instructions (Windows / Linux)
1. Install Python 3.8 or higher.
2. Create and activate a virtual environment (optional but recommended).
3. Install PyTorch with CUDA support (e.g., for CUDA 12.6):
   ```bash
   pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
   ```
4. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## macOS Setup
1. **Clone repository**:
   ```bash
   git clone https://github.com/codeby-arsb/nndl-mini-project.git
   cd nndl-mini-project
   ```
2. **Create .venv**:
   ```bash
   python3 -m venv .venv
   ```
3. **Activate .venv**:
   ```bash
   source .venv/bin/activate
   ```
4. **Install dependencies**:
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -r requirements.txt
   ```
5. **Download ADE20K if not present**:
   ```bash
   python scripts/download_dataset.py
   ```
6. **Run verification**:
   ```bash
   python scripts/verify_setup.py
   ```

### Cross-Platform Compute Backend
The codebase dynamically detects and configures the optimal compute backend across operating systems:
* **NVIDIA CUDA**: Used on compatible Windows/Linux systems for hardware-accelerated training and CUDA-based Automatic Mixed Precision (AMP).
* **Apple Silicon**: Metal Performance Shaders (`mps`) backend is used when available on Apple Silicon Macs (M1/M2/M3/M4/M5), supporting hardware acceleration and MPS autocast/GradScaler.
* **CPU**: Fully supported fallback across all platforms.

## Hardware Configuration
This project was developed and tested on:
* **Windows**: NVIDIA GeForce RTX 3050 Laptop GPU, PyTorch with CUDA 12.6
* **macOS**: Apple Silicon (M5), PyTorch with MPS backend

## Verify Setup
To verify that your environment is set up correctly, run the verification script:
```bash
python scripts/verify_setup.py
```

## Dataset
We use the **ADE20K** dataset for this project. ADE20K provides a highly diverse collection of indoor, outdoor, natural, urban, and object scenes, which makes it an excellent general-purpose dataset for image colorization. 

* **Usage**: The images are strictly used as RGB source images. Semantic segmentation masks and object annotations provided by ADE20K are ignored.
* **Storage Strategy**: The raw dataset images are stored locally in `data/raw/` but are excluded from version control (Git). Instead, deterministic split manifests (`data/splits/*.txt`) containing relative paths are tracked in Git to ensure dataset reproducibility without inflating the repository size.
* **Initial Experimental Subset**:
  * Training: 10,000 images
  * Validation: 1,000 images
  * Test: 500 images
* **Random Seed**: 42

### Generating Splits and Inspecting
To automatically download the raw dataset (if not already present), run:
```bash
python scripts/download_dataset.py
```

Once the dataset is downloaded, generate the deterministic train/val/test splits:
```bash
python scripts/create_splits.py
```

To inspect the dataset statistics (resolutions, formats, splits count):
```bash
python scripts/inspect_dataset.py
```

## CIE Lab Preprocessing
This project converts RGB images into the CIE Lab color space to train the colorization network. 
* **Why Lab?** The CIE Lab color space separates lightness (L*) from color (a* and b*), allowing the U-Net to be trained solely on predicting color from a grayscale-like input without having to simultaneously predict brightness.
* **Inputs & Targets**: The L* channel is extracted, normalized to `[-1, 1]`, and fed as the single-channel input to the model. The a* and b* channels are normalized to `[-1, 1]` and serve as the two-channel prediction targets.
* **Resizing**: All images are uniformly resized to 256x256 using bilinear interpolation before conversion.
* **Reconstruction**: After predictions are made, the predicted a* and b* channels are combined with the original L* channel, denormalized, and converted back to standard RGB.

To verify the preprocessing pipeline, check numerical boundaries, and inspect the reconstruction accuracy, run:
```bash
python scripts/test_preprocessing.py
```

### Dataset Pipeline
The data loading pipeline is implemented using standard PyTorch `Dataset` and `DataLoader` classes. 
* **Flow**: Manifest $\rightarrow$ RGB image $\rightarrow$ 256x256 resize $\rightarrow$ RGB to Lab conversion $\rightarrow$ L/ab normalization $\rightarrow$ PyTorch Tensors $\rightarrow$ DataLoader batch.
* **Batch Size**: 8 (optimized for 4GB VRAM environments).
* **Device**: The DataLoader returns CPU tensors. Transfer to GPU (if available) occurs explicitly during the training loop.
* **Concurrency**: `num_workers=0` initially to ensure stability on Windows.
* **Memory**: `pin_memory=True` if CUDA is available for faster host-to-device transfers.

## Architecture
The colorization model is a custom PyTorch U-Net architecture designed specifically for this project.
* **Input**: `1 × 256 × 256` (L* channel)
* **Encoder**: 6 stages of downsampling (`kernel_size=4`, `stride=2`, `padding=1`, `BatchNorm2d`, `LeakyReLU(0.2)`)
  * `1 → 64` (128x128)
  * `64 → 128` (64x64)
  * `128 → 256` (32x32)
  * `256 → 512` (16x16)
  * `512 → 512` (8x8)
  * `512 → 512` (4x4)
* **Bottleneck**: `512` channels at `4 × 4` spatial resolution.
* **Decoder**: Symmetric upsampling with skip connections (`ConvTranspose2d`, followed by concatenation, `BatchNorm2d`, `ReLU`).
  * Decoder channels halve sequentially after concatenating skip connections: `1024 → 1024 → 512 → 256 → 128`.
* **Output**: `2 × 256 × 256` (a*b* channels)
* **Output Activation**: `Tanh` (to constrain outputs approximately to `[-1, 1]`)

## Training Pipeline
The training pipeline uses PyTorch to optimize the U-Net on the ADE20K subset.
* **Loss Function**: `MSELoss()` computed directly on the normalized a* and b* tensors.
* **Optimizer**: `Adam` with an initial learning rate of `2e-4`.
* **Scheduler**: `StepLR`.
* **Batch Size**: 8 (optimized for 4GB VRAM).
* **AMP**: Automatic Mixed Precision is used when CUDA is available to reduce memory usage and accelerate training.
* **Validation**: Model is evaluated on the validation set without gradients (`model.eval()`).
* **Checkpointing**: The pipeline maintains `outputs/checkpoints/latest.pth` and `outputs/checkpoints/best.pth`.
* **Resume Support**: Training can be resumed seamlessly by providing a checkpoint path.
* **History**: Training history (loss, lr, time) is recorded in `outputs/training_history.csv`.

To run a quick one-epoch smoke test to verify the entire pipeline (including AMP and checkpointing) without training on the full dataset:
```bash
python -m src.train --smoke-test
```

## Baseline Analysis
To run qualitative and quantitative evaluation on the 20-epoch baseline model (`best.pth`):
```bash
python scripts/analyze_baseline.py
```
This generates high-resolution comparison sheets (`outputs/plots/baseline_best_analysis.png`), per-image metrics (`outputs/evaluation/per_image_metrics.csv`), and temporal training progression visualizations (`outputs/plots/training_progression_analysis.png`).

## Loss Function Experiments

The project supports a configurable loss function system designed to compare regression behaviors in colorization:

### 1. Baseline: MSE Loss (Completed)
* **Formulation**: $L_{\text{MSE}} = \frac{1}{N}\sum (y_{\text{pred}} - y_{\text{true}})^2$
* **Characteristics**: Penalizes larger errors quadratically ($2e$), causing gradient dominance from large errors and heavily favoring average, muted predictions (desaturation / sepia tint) to minimize expected variance.
* **Status**: Full 20-epoch baseline training completed (`outputs/checkpoints/best.pth`, epoch 13, val loss 0.009781).

### 2. Experiment 2: Smooth L1 / Huber Loss (Ready for Training)
* **Formulation**:
  $$\mathcal{L}_{\beta}(e) = \begin{cases} 0.5 \frac{e^2}{\beta}, & \text{if } |e| < \beta \\ |e| - 0.5\beta, & \text{otherwise} \end{cases}$$
  (with default transition threshold $\beta = 1.0$)
* **Characteristics**: Transitions smoothly from quadratic behavior for fine adjustments ($|e| < \beta$) to linear behavior for larger deviations ($|e| \ge \beta$). The gradient saturates at $\pm 1.0$, preventing extreme outlier penalties and encouraging the network to predict vibrant, highly chromatic colors without collapsing to gray.
* **Status**: Implementation verified across analytical regimes, gradient saturation, AMP/MPS compatibility, and controlled learning diagnostics. Full 20-epoch training ready to run.

### Running Loss Verification Tests
To verify loss function mathematics, gradient saturation, and device compatibility:
```bash
python scripts/test_losses.py
```
To verify the training loop integration and controlled single-batch learning diagnostic on MPS:
```bash
python scripts/test_loss_training.py
```

### Running Training by Loss Configuration
* **Run Baseline (MSE)**:
  ```bash
  python -m src.train --amp
  ```
  *(Checkpoints saved to `outputs/checkpoints/`)*

* **Run Experiment 2 (Smooth L1)**:
  ```bash
  python -m src.train --loss smooth_l1 --amp
  ```
  *(Checkpoints saved to `outputs/experiments/smooth_l1/checkpoints/` to ensure baseline protection)*

## Self-Correcting Colorization

This stage is an **iterative self-correction / refinement framework around the existing U-Net**. It does **not** retrain the U-Net and does **not** change U-Net weights. Feedback is applied in CIE Lab space to the predicted color image.

### Original pipeline

```text
B&W
 ↓
U-Net
 ↓
Colorized Image
```

### New pipeline

```text
B&W
 ↓
U-Net  (once, frozen weights)
 ↓
AI Evaluation
 ↓
Feedback
 ↓
Lab Refinement
 ↓
Re-evaluation
 ↓
Repeat (until score ≥ threshold or max iterations)
 ↓
Best Result
```

The U-Net is run **once**. Later iterations refine the Lab prediction from the previous iteration. Returning the last iteration is not automatic: the highest-scoring refined iteration is kept; if no refinement is needed, the baseline is returned.

### Evaluation metrics (0–100)

| Metric | What it measures |
| --- | --- |
| Semantic | Per-class Lab statistics vs. reference stats, using DeepLabV3 masks |
| Color realism | KL divergence of the a*/b* histogram vs. a reference histogram |
| Boundary consistency | Color change across luminance / semantic edges |
| Skin-tone consistency | Person-region Lab vs. a typical skin palette (`None` if no person) |
| Saturation | Mean chroma; both under- and over-saturation are penalized |
| Overall | Weighted combination of the available metrics |

Default weights: semantic 0.25, realism 0.20, boundary 0.20, skin 0.15, saturation 0.20. If skin is not applicable, the remaining weights are renormalized.

### Segmentation limitation (important)

Semantic and skin metrics use `torchvision.models.segmentation.deeplabv3_resnet101` with **`DeepLabV3_ResNet101_Weights.DEFAULT`**. Those weights are Pascal VOC 21-class (`COCO_WITH_VOC_LABELS_V1`). Runtime class IDs are taken from `weights.meta["categories"]`.

Available classes include `person` (VOC id 15) and objects such as `car` / `dog`, **not** ADE20K-style `sky`, `vegetation`, `water`, `road`, or `building`. The architecture can consume additional classes if they appear in both the weight metadata and `src/self_correct/semantic_ref_stats.json`. Those scene classes are **not fabricated**. With the default stats file, semantic evaluation is effectively **person-only**.

If DeepLab weights cannot be downloaded, semantic and skin scores are skipped (`skin=None`, semantic=0) and the remaining metrics still run.

### Feedback and Lab refinement

`FeedbackGenerator` converts low metric scores into bounded Lab adjustments:

* global a*/b* scale for saturation
* per-class (a*, b*) shifts for valid semantic masks
* skin (a*, b*) shift on the person mask only
* edge-aware `cv2.bilateralFilter` strength for boundary issues (a*/b* only; L* unchanged)

`refine()` copies the input, rejects NaN/inf, clips Lab to L* ∈ [0, 100] and a*/b* ∈ [-128, 128] (same convention as `src.utils.rgb_to_lab` / `lab_to_rgb`), and never mutates the original array.

### Loop control and baseline comparison

* Default `threshold = 85`, `max_iterations = 3`
* Stop early when overall score ≥ threshold
* Track `best_image`, `best_score`, `best_iteration` (iteration `0` = baseline)
* Report baseline score, best self-corrected score, absolute improvement, and percentage improvement
* Improvement **may be negative** if refinement does not help; that result is kept honestly

### CLI

```bash
python -m src.self_correct.cli --input <image> --threshold 85 --max-iterations 3 --device auto
```

Optional: `--checkpoint path/to/best.pth`, `--output-dir ...`, `--no-segmentation` (skips DeepLab download).

Prints:

```text
Self-Correcting Colorization

Baseline Score: XX.XX

Iteration 1: XX.XX
Issues:
  - ...

Best Iteration: X
Final Score: XX.XX
Improvement: XX.XX
```

Writes `outputs/self_correction/<image_name>/`:

* `baseline.png`, `iteration_*.png` (only iterations that ran), `final.png`
* `evaluation.json`, `feedback.json`
* `score_progression.png`, `comparison.png` (original B&W | baseline U-Net | final)

### Tests

```bash
pytest -q tests/self_correct/
pytest -q
```

Heavy pretrained segmentation weights are not required for the unit tests; the evaluator is constructed with `load_segmentation=False` and DeepLab is mocked by that flag.

