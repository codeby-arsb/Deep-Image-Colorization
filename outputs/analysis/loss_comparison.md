# Step 7.2: Comparative Analysis — MSE vs Smooth L1 (Huber) Loss

## 1. Experimental Setup

This investigation conducts a rigorous, controlled comparative evaluation between two full 20-epoch training experiments for deep image colorization using the custom `ColorizationUNet` architecture on the ADE20K dataset:

* **Experiment 1 (Baseline)**: Optimized with Mean Squared Error (MSE) loss on normalized CIE Lab $a^*b^*$ channels:
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum (y_{\text{pred}} - y_{\text{true}})^2$$
* **Experiment 2 (Improved Loss)**: Optimized with Smooth L1 (Huber) loss with transition parameter $\beta = 1.0$:
  $$\mathcal{L}_{\beta}(e) = \begin{cases} 0.5 \frac{e^2}{\beta}, & \text{if } |e| < \beta \\ |e| - 0.5\beta, & \text{otherwise} \end{cases}$$

### Controlled Equivalence Confirmation
The two experiments were executed under strictly identical training protocols:
* **Dataset Splits**: ADE20K deterministic manifests (10,000 training images, 1,000 validation images, 500 test images).
* **Random Seed**: Fixed at 42 with fresh initializations.
* **Architecture**: Unaltered `ColorizationUNet` (Encoder: 6 downsampling stages; Bottleneck: 512 channels @ 4x4; Decoder: symmetric upsampling with skip connections; Output activation: `Tanh`).
* **Optimization**: Adam optimizer ($\text{lr} = 2 \times 10^{-4}$), `StepLR` scheduler ($\text{step\_size} = 10$, $\gamma = 0.5$).
* **Batch Size & Duration**: Batch size 8, trained for exactly 20 epochs.
* **Compute Backend**: Apple Silicon MPS with Automatic Mixed Precision (AMP).
* **Evaluation Protocol**: Fixed deterministic 10-image subset (`outputs/evaluation/fixed_test_images.txt`) evaluated in sRGB color space via denormalized CIE Lab reconstruction.

---

## 2. Quantitative Results

The primary quantitative metrics evaluated on the fixed deterministic test set are summarized below.

| Experiment | Best Val Epoch | Best Val Loss (Loss Scale) | RGB Mean MAE | RGB Mean MSE | RGB Mean PSNR (dB) | Mean Chroma Retention (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (MSE)** | 13 | 0.009781 | **0.040320** | 0.004389 | **24.65 dB** | 68.4% (10.85 / 15.86) |
| **Smooth L1 (Best Val)** | 12 | 0.004872 | 0.040560 | **0.004373** | 24.62 dB | **75.3%** (11.93 / 15.86) |
| *Smooth L1 (Epoch 10)* | 10 | 0.005126 | **0.039501** | 0.004560 | **25.00 dB** | 73.1% (11.59 / 15.86) |
| *Smooth L1 (Epoch 20)* | 20 | 0.005287 | 0.048504 | 0.005916 | 22.97 dB | 78.6% (12.47 / 15.86) |

> **Important Note on Loss Magnitudes**: Validation loss values reflect different mathematical objectives and must not be compared directly. For deviations $|e| \in [0, 1]$, Smooth L1 loss is scaled by $0.5$ relative to raw squared errors ($0.5 \cdot e^2$ vs $e^2$), which accounts for the approximately $2\times$ difference in validation loss magnitude.

### Test Set Progression Across Epochs

| Checkpoint | MSE MAE | MSE MSE | MSE PSNR (dB) | Smooth L1 MAE | Smooth L1 MSE | Smooth L1 PSNR (dB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Epoch 01 | 0.052049 | 0.006899 | 22.58 | 0.051442 | 0.006776 | 22.63 |
| Epoch 05 | 0.043000 | 0.004480 | 24.40 | 0.042524 | 0.004845 | 24.30 |
| Epoch 10 | 0.042139 | 0.004333 | 24.58 | **0.039501** | 0.004560 | **25.00** |
| Epoch 15 | 0.046137 | 0.004762 | 23.91 | 0.045132 | 0.004988 | 23.75 |
| Epoch 20 | 0.045680 | 0.004996 | 23.52 | 0.048504 | 0.005916 | 22.97 |
| **Best** | **0.040320** (Ep 13) | **0.004389** (Ep 13) | **24.65** (Ep 13) | **0.040560** (Ep 12) | **0.004373** (Ep 12) | **24.62** (Ep 12) |

---

## 3. Training Dynamics

Both experiments exhibit remarkably parallel optimization trajectories:
1. **Rapid Initial Convergence (Epochs 1–5)**:
   * MSE training loss dropped from $0.035105 \rightarrow 0.010988$ ($-68.7\%$), validation loss from $0.015398 \rightarrow 0.010704$.
   * Smooth L1 training loss dropped from $0.017557 \rightarrow 0.005484$ ($-68.8\%$), validation loss from $0.007749 \rightarrow 0.005459$.
2. **Plateau & Optimal Generalization Window (Epochs 10–13)**:
   * Learning rate decayed by $50\%$ at Epoch 10 ($0.0002 \rightarrow 0.0001$).
   * MSE reached its global validation loss minimum at **Epoch 13** ($0.009781$).
   * Smooth L1 reached its global validation loss minimum at **Epoch 12** ($0.004872$), with adjacent epochs essentially tied (Epoch 11: $0.004892$; Epoch 13: $0.004877$; Epoch 15: $0.004878$).
3. **Late-Stage Overfitting (Epochs 16–20)**:
   * In both models, training loss continued to decline steadily (MSE: $0.009075 \rightarrow 0.007119$; Smooth L1: $0.004558 \rightarrow 0.003557$).
   * Concurrently, validation loss drifted upward (MSE: $+8.0\%$; Smooth L1: $+8.5\%$) and test PSNR declined significantly (MSE: $-1.13\text{ dB}$; Smooth L1: $-1.65\text{ dB}$ by Epoch 20).
   * This confirms genuine mild overfitting across both loss functions when training beyond Epoch 13–15 on this subset.

---

## 4. Epoch 10 vs Epoch 12 Discrepancy Analysis (Smooth L1)

A notable observation in Smooth L1 training is that **Epoch 10 achieved the highest PSNR (25.00 dB)**, whereas the checkpoint-selection rule (based on validation loss) chose **Epoch 12 (PSNR = 24.62 dB, Val Loss = 0.004872)**.

### Mathematical Explanation:
1. **Objective vs Evaluation Space**:
   * Validation loss is computed across **1,000 images in normalized CIE Lab space** ($ab \in [-1, 1]$) under the Smooth L1 objective.
   * Evaluation metrics are computed on **10 fixed images in sRGB space** ($[0, 1]$).
2. **Convexity of PSNR vs Pixel MSE**:
   * PSNR is defined as $\text{PSNR} = -10 \log_{10}(\text{MSE})$. Because the logarithm is concave (and negative log is convex), improvements on high-error images yield minimal logarithmic gains, whereas minor degradations on near-zero error images produce severe logarithmic penalties.
   * On 6 of the 10 images (`028`, `106`, `150`, `151`, `203`, `324`), Epoch 12 achieved **lower MSE** than Epoch 10, bringing the overall mean test MSE down from $0.004560 \rightarrow 0.004373$ (an aggregate improvement of $4.1\%$).
   * However, on two near-neutral images (`026` and `378`), Epoch 10 achieved near-zero error ($\text{PSNR} > 29.4\text{ dB}$), which disproportionately elevated the arithmetic average PSNR at Epoch 10. When Epoch 12 slightly saturated these regions, their PSNRs dropped by $\sim 2.8\text{ dB}$, skewing the arithmetic mean downward despite the lower aggregate MSE.
3. **Conclusion on Discrepancy**:
   * Epoch 12 is genuinely the better model under aggregate squared error and validation loss.
   * The discrepancy is an artifact of the small evaluation sample ($N=10$) coupled with the nonlinear sensitivity of arithmetic mean PSNR.

---

## 5. Per-Image Results

Per-image metrics on the 10 fixed evaluation images comparing the best checkpoints:

| Image ID | Category / Description | MSE MAE | SL1 MAE | MSE PSNR | SL1 PSNR | $\Delta$ PSNR (SL1 - MSE) | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ADE_train_00000026.jpg` | Low-chroma interior / floor | 0.030039 | 0.027887 | 27.43 dB | 27.61 dB | +0.18 dB | **Tied** |
| `ADE_train_00000028.jpg` | Outdoor field & dense foliage | 0.067865 | 0.067729 | 19.14 dB | 19.15 dB | +0.01 dB | **Tied** |
| `ADE_train_00000106.jpg` | Street & buildings | 0.034266 | 0.034940 | 25.37 dB | 25.30 dB | -0.06 dB | **Tied** |
| `ADE_train_00000150.jpg` | High-chroma indoor decor | 0.065001 | 0.064647 | 20.03 dB | 20.06 dB | +0.04 dB | **Tied** |
| `ADE_train_00000151.jpg` | Dining area / warm wood tones | 0.036334 | 0.034316 | 25.14 dB | 25.50 dB | +0.36 dB | **Smooth L1** |
| `ADE_train_00000156.jpg` | Room with bed & curtains | 0.026645 | 0.029867 | 27.86 dB | 27.21 dB | -0.66 dB | **MSE** |
| `ADE_train_00000203.jpg` | Neutral architecture / hallway | 0.032159 | 0.033952 | 26.22 dB | 26.23 dB | +0.01 dB | **Tied** |
| `ADE_train_00000324.jpg` | Room with vibrant furniture | 0.045849 | 0.042592 | 24.15 dB | 25.02 dB | +0.87 dB | **Smooth L1** |
| `ADE_train_00000354.jpg` | Kitchen interior | 0.035043 | 0.037222 | 23.60 dB | 23.31 dB | -0.29 dB | **MSE** |
| `ADE_train_00000378.jpg` | Soft lit indoor furniture | 0.029998 | 0.032443 | 27.54 dB | 26.78 dB | -0.76 dB | **MSE** |

### Breakdown:
* **Tied ($\pm 0.2\text{ dB}$)**: 5 / 10 images ($50\%$). Both models produce nearly indistinguishable pixel errors.
* **Smooth L1 Win ($> +0.2\text{ dB}$)**: 2 / 10 images ($20\%$). Clear improvements on scenes with rich warm chroma (`00000324.jpg` and `00000151.jpg`).
* **MSE Win ($< -0.2\text{ dB}$)**: 3 / 10 images ($30\%$). Slight advantage on low-saturation/neutral scenes where Smooth L1 adds modest chroma variance.

---

## 6. Qualitative Comparison & Colorization Behavior

Visual inspection of `outputs/analysis/visual_comparison_all.png` and `outputs/analysis/visual_comparison_representative.png` reveals several distinct colorization characteristics:

1. **Chroma Retention & Saturation**:
   * Across all test images, Smooth L1 yields higher mean chroma ($11.93$ vs $10.85$, representing **$75.3\%$ vs $68.4\%$ retention** of ground-truth chroma).
   * In warm indoor scenes (e.g., `00000324.jpg` and `00000151.jpg`), Smooth L1 reproduces warmer wood and upholstery tones with visibly less sepia haze.
2. **Dense Foliage & Green Reconstruction**:
   * In `ADE_train_00000028.jpg` (ground-truth chroma $26.71$), both models exhibit severe desaturation, predicting olive-brown / muted gray tones (MSE chroma $7.59$, Smooth L1 chroma $7.30$, both $\sim 19.15\text{ dB}$).
   * Neither loss successfully breaks the regression-to-the-mean barrier for complex vegetation without semantic priors or class-balancing weights.
3. **Neutral & Architectural Regions**:
   * On neutral gray/white surfaces (`00000378.jpg`), Smooth L1 occasionally introduces faint warm chromatic bleeding (mean chroma $11.83$ vs GT $6.81$), whereas MSE stays closer to neutral gray ($9.38$), explaining MSE's $+0.76\text{ dB}$ margin on that scene.
4. **Spatial Boundary Coherence**:
   * Both models utilize the identical U-Net skip-connection topology, resulting in sharp, well-aligned luminance boundaries without spatial distortion or unnatural edge ringing.

---

## 7. Interpretation

> **Question**: Did Smooth L1 improve colorization compared with MSE under the current experimental configuration?

### Evidence-Based Conclusion:
**No, Smooth L1 did not provide a decisive overall reconstruction improvement over MSE under the current configuration.**

While Smooth L1 modestly increased average chroma retention ($75.3\%$ vs $68.4\%$) and achieved superior reconstruction on specific warm indoor scenes (+0.87 dB on `00000324`), its aggregate performance across the test set is virtually tied with the baseline:
* Mean MAE: **$0.040560$ (Smooth L1) vs $0.040320$ (MSE)** ($\Delta = +0.6\%$)
* Mean MSE: **$0.004373$ (Smooth L1) vs $0.004389$ (MSE)** ($\Delta = -0.4\%$)
* Mean PSNR: **$24.62\text{ dB}$ (Smooth L1) vs $24.65\text{ dB}$ (MSE)** ($\Delta = -0.03\text{ dB}$)

Smooth L1 with default threshold $\beta = 1.0$ operates mostly in its quadratic regime because normalized $a^*b^*$ pixel errors are predominantly $< 1.0$. Consequently, its gradients behave similarly to MSE over the bulk of the training distribution, explaining the near-identical convergence trajectory and aggregate metrics.

---

## 8. Limitations

1. **Fixed Evaluation Sample Size**: Evaluation metrics are reported over a 10-image fixed subset; while deterministic and consistent across experiments, local variance in a few images strongly impacts aggregate averages.
2. **Single Parameter Configuration ($\beta = 1.0$)**: The transition threshold $\beta = 1.0$ is matched to the bounds of the normalized $[-1, 1]$ target range. In practice, most $a^*b^*$ errors are between $0.05$ and $0.30$, meaning the linear outlier regime was only engaged for extreme color errors.
3. **Unimodal Regression Limitation**: Both MSE and Smooth L1 are regression losses that fundamentally predict the mean or median of the output distribution. They cannot resolve multimodal color ambiguities (e.g., an object that could plausibly be red or blue).
4. **Single Random Seed**: Both runs used seed 42 to ensure control; variance across random initializations was not quantified.

---

## 9. Recommendations for Next Experiment

Based strictly on the empirical findings, the following directions represent the most principled next steps:

1. **Perceptual / Semantic Loss (LPIPS or Feature Loss)**:
   * Pixel regression losses (MSE, Smooth L1, L1) inevitably compromise chroma to minimize spatial variance. Adding a pretrained perceptual loss (e.g., VGG-based feature loss) directly penalizes blurry/desaturated predictions at a semantic feature level.
2. **Lower Beta Threshold for Smooth L1 ($\beta = 0.1 - 0.2$)**:
   * Tuning $\beta$ down into the empirical error range ($0.1$) would ensure the linear regime is actively engaged for typical chromatic errors, providing a much stronger gradient for moderately desaturated pixels.
3. **Chroma-Weighted or Class-Balanced Loss**:
   * Weighting pixels proportionally to their ground-truth chroma ($w = 1 + \alpha \sqrt{a^{*2} + b^{*2}}$) to directly penalize the observed foliage and high-saturation failure modes.

