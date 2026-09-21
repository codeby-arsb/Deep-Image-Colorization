# Investigation & Implementation Report: Perceptual / Feature Loss for Deep Image Colorization

**Step:** Step 7.4 — Investigation, Differentiable Architecture Design, and Feasibility Verification  
**Date:** September 21, 2026  
**Platform:** macOS Apple Silicon M5 (16 GB Unified Memory, Apple Silicon MPS Backend, PyTorch Automatic Mixed Precision)  
**Project:** Deep Image Colorization using U-Net  
**Repository:** `https://github.com/codeby-arsb/nndl-mini-project.git`  

---

## 1. Objective

In Experiments 1, 2, and 3 (Baseline MSE, Smooth L1, and Chroma-Weighted MSE), colorization was supervised purely through pixel-level regression objectives formulated on normalized CIE Lab chrominance ($a^*b^*$). 

While Chroma-Weighted MSE increased global chroma retention from **68.4%** to **77.4%** and improved reconstruction on vibrant scenes, our deep-dive into outdoor vegetation (`ADE_train_00000028.jpg`) exposed a fundamental structural limitation of pixel-wise losses:
- **Radial Symmetry and Directional Ambiguity:** Pixel-level losses penalize distance from target coordinates without understanding structural semantics or contextual naturalness. In vegetation, the network learned to boost yellow chrominance ($b^* = +8.22$), but predicted positive $a^*$ (+2.61 vs target -4.77), resulting in an unnatural muddy brown cast rather than true chlorophyll green.
- **The Averaging Pathology of Pixel Losses:** When an object can plausibly take multiple vibrant hues (e.g. green grass vs autumn turf), pixel-wise regression averages over multimodal distributions, collapsing toward the desaturated mean.

The scientific objective of this investigation is to answer:
> **Can a feature-space / perceptual loss supervision in reconstructed RGB space encourage the U-Net to produce more visually plausible, semantically coherent color distributions than pixel regression alone?**

---

## 2. Candidate Approaches Considered

We evaluated four candidate architectural strategies for incorporating perceptual supervision:

| Candidate Approach | Mechanism | Advantages | Disadvantages / Blockers | Feasibility |
| :--- | :--- | :--- | :--- | :---: |
| **1. ResNet-50 Feature Loss** | Extract intermediate activations after ResNet Residual Blocks | Modern residual architecture; widely used in classification | Residual skip connections pass low-frequency identity features; less sensitive to fine textural color boundaries than VGG | Moderate |
| **2. PatchGAN / Adversarial Discriminator** | Train an auxiliary convolutional discriminator network concurrently | Learns domain-specific color distribution; directly penalizes unrealistic color patches | Requires concurrent minimax optimization; high risk of mode collapse and training instability; breaks controlled loss comparison | High complexity; deferred to GAN stage |
| **3. Direct Lab-Space Feature Loss** | Pass normalized Lab directly into a custom feature network | Bypasses Lab $\rightarrow$ RGB color-space conversion | No standard pretrained feature extractors exist for Lab space; training an extractor from scratch introduces confounding variables | Unviable |
| **4. VGG-16 Image-Space Feature Loss (Selected)** | Differentiably reconstruct sRGB from $(L_{\text{in}}, ab_{\text{pred}})$, standardize with ImageNet statistics, and extract intermediate features via frozen VGG-16 | Quintessential standard for perceptual loss (Johnson et al., Ledig et al., Zhang et al.); VGG pooling hierarchy strongly correlates with human perceptual similarity | Requires fully differentiable in-graph Lab $\rightarrow$ sRGB conversion; higher memory footprint | **Highly Feasible & Scientifically Justified** |

---

## 3. Selected Approach: Architecture & Mathematical Formulation

### 3.1 End-to-End Differentiable Computation Graph

```text
Input L* [B, 1, 256, 256] in [-1, 1]
           │
      ColorizationUNet
           │
     Predicted ab [B, 2, 256, 256] in [-1, 1] ───► L_recon = MSE(ab_pred, ab_true)
           │
           ▼
     Differentiable Lab → RGB (src/utils.py: lab_to_rgb_torch)
     1. Denormalize: L = (L_norm + 1) * 50, ab = ab_norm * 128
     2. Lab → CIE 1931 XYZ (D65 standard illuminant reference)
     3. XYZ → Linear sRGB (ITU-R BT.709 D65 matrix)
     4. Linear sRGB → Non-linear sRGB (gamma correction)
           │
     Predicted sRGB [B, 3, 256, 256] in [0, 1]
           │
     ImageNet Standardization: (RGB - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
           │
     Frozen VGG-16 Feature Extractor (features[:9], relu2_2)
           │
     Feature Representation phi(RGB_pred) [B, 128, 128, 128]
           │
           ▼
     L_perceptual = MSE(phi(RGB_pred), phi(RGB_true))
           │
           ▼
     L_total = L_recon + lambda_perceptual * L_perceptual
```

### 3.2 Differentiable Color-Space Conversion (`lab_to_rgb_torch`)
To ensure gradients propagate back to the U-Net without breaking autograd, the conversion pipeline avoids NumPy, PIL, and in-place tensor operations:

1. **Denormalization:**
   $$L = (L_{\text{norm}} + 1.0) \times 50.0 \in [0, 100]$$
   $$a = a_{\text{norm}} \times 128.0 \in [-128, 128], \quad b = b_{\text{norm}} \times 128.0 \in [-128, 128]$$

2. **Lab $\rightarrow$ XYZ (D65 White Point: $X_n = 0.95047, Y_n = 1.00000, Z_n = 1.08883$):**
   $$f_y = \frac{L + 16.0}{116.0}, \quad f_x = \frac{a}{500.0} + f_y, \quad f_z = \text{clamp}\left(f_y - \frac{b}{200.0}, \min=0.0\right)$$
   For $t \in \{f_x, f_y, f_z\}$:
   $$f^{-1}(t) = \begin{cases} t^3 & \text{if } t > 0.2068966 \\ \frac{t - 16/116}{7.787} & \text{otherwise} \end{cases}$$
   $$X = 0.95047 \cdot f^{-1}(f_x), \quad Y = 1.00000 \cdot f^{-1}(f_y), \quad Z = 1.08883 \cdot f^{-1}(f_z)$$

3. **XYZ $\rightarrow$ Linear sRGB:**
   $$\begin{bmatrix} R_{\text{lin}} \\ G_{\text{lin}} \\ B_{\text{lin}} \end{bmatrix} = \begin{bmatrix} 3.24048134 & -1.53715152 & -0.49853633 \\ -0.96925495 & 1.87599000 & 0.04155593 \\ 0.05564664 & -0.20404134 & 1.05731107 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}$$

4. **Gamma Correction & Clamping:**
   $$C_{\text{srgb}} = \begin{cases} 12.92 \cdot C_{\text{lin}} & \text{if } C_{\text{lin}} \le 0.0031308 \\ 1.055 \cdot (C_{\text{lin}})^{1/2.4} - 0.055 & \text{otherwise} \end{cases}$$
   $$\text{RGB} = \text{clamp}(C_{\text{srgb}}, 0.0, 1.0)$$

**Numerical Verification:** Compared against `skimage.color.lab2rgb` on a dense 3D lattice of 512 CIE Lab coordinates spanning $L \in [0, 100], a \in [-100, 100], b \in [-100, 100]$, the maximum absolute discrepancy was **$6.88 \times 10^{-7}$** and mean discrepancy was **$3.02 \times 10^{-8}$**, establishing numerical equivalence.

### 3.3 Selected Layer: VGG-16 `relu2_2`
In `torchvision.models.vgg16`, `features[:9]` extracts the activations immediately following the second convolutional block (after `Conv2_2` and its `ReLU`):
- **Spatial Resolution:** $128 \times 128$ (half of input resolution), preserving spatial color boundary alignment.
- **Channel Depth:** 128 feature maps capturing mid-level textures, local contrast, and chromatic edges.
- **Efficiency:** By slicing the network at index 9, the remaining 22 layers (Conv3, Conv4, Conv5, and all fully connected classifiers) are neither loaded nor computed during the forward/backward passes, saving memory and compute on Apple Silicon MPS.

### 3.4 Loss Formulation & Hyperparameter Justification
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(ab_{\text{pred}}, ab_{\text{true}}) + \lambda_{\text{perceptual}} \cdot \frac{1}{CHW} \sum_{c, h, w} \left(\phi(\text{RGB}_{\text{pred}}) - \phi(\text{RGB}_{\text{true}})\right)^2$$

**Scale Analysis for $\lambda_{\text{perceptual}}$:**
Empirical measurements on ADE20K validation images showed:
- Pixel-level reconstruction MSE on $ab \in [-1, 1]$ is typically $\approx 0.009 - 0.015$.
- VGG-16 `relu2_2` feature MSE on standardized images is typically $\approx 0.50 - 1.20$ ($\sim 50\times - 70\times$ larger in raw numerical magnitude).
- Setting **$\lambda_{\text{perceptual}} = 0.01$** scales the perceptual term to $\approx 0.005 - 0.012$, ensuring that the perceptual loss provides a balanced, guiding gradient without overpowering the pixel-level ground-truth chromatic coordinate anchors.

---

## 4. Technical Feasibility & Verification Results

### 4.1 Unit Test Suite Results (`scripts/test_perceptual_loss.py`)
All 11 unit tests passed with 100% success:

| Test ID | Objective | Verified Property | Status |
| :--- | :--- | :--- | :---: |
| **Test A** | Perfect Prediction | When $ab_{\text{pred}} = ab_{\text{true}}$, $\mathcal{L}_{\text{perceptual}} = 0.00000000$ | **PASS** |
| **Test B** | Perturbed Prediction | Perturbations yield strictly positive loss ($0.100337 > 0$) | **PASS** |
| **Test C** | Gradient Flow | Autograd produces finite, non-zero gradients to $ab_{\text{pred}}$ (`[-0.00459, +0.00519]`) | **PASS** |
| **Test D** | Parameter Freezing | All 260,160 VGG parameters have `requires_grad == False` | **PASS** |
| **Test E** | Eval Mode Locking | Feature extractor stays in `eval()` mode even when parent module enters `train()` | **PASS** |
| **Test F** | CPU Execution | Forward and backward passes execute cleanly on CPU | **PASS** |
| **Test G** | MPS Execution | Forward and backward passes execute natively on Apple Silicon MPS | **PASS** |
| **Test H** | Batch Shapes | Forward pass operates seamlessly across batch sizes 1, 4, 8 | **PASS** |
| **Test I** | Numerical Stability | Extreme boundary values in Lab produce strictly valid RGB in $[0, 1]$ without NaN/Inf | **PASS** |
| **Test J** | skimage Equivalence | Max absolute difference vs `skimage.color.lab2rgb` is $6.88 \times 10^{-7} < 10^{-4}$ | **PASS** |
| **Test K** | Non-Regression | MSE, Smooth L1, and Chroma-Weighted MSE remain fully functional | **PASS** |

### 4.2 Smoke Test Validation (`python -m src.train --smoke-test --loss perceptual --amp`)
- **Forward Pass:** Successfully computed for 5 train batches and 2 validation batches.
- **Loss Finite:** PASS (Final Train Loss: 0.6729, Val Loss: 0.1800; zero NaN/Inf).
- **Backward Pass:** PASS (Gradients computed and backpropagated through VGG, Lab-to-RGB, and U-Net).
- **Parameter Update Test:** PASS (Model weights updated after Adam step).
- **Checkpoint Integrity:** PASS (Reloaded checkpoint and verified shape `[8, 2, 256, 256]`).
- **Memory Footprint:** Peak allocated memory on MPS was **1081.48 MB** (well within the 16 GB Unified Memory capacity).
- **Smoke Test Execution Time:** 3.32 seconds on Apple Silicon M5.

---

## 5. Potential Risks & Failure Modes

1. **ImageNet Semantic Prior Bias:**
   VGG-16 was pretrained for classification on ImageNet RGB images. Its feature representations prioritize discriminative category boundaries (e.g. dog ears, wheels) rather than pure color naturalness. In colorization, this can occasionally manifest as hallucinated texture-color artifacts if the perceptual weight is too high.
2. **Computational Overhead:**
   Extracting VGG features on reconstructed RGB adds forward and backward computation steps. In our benchmark, a forward+backward pass of VGG `relu2_2` took ~68.7 ms per batch of 8 images on MPS, increasing overall per-epoch training time from ~6.2 minutes (pure MSE) to ~8.5–9.5 minutes.
3. **Loss Balancing Sensitivity ($\lambda$):**
   If $\lambda_{\text{perceptual}}$ is too small ($< 0.001$), the network behaves identically to baseline MSE. If $\lambda_{\text{perceptual}}$ is too large ($> 0.1$), the network may sacrifice low-level pixel alignment in favor of high-level feature activations, causing edge bleed and color fringes. $\lambda = 0.01$ is the documented starting point.

---

## 6. Checkpoint Integrity & Experiment Isolation

To preserve previous scientific benchmarks:
- **Baseline MSE Checkpoint:** `outputs/checkpoints/best.pth` (MD5: `4f04f028ef59d8dc11d7968a4e3d05ac` — **VERIFIED UNTOUCHED**).
- **Smooth L1 Checkpoint:** `outputs/experiments/smooth_l1/checkpoints/best.pth` (MD5: `83fbaed6909c309d92da38118fb11aa2` — **VERIFIED UNTOUCHED**).
- **Chroma-Weighted Checkpoint:** `outputs/experiments/chroma_weighted/checkpoints/best.pth` (MD5: `7b1240ae4610f145a35c6d0600ee0b26` — **VERIFIED UNTOUCHED**).
- **Perceptual Output Directory:** Designated exclusively as `outputs/experiments/perceptual/`.

---

## 7. Decision

**The perceptual loss pipeline is technically verified, numerically stable, autograd-differentiable, and fully ready for a controlled 20-epoch training experiment.**

Per the strict instructions for Step 7.4, **no full training experiment was initiated**. Training is awaiting explicit instruction.
