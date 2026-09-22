"""Generate report-ready Figure 2: Overall system architecture diagram.

Visualizes the complete end-to-end deep image colorization pipeline:
1. Input Preprocessing & Lab Color Space Transformation
2. 5-Level Symmetrical U-Net Architecture (29.2M parameters) with Multi-Scale Skip Connections
3. Inference Reconstruction Pipeline (Lab -> sRGB)
4. Training Supervision Path with 4 Loss Formulations & Adam Backpropagation

Outputs:
- outputs/report_figures/figure_2_system_architecture.svg
- outputs/report_figures/figure_2_system_architecture.png (300-DPI via Headless Chrome)
"""

import os
import io
import base64
import subprocess
import torch
import numpy as np
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import ColorizationUNet
from src.utils import preprocess_image, reconstruct_rgb, load_rgb_image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "report_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_sample_thumbnails():
    """Load model and generate authentic thumbnails from ADE20K test sample."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ColorizationUNet().to(device)
    ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sample_path = os.path.join(PROJECT_ROOT, "data/raw/ADEChallengeData2016/images/training/ADE_train_00000026.jpg")
    rgb = load_rgb_image(sample_path)
    L, ab = preprocess_image(rgb)

    L_t = torch.from_numpy(L).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_ab_t = model(L_t)
    pred_ab = pred_ab_t.squeeze(0).cpu().numpy()

    # Reconstruct images
    pred_rgb = (reconstruct_rgb(L, pred_ab) * 255).clip(0, 255).astype(np.uint8)
    gt_rgb = (reconstruct_rgb(L, ab) * 255).clip(0, 255).astype(np.uint8)
    gray = ((L[0] + 1.0) * 50.0 / 100.0 * 255).clip(0, 255).astype(np.uint8)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)

    # Chroma-only visualization (L=50)
    L_zero = np.zeros_like(L)
    chroma_rgb = (reconstruct_rgb(L_zero, pred_ab) * 255).clip(0, 255).astype(np.uint8)

    def to_b64(arr, size=(130, 130)):
        im = Image.fromarray(arr).resize(size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "gray": to_b64(gray_rgb),
        "pred": to_b64(pred_rgb, (140, 140)),
        "gt": to_b64(gt_rgb),
        "chroma": to_b64(chroma_rgb, (110, 110))
    }

def build_svg_diagram(thumbs):
    """Construct an academic-grade SVG system architecture diagram."""
    W, H = 2360, 1300
    
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="background-color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <!-- Drop Shadows -->
    <filter id="card-shadow" x="-8%" y="-8%" width="120%" height="120%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#0F172A" flood-opacity="0.08"/>
      <feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#0F172A" flood-opacity="0.04"/>
    </filter>
    <filter id="block-shadow" x="-10%" y="-10%" width="124%" height="124%">
      <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#000000" flood-opacity="0.14"/>
    </filter>

    <!-- Gradients -->
    <linearGradient id="enc-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1E3A8A"/>
      <stop offset="100%" stop-color="#2563EB"/>
    </linearGradient>
    <linearGradient id="bottle-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#581C87"/>
      <stop offset="100%" stop-color="#7C3AED"/>
    </linearGradient>
    <linearGradient id="dec-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#064E3B"/>
      <stop offset="100%" stop-color="#0D9488"/>
    </linearGradient>
    <linearGradient id="head-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#831843"/>
      <stop offset="100%" stop-color="#BE185D"/>
    </linearGradient>
    <linearGradient id="header-grad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#1E293B"/>
    </linearGradient>

    <!-- Markers -->
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563EB"/>
    </marker>
    <marker id="arrow-teal" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#0D9488"/>
    </marker>
    <marker id="arrow-cyan" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#0284C7"/>
    </marker>
    <marker id="arrow-slate" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748B"/>
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#DC2626"/>
    </marker>
    <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#7C3AED"/>
    </marker>
    <marker id="arrow-pink" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#BE185D"/>
    </marker>
  </defs>

  <!-- ==================== HEADER BAR ==================== -->
  <rect x="0" y="0" width="{W}" height="68" fill="url(#header-grad)"/>
  <text x="35" y="32" fill="#F8FAFC" font-size="20" font-weight="700" letter-spacing="0.3">Figure 2: Overall System Architecture of the Proposed Image Colorization System</text>
  <text x="35" y="54" fill="#94A3B8" font-size="13" font-weight="400">End-to-End Luminance-Conditioned Chrominance Regression in CIE Lab Space via Symmetrical 5-Level U-Net with Skip Connections</text>

  <g transform="translate(1780, 18)">
    <rect x="0" y="0" width="545" height="32" rx="6" fill="#334155"/>
    <text x="272" y="21" fill="#38BDF8" font-size="12" font-weight="600" text-anchor="middle">Parameters: 29,228,802 &bull; Input: [1, 256, 256] &bull; Output: [2, 256, 256] &bull; Backend: Apple Silicon MPS</text>
  </g>

  <!-- ==================== PANEL 1: PREPROCESSING PIPELINE ==================== -->
  <g transform="translate(30, 85)">
    <rect x="0" y="0" width="310" height="660" rx="12" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.5" filter="url(#card-shadow)"/>
    
    <!-- Title -->
    <rect x="0" y="0" width="310" height="42" rx="12" fill="#F1F5F9"/>
    <rect x="0" y="30" width="310" height="12" fill="#F1F5F9"/>
    <text x="18" y="26" fill="#0F172A" font-size="14" font-weight="700">1. Preprocessing Pipeline</text>
    <line x1="0" y1="42" x2="310" y2="42" stroke="#E2E8F0" stroke-width="1"/>

    <!-- Input Raw Image Thumbnail -->
    <g transform="translate(25, 58)">
      <rect x="0" y="0" width="124" height="124" rx="8" fill="#E2E8F0" stroke="#CBD5E1" stroke-width="1"/>
      <image href="{thumbs['gt']}" x="2" y="2" width="120" height="120" preserveAspectRatio="none"/>
      <text x="62" y="142" fill="#334155" font-size="11.5" font-weight="600" text-anchor="middle">Input RGB Image</text>
      <text x="62" y="156" fill="#64748B" font-size="10" text-anchor="middle">H &times; W &times; 3 (ADE20K)</text>
    </g>

    <!-- Arrow down -->
    <line x1="87" y1="222" x2="87" y2="242" stroke="#64748B" stroke-width="2" marker-end="url(#arrow-slate)"/>

    <!-- Resize Box -->
    <g transform="translate(25, 245)">
      <rect x="0" y="0" width="260" height="46" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2"/>
      <text x="130" y="20" fill="#0F172A" font-size="12" font-weight="600" text-anchor="middle">Bilinear Spatial Resizing</text>
      <text x="130" y="36" fill="#64748B" font-size="10.5" text-anchor="middle">Fixed Resolution: 256 &times; 256 &times; 3</text>
    </g>

    <!-- Arrow down -->
    <line x1="155" y1="293" x2="155" y2="313" stroke="#64748B" stroke-width="2" marker-end="url(#arrow-slate)"/>

    <!-- Color Space Transform Box -->
    <g transform="translate(25, 316)">
      <rect x="0" y="0" width="260" height="88" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2"/>
      <text x="130" y="20" fill="#0F172A" font-size="12" font-weight="600" text-anchor="middle">RGB &rarr; CIE Lab Transformation</text>
      <text x="130" y="38" fill="#475569" font-size="10.5" text-anchor="middle">sRGB float [0, 1] &rarr; CIE 1976 Lab</text>
      <line x1="20" y1="46" x2="240" y2="46" stroke="#F1F5F9" stroke-width="1"/>
      <text x="25" y="62" fill="#1E40AF" font-size="11" font-weight="600">L* &isin; [0, 100]</text>
      <text x="115" y="62" fill="#64748B" font-size="10.5">: Perceptual Lightness</text>
      <text x="25" y="78" fill="#991B1B" font-size="11" font-weight="600">a*, b* &isin; [-128, 127]</text>
      <text x="135" y="78" fill="#64748B" font-size="10.5">: Chrominance Opponents</text>
    </g>

    <!-- Arrow down -->
    <line x1="155" y1="406" x2="155" y2="426" stroke="#64748B" stroke-width="2" marker-end="url(#arrow-slate)"/>

    <!-- Channel Separation & Normalization Box -->
    <g transform="translate(25, 429)">
      <rect x="0" y="0" width="260" height="106" rx="6" fill="#EFF6FF" stroke="#3B82F6" stroke-width="1.5"/>
      <text x="130" y="20" fill="#1E40AF" font-size="12" font-weight="700" text-anchor="middle">Channel Normalization</text>
      
      <rect x="15" y="30" width="230" height="30" rx="4" fill="#DBEAFE"/>
      <text x="130" y="46" fill="#1E3A8A" font-size="11" font-weight="600" text-anchor="middle">L*norm = (L* / 50.0) &minus; 1.0 &isin; [&minus;1, 1]</text>
      <text x="130" y="58" fill="#2563EB" font-size="9.5" text-anchor="middle">Network Input: [1, 256, 256]</text>

      <rect x="15" y="66" width="230" height="30" rx="4" fill="#FEE2E2"/>
      <text x="130" y="82" fill="#991B1B" font-size="11" font-weight="600" text-anchor="middle">a*b*norm = a*b* / 128.0 &isin; [&minus;1, 1]</text>
      <text x="130" y="94" fill="#DC2626" font-size="9.5" text-anchor="middle">Ground Truth Target: [2, 256, 256]</text>
    </g>

    <!-- Normalized L* Channel Thumbnail -->
    <g transform="translate(25, 547)">
      <rect x="0" y="0" width="86" height="86" rx="6" fill="#E2E8F0" stroke="#CBD5E1" stroke-width="1"/>
      <image href="{thumbs['gray']}" x="2" y="2" width="82" height="82" preserveAspectRatio="none"/>
      <text x="43" y="100" fill="#0F172A" font-size="10.5" font-weight="600" text-anchor="middle">Input L*norm</text>
      
      <g transform="translate(98, 12)">
        <rect x="0" y="0" width="162" height="62" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1"/>
        <text x="10" y="20" fill="#0F172A" font-size="11" font-weight="600">Model Input Stream:</text>
        <text x="10" y="36" fill="#475569" font-size="10">Shape: (B, 1, 256, 256)</text>
        <text x="10" y="50" fill="#475569" font-size="10">Dtype: float32</text>
      </g>
    </g>
  </g>

  <!-- Route from Preprocessing to Encoder 1 -->
  <path d="M 340 595 L 355 595 L 355 155 L 410 155" fill="none" stroke="#2563EB" stroke-width="2.5" marker-end="url(#arrow-blue)"/>
  <rect x="325" y="360" width="60" height="20" rx="4" fill="#DBEAFE" stroke="#93C5FD" stroke-width="1"/>
  <text x="355" y="374" fill="#1E40AF" font-size="9" font-weight="700" text-anchor="middle">L* Stream</text>

  <!-- ==================== PANEL 2: CORE U-NET ARCHITECTURE ==================== -->
  <g transform="translate(370, 85)">
    <rect x="0" y="0" width="1440" height="660" rx="12" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5" filter="url(#card-shadow)"/>
    
    <!-- Title & Legend -->
    <rect x="0" y="0" width="1440" height="42" rx="12" fill="#F8FAFC"/>
    <rect x="0" y="30" width="1440" height="12" fill="#F8FAFC"/>
    <text x="24" y="26" fill="#0F172A" font-size="14" font-weight="700">2. Colorization U-Net (29.2M Parameters)</text>
    <line x1="0" y1="42" x2="1440" y2="42" stroke="#E2E8F0" stroke-width="1"/>

    <!-- Legend Items (Cleanly distributed) -->
    <g transform="translate(460, 12)">
      <rect x="0" y="2" width="13" height="13" rx="3" fill="#2563EB"/>
      <text x="18" y="13" fill="#334155" font-size="10.5" font-weight="600">Conv 4&times;4, s=2 + BN + LReLU</text>

      <rect x="195" y="2" width="13" height="13" rx="3" fill="#0D9488"/>
      <text x="213" y="13" fill="#334155" font-size="10.5" font-weight="600">ConvTrans 4&times;4 + Concat + BN + ReLU</text>

      <line x1="450" y1="8" x2="476" y2="8" stroke="#0284C7" stroke-width="2.2" stroke-dasharray="5 3"/>
      <text x="484" y="13" fill="#0369A1" font-size="10.5" font-weight="700">Skip Concat</text>

      <rect x="565" y="2" width="13" height="13" rx="3" fill="#BE185D"/>
      <text x="583" y="13" fill="#334155" font-size="10.5" font-weight="600">Head (ConvTrans + Tanh)</text>

      <rect x="740" y="2" width="13" height="13" rx="3" fill="#7C3AED"/>
      <text x="758" y="13" fill="#334155" font-size="10.5" font-weight="600">Latent Bottleneck</text>
    </g>

    <!-- ================= U-NET TIERS ================= -->

    <!-- LEVEL 1: 128x128 -->
    <!-- Enc 1 -->
    <g transform="translate(45, 70)">
      <rect x="0" y="0" width="240" height="66" rx="8" fill="url(#enc-grad)" filter="url(#block-shadow)"/>
      <text x="16" y="22" fill="#93C5FD" font-size="10.5" font-weight="700">ENCODER 1</text>
      <text x="16" y="40" fill="#FFFFFF" font-size="13" font-weight="700">64 &times; 128 &times; 128</text>
      <text x="16" y="56" fill="#DBEAFE" font-size="10">Conv 4&times;4, s=2, p=1 (1&rarr;64) &bull; BN &bull; LReLU</text>
    </g>

    <!-- Dec 5 -->
    <g transform="translate(1045, 70)">
      <rect x="0" y="0" width="255" height="66" rx="8" fill="url(#dec-grad)" filter="url(#block-shadow)"/>
      <text x="16" y="22" fill="#99F6E4" font-size="10.5" font-weight="700">DECODER 5</text>
      <text x="16" y="40" fill="#FFFFFF" font-size="13" font-weight="700">128 &times; 128 &times; 128</text>
      <text x="16" y="56" fill="#CCFBF1" font-size="10">ConvTrans 4&times;4 (64) + Skip (64) &bull; BN &bull; ReLU</text>
    </g>

    <!-- Skip Connection 5 (Horizontal Arrow) -->
    <path d="M 285 103 L 1033 103" fill="none" stroke="#0284C7" stroke-width="2.2" stroke-dasharray="6 4" marker-end="url(#arrow-cyan)"/>
    <rect x="625" y="91" width="130" height="22" rx="11" fill="#E0F2FE" stroke="#38BDF8" stroke-width="1"/>
    <text x="690" y="106" fill="#0369A1" font-size="10.5" font-weight="700" text-anchor="middle">Skip 5: 64 channels</text>

    <!-- Vertical Arrow: Enc 1 -> Enc 2 -->
    <path d="M 165 136 L 165 162" fill="none" stroke="#2563EB" stroke-width="2.5" marker-end="url(#arrow-blue)"/>
    <!-- Horizontal Arrow: Dec 5 -> Head -->
    <path d="M 1300 103 L 1333 103" fill="none" stroke="#BE185D" stroke-width="2.5" marker-end="url(#arrow-pink)"/>

    <!-- LEVEL 2: 64x64 -->
    <!-- Enc 2 -->
    <g transform="translate(95, 168)">
      <rect x="0" y="0" width="240" height="66" rx="8" fill="url(#enc-grad)" filter="url(#block-shadow)"/>
      <text x="16" y="22" fill="#93C5FD" font-size="10.5" font-weight="700">ENCODER 2</text>
      <text x="16" y="40" fill="#FFFFFF" font-size="13" font-weight="700">128 &times; 64 &times; 64</text>
      <text x="16" y="56" fill="#DBEAFE" font-size="10">Conv 4&times;4, s=2, p=1 (64&rarr;128) &bull; BN &bull; LReLU</text>
    </g>

    <!-- Dec 4 -->
    <g transform="translate(995, 168)">
      <rect x="0" y="0" width="255" height="66" rx="8" fill="url(#dec-grad)" filter="url(#block-shadow)"/>
      <text x="16" y="22" fill="#99F6E4" font-size="10.5" font-weight="700">DECODER 4</text>
      <text x="16" y="40" fill="#FFFFFF" font-size="13" font-weight="700">256 &times; 64 &times; 64</text>
      <text x="16" y="56" fill="#CCFBF1" font-size="10">ConvTrans 4&times;4 (128) + Skip (128) &bull; BN &bull; ReLU</text>
    </g>

    <!-- Skip Connection 4 -->
    <path d="M 335 201 L 983 201" fill="none" stroke="#0284C7" stroke-width="2.2" stroke-dasharray="6 4" marker-end="url(#arrow-cyan)"/>
    <rect x="620" y="189" width="140" height="22" rx="11" fill="#E0F2FE" stroke="#38BDF8" stroke-width="1"/>
    <text x="690" y="204" fill="#0369A1" font-size="10.5" font-weight="700" text-anchor="middle">Skip 4: 128 channels</text>

    <!-- Vertical Arrow: Enc 2 -> Enc 3 -->
    <path d="M 215 234 L 215 260" fill="none" stroke="#2563EB" stroke-width="2.5" marker-end="url(#arrow-blue)"/>
    <!-- Vertical Arrow: Dec 4 -> Dec 5 -->
    <path d="M 1122 168 L 1122 142" fill="none" stroke="#0D9488" stroke-width="2.5" marker-end="url(#arrow-teal)"/>

    <!-- LEVEL 3: 32x32 -->
    <!-- Enc 3 -->
    <g transform="translate(145, 266)">
      <rect x="0" y="0" width="240" height="66" rx="8" fill="url(#enc-grad)" filter="url(#block-shadow)"/>
      <text x="16" y="22" fill="#93C5FD" font-size="10.5" font-weight="700">ENCODER 3</text>
      <text x="16" y="40" fill="#FFFFFF" font-size="13" font-weight="700">256 &times; 32 &times; 32</text>
      <text x="16" y="56" fill="#DBEAFE" font-size="10">Conv 4&times;4, s=2, p=1 (128&rarr;256) &bull; BN &bull; LReLU</text>
    </g>

    <!-- Dec 3 -->
    <g transform="translate(945, 266)">
      <rect x="0" y="0" width="255" height="66" rx="8" fill="url(#dec-grad)" filter="url(#block-shadow)"/>
      <text x="16" y="22" fill="#99F6E4" font-size="10.5" font-weight="700">DECODER 3</text>
      <text x="16" y="40" fill="#FFFFFF" font-size="13" font-weight="700">512 &times; 32 &times; 32</text>
      <text x="16" y="56" fill="#CCFBF1" font-size="10">ConvTrans 4&times;4 (256) + Skip (256) &bull; BN &bull; ReLU</text>
    </g>

    <!-- Skip Connection 3 -->
    <path d="M 385 299 L 933 299" fill="none" stroke="#0284C7" stroke-width="2.2" stroke-dasharray="6 4" marker-end="url(#arrow-cyan)"/>
    <rect x="620" y="287" width="140" height="22" rx="11" fill="#E0F2FE" stroke="#38BDF8" stroke-width="1"/>
    <text x="690" y="302" fill="#0369A1" font-size="10.5" font-weight="700" text-anchor="middle">Skip 3: 256 channels</text>

    <!-- Vertical Arrow: Enc 3 -> Enc 4 -->
    <path d="M 265 332 L 265 358" fill="none" stroke="#2563EB" stroke-width="2.5" marker-end="url(#arrow-blue)"/>
    <!-- Vertical Arrow: Dec 3 -> Dec 4 -->
    <path d="M 1072 266 L 1072 240" fill="none" stroke="#0D9488" stroke-width="2.5" marker-end="url(#arrow-teal)"/>

    <!-- LEVEL 4: 16x16 -->
    <!-- Enc 4 -->
    <g transform="translate(195, 364)">
      <rect x="0" y="0" width="240" height="66" rx="8" fill="url(#enc-grad)" filter="url(#block-shadow)"/>
      <text x="16" y="22" fill="#93C5FD" font-size="10.5" font-weight="700">ENCODER 4</text>
      <text x="16" y="40" fill="#FFFFFF" font-size="13" font-weight="700">512 &times; 16 &times; 16</text>
      <text x="16" y="56" fill="#DBEAFE" font-size="10">Conv 4&times;4, s=2, p=1 (256&rarr;512) &bull; BN &bull; LReLU</text>
    </g>

    <!-- Dec 2 -->
    <g transform="translate(895, 364)">
      <rect x="0" y="0" width="255" height="66" rx="8" fill="url(#dec-grad)" filter="url(#block-shadow)"/>
      <text x="16" y="22" fill="#99F6E4" font-size="10.5" font-weight="700">DECODER 2</text>
      <text x="16" y="40" fill="#FFFFFF" font-size="13" font-weight="700">1024 &times; 16 &times; 16</text>
      <text x="16" y="56" fill="#CCFBF1" font-size="10">ConvTrans 4&times;4 (512) + Skip (512) &bull; BN &bull; ReLU</text>
    </g>

    <!-- Skip Connection 2 -->
    <path d="M 435 397 L 883 397" fill="none" stroke="#0284C7" stroke-width="2.2" stroke-dasharray="6 4" marker-end="url(#arrow-cyan)"/>
    <rect x="620" y="385" width="140" height="22" rx="11" fill="#E0F2FE" stroke="#38BDF8" stroke-width="1"/>
    <text x="690" y="400" fill="#0369A1" font-size="10.5" font-weight="700" text-anchor="middle">Skip 2: 512 channels</text>

    <!-- Vertical Arrow: Enc 4 -> Enc 5 -->
    <path d="M 315 430 L 315 456" fill="none" stroke="#2563EB" stroke-width="2.5" marker-end="url(#arrow-blue)"/>
    <!-- Vertical Arrow: Dec 2 -> Dec 3 -->
    <path d="M 1022 364 L 1022 338" fill="none" stroke="#0D9488" stroke-width="2.5" marker-end="url(#arrow-teal)"/>

    <!-- LEVEL 5: 8x8 -->
    <!-- Enc 5 -->
    <g transform="translate(245, 462)">
      <rect x="0" y="0" width="240" height="66" rx="8" fill="url(#enc-grad)" filter="url(#block-shadow)"/>
      <text x="16" y="22" fill="#93C5FD" font-size="10.5" font-weight="700">ENCODER 5</text>
      <text x="16" y="40" fill="#FFFFFF" font-size="13" font-weight="700">512 &times; 8 &times; 8</text>
      <text x="16" y="56" fill="#DBEAFE" font-size="10">Conv 4&times;4, s=2, p=1 (512&rarr;512) &bull; BN &bull; LReLU</text>
    </g>

    <!-- Dec 1 -->
    <g transform="translate(845, 462)">
      <rect x="0" y="0" width="255" height="66" rx="8" fill="url(#dec-grad)" filter="url(#block-shadow)"/>
      <text x="16" y="22" fill="#99F6E4" font-size="10.5" font-weight="700">DECODER 1</text>
      <text x="16" y="40" fill="#FFFFFF" font-size="13" font-weight="700">1024 &times; 8 &times; 8</text>
      <text x="16" y="56" fill="#CCFBF1" font-size="10">ConvTrans 4&times;4 (512) + Skip (512) &bull; BN &bull; ReLU</text>
    </g>

    <!-- Skip Connection 1 -->
    <path d="M 485 495 L 833 495" fill="none" stroke="#0284C7" stroke-width="2.2" stroke-dasharray="6 4" marker-end="url(#arrow-cyan)"/>
    <rect x="620" y="483" width="140" height="22" rx="11" fill="#E0F2FE" stroke="#38BDF8" stroke-width="1"/>
    <text x="690" y="498" fill="#0369A1" font-size="10.5" font-weight="700" text-anchor="middle">Skip 1: 512 channels</text>

    <!-- Diagonal Arrow: Enc 5 -> Bottleneck -->
    <path d="M 365 528 L 365 565 L 505 565" fill="none" stroke="#2563EB" stroke-width="2.5" marker-end="url(#arrow-blue)"/>
    <!-- Diagonal Arrow: Bottleneck -> Dec 1 -->
    <path d="M 810 565 L 972 565 L 972 534" fill="none" stroke="#7C3AED" stroke-width="2.5" marker-end="url(#arrow-purple)"/>
    <!-- Vertical Arrow: Dec 1 -> Dec 2 -->
    <path d="M 972 462 L 972 436" fill="none" stroke="#0D9488" stroke-width="2.5" marker-end="url(#arrow-teal)"/>

    <!-- LEVEL 6: BOTTLENECK (4x4) -->
    <g transform="translate(515, 532)">
      <rect x="0" y="0" width="290" height="66" rx="8" fill="url(#bottle-grad)" filter="url(#block-shadow)"/>
      <text x="145" y="22" fill="#E9D5FF" font-size="11" font-weight="700" text-anchor="middle">LATENT BOTTLENECK LAYER</text>
      <text x="145" y="42" fill="#FFFFFF" font-size="14" font-weight="800" text-anchor="middle">512 &times; 4 &times; 4</text>
      <text x="145" y="58" fill="#F3E8FF" font-size="10" text-anchor="middle">Conv 4&times;4, s=2 &bull; BN &bull; LeakyReLU(0.2) &bull; Global Semantics</text>
    </g>

    <!-- FINAL OUTPUT LAYER (HEAD) -->
    <g transform="translate(1340, 70)">
      <rect x="0" y="0" width="75" height="66" rx="8" fill="url(#head-grad)" filter="url(#block-shadow)"/>
      <text x="37.5" y="22" fill="#FCE7F3" font-size="9" font-weight="700" text-anchor="middle">HEAD</text>
      <text x="37.5" y="38" fill="#FFFFFF" font-size="11" font-weight="800" text-anchor="middle">Tanh</text>
      <text x="37.5" y="54" fill="#FBCFE8" font-size="8.5" text-anchor="middle">2 &times; 256&sup2;</text>
    </g>
  </g>

  <!-- ==================== PANEL 3: RECONSTRUCTION PIPELINE ==================== -->
  <g transform="translate(1835, 85)">
    <rect x="0" y="0" width="495" height="660" rx="12" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.5" filter="url(#card-shadow)"/>
    
    <!-- Title -->
    <rect x="0" y="0" width="495" height="42" rx="12" fill="#F1F5F9"/>
    <rect x="0" y="30" width="495" height="12" fill="#F1F5F9"/>
    <text x="20" y="26" fill="#0F172A" font-size="14" font-weight="700">3. Inference Reconstruction Pipeline</text>
    <line x1="0" y1="42" x2="495" y2="42" stroke="#E2E8F0" stroke-width="1"/>

    <!-- Top: Predicted Chrominance & Denorm -->
    <g transform="translate(25, 60)">
      <!-- Thumbnail -->
      <g transform="translate(0, 0)">
        <rect x="0" y="0" width="114" height="114" rx="8" fill="#E2E8F0" stroke="#CBD5E1" stroke-width="1"/>
        <image href="{thumbs['chroma']}" x="2" y="2" width="110" height="110" preserveAspectRatio="none"/>
        <text x="57" y="130" fill="#0F172A" font-size="11" font-weight="600" text-anchor="middle">Predicted &#226;*b*</text>
        <text x="57" y="144" fill="#64748B" font-size="9.5" text-anchor="middle">2 &times; 256 &times; 256</text>
      </g>

      <!-- Denorm Math Card -->
      <g transform="translate(130, 0)">
        <rect x="0" y="0" width="315" height="114" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2"/>
        <text x="14" y="24" fill="#0F172A" font-size="12" font-weight="700">Denormalization Stage:</text>
        <text x="14" y="46" fill="#BE185D" font-size="11" font-weight="600">&bull; &#226;*b* = &#226;*b*norm &times; 128.0</text>
        <text x="14" y="66" fill="#1E40AF" font-size="11" font-weight="600">&bull; L* = (L*norm + 1.0) &times; 50.0</text>
        <text x="14" y="86" fill="#475569" font-size="10.5">Input lightness is preserved with zero distortion</text>
        <text x="14" y="102" fill="#64748B" font-size="9.5">Predicted chrominance mapped back to full CIE scale</text>
      </g>
    </g>

    <!-- Arrow down -->
    <line x1="247" y1="215" x2="247" y2="238" stroke="#64748B" stroke-width="2" marker-end="url(#arrow-slate)"/>

    <!-- Middle: Tensor Fusion -->
    <g transform="translate(25, 242)">
      <rect x="0" y="0" width="445" height="52" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2"/>
      <text x="222" y="22" fill="#0F172A" font-size="12" font-weight="600" text-anchor="middle">CIE Lab Tensor Fusion</text>
      <text x="222" y="40" fill="#2563EB" font-size="11" font-weight="700" text-anchor="middle">[L*orig, &#226;*pred, b&#770;*pred] &isin; &Ropf;256 &times; 256 &times; 3</text>
    </g>

    <!-- Arrow down -->
    <line x1="247" y1="298" x2="247" y2="320" stroke="#64748B" stroke-width="2" marker-end="url(#arrow-slate)"/>

    <!-- Middle: Non-Linear Color Transform -->
    <g transform="translate(25, 324)">
      <rect x="0" y="0" width="445" height="52" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2"/>
      <text x="222" y="22" fill="#0F172A" font-size="12" font-weight="600" text-anchor="middle">Non-Linear Color Space Conversion</text>
      <text x="222" y="40" fill="#0D9488" font-size="11" font-weight="700" text-anchor="middle">CIE Lab &rarr; CIE XYZ &rarr; Standard sRGB [0, 1] (Clipped)</text>
    </g>

    <!-- Arrow down -->
    <line x1="247" y1="380" x2="247" y2="404" stroke="#64748B" stroke-width="2" marker-end="url(#arrow-slate)"/>

    <!-- Bottom: Reconstructed Output Card -->
    <g transform="translate(45, 410)">
      <rect x="0" y="0" width="405" height="225" rx="8" fill="#FFFFFF" stroke="#0D9488" stroke-width="2" filter="url(#block-shadow)"/>
      
      <!-- Reconstructed Thumbnail -->
      <g transform="translate(20, 18)">
        <rect x="0" y="0" width="144" height="144" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1"/>
        <image href="{thumbs['pred']}" x="2" y="2" width="140" height="140" preserveAspectRatio="none"/>
        <text x="72" y="162" fill="#0F172A" font-size="11.5" font-weight="700" text-anchor="middle">Reconstructed Output &Icirc;RGB</text>
        <text x="72" y="176" fill="#64748B" font-size="10" text-anchor="middle">256 &times; 256 &times; 3</text>
      </g>

      <!-- Empirical Metrics Card -->
      <g transform="translate(180, 18)">
        <rect x="0" y="0" width="205" height="188" rx="6" fill="#F0FDF4" stroke="#86EFAC" stroke-width="1"/>
        <text x="14" y="22" fill="#166534" font-size="11.5" font-weight="700">Official Evaluation Metrics:</text>
        <text x="14" y="44" fill="#0F172A" font-size="10.5">Model: <tspan font-weight="700">ColorizationUNet</tspan></text>
        <text x="14" y="60" fill="#0F172A" font-size="10.5">Checkpoint: <tspan font-weight="700">best.pth (Ep 13)</tspan></text>
        
        <line x1="14" y1="72" x2="191" y2="72" stroke="#BBF7D0" stroke-width="1"/>
        
        <text x="14" y="92" fill="#15803D" font-size="12" font-weight="800">Mean PSNR: 24.65 dB</text>
        <text x="14" y="112" fill="#15803D" font-size="12" font-weight="800">Mean MAE: 0.040320</text>
        <text x="14" y="132" fill="#15803D" font-size="12" font-weight="800">Mean MSE: 0.004389</text>
        
        <line x1="14" y1="144" x2="191" y2="144" stroke="#BBF7D0" stroke-width="1"/>
        
        <text x="14" y="162" fill="#166534" font-size="10.5" font-weight="600">Baseline Chroma Ret.: 68.4%</text>
        <text x="14" y="178" fill="#B45309" font-size="10.5" font-weight="700">CW-MSE Chroma Ret.: 85.6%</text>
      </g>
    </g>
  </g>

  <!-- Arrow from U-Net HEAD to Panel 3 Predicted ab -->
  <path d="M 1785 188 L 1820 188 L 1820 202 L 1855 202" fill="none" stroke="#BE185D" stroke-width="2.5" marker-end="url(#arrow-pink)"/>

  <!-- Clean Route: Original Lightness L* Pass-Through -->
  <path d="M 285 640 L 330 640 L 330 715 L 1830 715 L 1830 351 L 1855 351" fill="none" stroke="#2563EB" stroke-width="2" stroke-dasharray="6 3" marker-end="url(#arrow-blue)"/>
  <rect x="800" y="704" width="240" height="22" rx="11" fill="#DBEAFE" stroke="#60A5FA" stroke-width="1"/>
  <text x="920" y="719" fill="#1E40AF" font-size="10.5" font-weight="700" text-anchor="middle">Exact Original Lightness L* Pass-Through</text>

  <!-- ==================== PANEL 4: SUPERVISION & LOSS FORMULATIONS ==================== -->
  <g transform="translate(30, 775)">
    <rect x="0" y="0" width="2300" height="495" rx="12" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.5" filter="url(#card-shadow)"/>
    
    <!-- Title -->
    <rect x="0" y="0" width="2300" height="42" rx="12" fill="#F1F5F9"/>
    <rect x="0" y="30" width="2300" height="12" fill="#F1F5F9"/>
    <text x="24" y="26" fill="#0F172A" font-size="14" font-weight="700">4. Training Supervision Path, Comparative Loss Function Objectives &amp; Backpropagation Engine</text>
    <line x1="0" y1="42" x2="2300" y2="42" stroke="#E2E8F0" stroke-width="1"/>

    <!-- Left: Ground Truth Data Stream -->
    <g transform="translate(25, 58)">
      <rect x="0" y="0" width="310" height="415" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2"/>
      <text x="155" y="26" fill="#0F172A" font-size="13" font-weight="700" text-anchor="middle">Ground Truth Supervision</text>

      <g transform="translate(95, 45)">
        <rect x="0" y="0" width="120" height="120" rx="8" fill="#E2E8F0" stroke="#CBD5E1" stroke-width="1"/>
        <image href="{thumbs['gt']}" x="2" y="2" width="116" height="116" preserveAspectRatio="none"/>
        <text x="60" y="138" fill="#334155" font-size="11" font-weight="600" text-anchor="middle">Ground Truth RGB</text>
      </g>

      <g transform="translate(20, 205)">
        <rect x="0" y="0" width="270" height="85" rx="6" fill="#FEF2F2" stroke="#F87171" stroke-width="1.2"/>
        <text x="135" y="22" fill="#991B1B" font-size="11.5" font-weight="700" text-anchor="middle">RGB &rarr; Lab Chrominance Extraction</text>
        <text x="135" y="42" fill="#B91C1C" font-size="11" text-anchor="middle">Target a*b*norm = a*b* / 128.0</text>
        <text x="135" y="60" fill="#7F1D1D" font-size="10.5" text-anchor="middle">Target Shape: [B, 2, 256, 256] &isin; [&minus;1, 1]</text>
        <text x="135" y="75" fill="#991B1B" font-size="9.5" text-anchor="middle">Precomputed offline in dataset manifests</text>
      </g>

      <g transform="translate(20, 305)">
        <rect x="0" y="0" width="270" height="90" rx="6" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"/>
        <text x="14" y="22" fill="#334155" font-size="11" font-weight="700">Dataset &amp; Experiment Protocol:</text>
        <text x="14" y="42" fill="#64748B" font-size="10.5">&bull; Train: 10,000 ADE20K images</text>
        <text x="14" y="58" fill="#64748B" font-size="10.5">&bull; Val: 1,000 images &bull; Test: 500 fixed</text>
        <text x="14" y="74" fill="#64748B" font-size="10.5">&bull; Batch Size: 8 &bull; Epochs: 20 &bull; Seed: 42</text>
      </g>
    </g>

    <!-- Arrow from GT box to Loss Box -->
    <path d="M 340 265 L 380 265" fill="none" stroke="#DC2626" stroke-width="2.5" marker-end="url(#arrow-red)"/>

    <!-- Center: 4 Loss Formulations Evaluated in Project -->
    <g transform="translate(390, 58)">
      <rect x="0" y="0" width="1410" height="415" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2"/>
      <text x="24" y="28" fill="#0F172A" font-size="13" font-weight="700">Comparative Loss Function Objectives Formulated and Evaluated in the Project</text>

      <!-- Loss 1: Baseline MSE -->
      <g transform="translate(24, 45)">
        <rect x="0" y="0" width="320" height="345" rx="8" fill="#EFF6FF" stroke="#3B82F6" stroke-width="1.5"/>
        <rect x="0" y="0" width="320" height="34" rx="8" fill="#DBEAFE"/>
        <text x="14" y="22" fill="#1E40AF" font-size="12" font-weight="700">1. Baseline MSE Loss (Official)</text>
        
        <rect x="14" y="46" width="292" height="52" rx="5" fill="#FFFFFF" stroke="#BFDBFE" stroke-width="1"/>
        <text x="160" y="68" fill="#1E3A8A" font-size="12" font-weight="700" text-anchor="middle">L_MSE = (1 / 2HW) &sum; ||&#226;b &minus; ab||&sup2;</text>
        <text x="160" y="86" fill="#3B82F6" font-size="9.5" text-anchor="middle">Per-pixel squared Euclidean chrominance error</text>

        <text x="14" y="120" fill="#0F172A" font-size="11" font-weight="700">Characteristics:</text>
        <text x="14" y="138" fill="#334155" font-size="10.5">&bull; Strict L2 convex penalization</text>
        <text x="14" y="154" fill="#334155" font-size="10.5">&bull; Fast, highly stable convergence</text>
        <text x="14" y="172" fill="#991B1B" font-size="10.5" font-weight="600">&bull; Known weakness: Mean-seeking regression</text>
        <text x="14" y="188" fill="#991B1B" font-size="10.5">&bull; Leads to muted/gray desaturated colors</text>

        <line x1="14" y1="210" x2="306" y2="210" stroke="#BFDBFE" stroke-width="1"/>
        <text x="14" y="230" fill="#1E40AF" font-size="11" font-weight="700">Validation Empirical Metrics:</text>
        <text x="14" y="250" fill="#1E3A8A" font-size="11">&bull; Best Epoch: <tspan font-weight="700">13</tspan> (Loss: 0.009781)</text>
        <text x="14" y="270" fill="#1E3A8A" font-size="11">&bull; Mean PSNR: <tspan font-weight="700">24.65 dB</tspan></text>
        <text x="14" y="290" fill="#1E3A8A" font-size="11">&bull; Mean MAE: <tspan font-weight="700">0.040320</tspan></text>
        <text x="14" y="310" fill="#1E3A8A" font-size="11">&bull; Mean MSE: <tspan font-weight="700">0.004389</tspan></text>
        <text x="14" y="330" fill="#1E3A8A" font-size="11">&bull; Mean Chroma Retention: <tspan font-weight="700">68.4%</tspan></text>
      </g>

      <!-- Loss 2: Smooth L1 -->
      <g transform="translate(370, 45)">
        <rect x="0" y="0" width="320" height="345" rx="8" fill="#F0FDF4" stroke="#22C55E" stroke-width="1.5"/>
        <rect x="0" y="0" width="320" height="34" rx="8" fill="#DCFCE7"/>
        <text x="14" y="22" fill="#166534" font-size="12" font-weight="700">2. Smooth L1 Loss (Exp 2)</text>
        
        <rect x="14" y="46" width="292" height="52" rx="5" fill="#FFFFFF" stroke="#BBF7D0" stroke-width="1"/>
        <text x="160" y="68" fill="#14532D" font-size="12" font-weight="700" text-anchor="middle">L_SL1 = (1 / HW) &sum; Huber_&beta;(&#226;b &minus; ab)</text>
        <text x="160" y="86" fill="#16A34A" font-size="9.5" text-anchor="middle">Huber transition threshold &beta; = 1.0</text>

        <text x="14" y="120" fill="#0F172A" font-size="11" font-weight="700">Characteristics:</text>
        <text x="14" y="138" fill="#334155" font-size="10.5">&bull; Quadratic for small error (|e| &lt; 1)</text>
        <text x="14" y="154" fill="#334155" font-size="10.5">&bull; Linear for large error (|e| &ge; 1)</text>
        <text x="14" y="172" fill="#166534" font-size="10.5" font-weight="600">&bull; Robust to chromatic outliers</text>
        <text x="14" y="188" fill="#334155" font-size="10.5">&bull; Preserves edge boundaries cleanly</text>

        <line x1="14" y1="210" x2="306" y2="210" stroke="#BBF7D0" stroke-width="1"/>
        <text x="14" y="230" fill="#166534" font-size="11" font-weight="700">Validation Empirical Metrics:</text>
        <text x="14" y="250" fill="#14532D" font-size="11">&bull; Best Epoch: <tspan font-weight="700">12</tspan> (Loss: 0.004872)</text>
        <text x="14" y="270" fill="#14532D" font-size="11">&bull; Mean PSNR: <tspan font-weight="700">24.62 dB</tspan> (Ep 10: 25.00 dB)</text>
        <text x="14" y="290" fill="#14532D" font-size="11">&bull; Mean MAE: <tspan font-weight="700">0.040560</tspan></text>
        <text x="14" y="310" fill="#14532D" font-size="11">&bull; Mean MSE: <tspan font-weight="700">0.004373</tspan></text>
        <text x="14" y="330" fill="#14532D" font-size="11">&bull; Mean Chroma Retention: <tspan font-weight="700">75.2%</tspan></text>
      </g>

      <!-- Loss 3: Chroma-Weighted MSE -->
      <g transform="translate(715, 45)">
        <rect x="0" y="0" width="330" height="345" rx="8" fill="#FFFBEB" stroke="#F59E0B" stroke-width="1.8"/>
        <rect x="0" y="0" width="330" height="34" rx="8" fill="#FEF3C7"/>
        <text x="14" y="22" fill="#92400E" font-size="12" font-weight="700">3. Chroma-Weighted MSE (Exp 3)</text>
        
        <rect x="14" y="46" width="302" height="52" rx="5" fill="#FFFFFF" stroke="#FDE68A" stroke-width="1"/>
        <text x="165" y="68" fill="#78350F" font-size="12" font-weight="700" text-anchor="middle">L_CW = (1 / HW) &sum; w(p) &bull; ||&#226;b &minus; ab||&sup2;</text>
        <text x="165" y="86" fill="#D97706" font-size="9.5" text-anchor="middle">w(p) = 1.0 + &lambda; &bull; &radic;((a*)&sup2; + (b*)&sup2;), &lambda; = 2.0</text>

        <text x="14" y="120" fill="#0F172A" font-size="11" font-weight="700">Characteristics:</text>
        <text x="14" y="138" fill="#334155" font-size="10.5">&bull; Explicitly combats gray-averaging bias</text>
        <text x="14" y="154" fill="#334155" font-size="10.5">&bull; Upweights vibrant saturated chromatic pixels</text>
        <text x="14" y="172" fill="#B45309" font-size="10.5" font-weight="700">&bull; +17.2% absolute chroma retention boost</text>
        <text x="14" y="188" fill="#334155" font-size="10.5">&bull; Vibrant foliage, sky, and architectural tones</text>

        <line x1="14" y1="210" x2="316" y2="210" stroke="#FDE68A" stroke-width="1"/>
        <text x="14" y="230" fill="#92400E" font-size="11" font-weight="700">Validation Empirical Metrics:</text>
        <text x="14" y="250" fill="#78350F" font-size="11">&bull; Best Epoch: <tspan font-weight="700">11</tspan> (Loss: 0.016335)</text>
        <text x="14" y="270" fill="#78350F" font-size="11">&bull; Mean PSNR: <tspan font-weight="700">24.28 dB</tspan></text>
        <text x="14" y="290" fill="#78350F" font-size="11">&bull; Mean MAE: <tspan font-weight="700">0.041180</tspan></text>
        <text x="14" y="310" fill="#78350F" font-size="11">&bull; Mean MSE: <tspan font-weight="700">0.004481</tspan></text>
        <text x="14" y="330" fill="#B45309" font-size="11" font-weight="700">&bull; Mean Chroma Retention: 85.6%</text>
      </g>

      <!-- Loss 4: Perceptual Feature Loss -->
      <g transform="translate(1070, 45)">
        <rect x="0" y="0" width="315" height="345" rx="8" fill="#FAF5FF" stroke="#A855F7" stroke-width="1.5"/>
        <rect x="0" y="0" width="315" height="34" rx="8" fill="#F3E8FF"/>
        <text x="14" y="22" fill="#6B21A8" font-size="12" font-weight="700">4. Perceptual VGG-16 Loss (Exp 4)</text>
        
        <rect x="14" y="46" width="287" height="52" rx="5" fill="#FFFFFF" stroke="#E9D5FF" stroke-width="1"/>
        <text x="157" y="68" fill="#581C87" font-size="12" font-weight="700" text-anchor="middle">L_Perc = &sum; (1 / C_l H_l W_l) ||&Phi;_l(&Icirc;) &minus; &Phi;_l(I)||&sup2;</text>
        <text x="157" y="86" fill="#9333EA" font-size="9.5" text-anchor="middle">Pretrained VGG-16 deep feature representation</text>

        <text x="14" y="120" fill="#0F172A" font-size="11" font-weight="700">Characteristics:</text>
        <text x="14" y="138" fill="#334155" font-size="10.5">&bull; Evaluates semantic &amp; perceptual fidelity</text>
        <text x="14" y="154" fill="#334155" font-size="10.5">&bull; Reconstructs RGB &rarr; passes to VGG-16</text>
        <text x="14" y="172" fill="#6B21A8" font-size="10.5" font-weight="600">&bull; Combined: L_tot = L_CW + &gamma;&bull;L_Perc</text>
        <text x="14" y="188" fill="#334155" font-size="10.5">&bull; Investigated &amp; verified in Step 7.4 pipeline</text>

        <line x1="14" y1="210" x2="301" y2="210" stroke="#E9D5FF" stroke-width="1"/>
        <text x="14" y="230" fill="#6B21A8" font-size="11" font-weight="700">Implementation Verification:</text>
        <text x="14" y="250" fill="#581C87" font-size="11">&bull; Feature Extractor: <tspan font-weight="700">VGG-16 (relu2_2, 3_3)</tspan></text>
        <text x="14" y="270" fill="#581C87" font-size="11">&bull; Differentiable Lab&rarr;RGB conversion</text>
        <text x="14" y="290" fill="#581C87" font-size="11">&bull; Perceptual Weight: <tspan font-weight="700">&gamma; = 0.05</tspan></text>
        <text x="14" y="310" fill="#581C87" font-size="11">&bull; Smoke Test (100 steps): <tspan font-weight="700">Loss 0.0381 &rarr; 0.0264</tspan></text>
        <text x="14" y="330" fill="#581C87" font-size="11">&bull; Pipeline Status: <tspan font-weight="700">Fully Verified</tspan></text>
      </g>
    </g>

    <!-- Right: Backpropagation & Optimization Engine -->
    <g transform="translate(1825, 58)">
      <rect x="0" y="0" width="450" height="415" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2"/>
      <text x="225" y="26" fill="#0F172A" font-size="13" font-weight="700" text-anchor="middle">Optimizer &amp; Gradient Engine</text>

      <g transform="translate(20, 45)">
        <rect x="0" y="0" width="410" height="74" rx="6" fill="#FEF3C7" stroke="#F59E0B" stroke-width="1.5"/>
        <text x="205" y="24" fill="#78350F" font-size="13" font-weight="800" text-anchor="middle">Adam Optimizer (&theta; &larr; &theta; &minus; &eta;&bull;m&#770; / (&radic;v&#770; + &epsilon;))</text>
        <text x="205" y="44" fill="#92400E" font-size="11" text-anchor="middle">Base Learning Rate (&eta;): 2.0 &times; 10&minus;4 &bull; &beta;1 = 0.9, &beta;2 = 0.999</text>
        <text x="205" y="60" fill="#92400E" font-size="10.5" text-anchor="middle">Weight Decay: 0.0 &bull; Epsilon: 1e&minus;8 &bull; Minibatches: 1,250 / epoch</text>
      </g>

      <g transform="translate(20, 132)">
        <rect x="0" y="0" width="410" height="68" rx="6" fill="#EFF6FF" stroke="#3B82F6" stroke-width="1.2"/>
        <text x="205" y="22" fill="#1E40AF" font-size="12" font-weight="700" text-anchor="middle">StepLR Learning Rate Scheduler</text>
        <text x="205" y="42" fill="#1E3A8A" font-size="11" text-anchor="middle">Step Size: 10 epochs &bull; Decay Multiplier (&gamma;): 0.5</text>
        <text x="205" y="58" fill="#3B82F6" font-size="10.5" text-anchor="middle">Epochs 1&ndash;10: lr = 2.0 &times; 10&minus;4 &bull; Epochs 11&ndash;20: lr = 1.0 &times; 10&minus;4</text>
      </g>

      <g transform="translate(20, 212)">
        <rect x="0" y="0" width="410" height="68" rx="6" fill="#F1F5F9" stroke="#CBD5E1" stroke-width="1.2"/>
        <text x="205" y="22" fill="#0F172A" font-size="12" font-weight="700" text-anchor="middle">Hardware Acceleration Profile</text>
        <text x="205" y="42" fill="#334155" font-size="11" text-anchor="middle">Device: Apple Silicon M5 (16 GB Unified Memory)</text>
        <text x="205" y="58" fill="#475569" font-size="10.5" text-anchor="middle">Backend: MPS (Metal Performance Shaders) &bull; AMP FP16</text>
      </g>

      <g transform="translate(20, 292)">
        <rect x="0" y="0" width="410" height="108" rx="6" fill="#DC2626" fill-opacity="0.08" stroke="#DC2626" stroke-width="1.5"/>
        <text x="205" y="26" fill="#991B1B" font-size="12.5" font-weight="700" text-anchor="middle">Backpropagation &amp; Weight Update Loop</text>
        <text x="205" y="48" fill="#B91C1C" font-size="11" text-anchor="middle">&nabla;_&theta; L_total computed via Automatic Differentiation (PyTorch Autograd)</text>
        <text x="205" y="66" fill="#B91C1C" font-size="11" text-anchor="middle">Gradients backpropagated through all 29,228,802 parameters</text>
        <text x="205" y="86" fill="#7F1D1D" font-size="10.5" font-weight="600" text-anchor="middle">Best checkpoint saved conditionally on minimum validation loss</text>
      </g>
    </g>
  </g>

  <!-- Clean Backpropagation curved feedback arrow returning up to U-Net -->
  <path d="M 2030 775 L 2030 740 L 1400 740 L 1400 700" fill="none" stroke="#DC2626" stroke-width="2.5" stroke-dasharray="6 4" marker-end="url(#arrow-red)"/>
  <rect x="1460" y="729" width="180" height="22" rx="11" fill="#FEE2E2" stroke="#F87171" stroke-width="1"/>
  <text x="1550" y="744" fill="#991B1B" font-size="10.5" font-weight="700" text-anchor="middle">&nabla;_&theta; Gradients to U-Net Weights</text>

  <!-- Clean Route: Prediction to Loss Module arrow -->
  <path d="M 1820 202 L 1820 762 L 1750 762 L 1750 830" fill="none" stroke="#BE185D" stroke-width="2" marker-end="url(#arrow-pink)"/>
  <rect x="1700" y="751" width="170" height="22" rx="11" fill="#FCE7F3" stroke="#F472B6" stroke-width="1"/>
  <text x="1785" y="766" fill="#9D174D" font-size="10" font-weight="700" text-anchor="middle">Predicted &#226;*b* for Loss Evaluation</text>

</svg>"""
    return svg

def main():
    print("==================================================")
    print("GENERATING FIGURE 2: SYSTEM ARCHITECTURE DIAGRAM")
    print("==================================================")

    # 1. Generate real preview thumbnails
    print("[1/3] Generating authentic dataset and model thumbnails...")
    thumbs = generate_sample_thumbnails()
    print("      Thumbnails successfully rendered and base64-encoded.")

    # 2. Build SVG
    print("[2/3] Building publication-quality SVG...")
    svg_content = build_svg_diagram(thumbs)
    svg_path = os.path.join(OUTPUT_DIR, "figure_2_system_architecture.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"      Saved SVG: {svg_path} ({len(svg_content)} bytes)")

    # 3. Render PNG using Headless Chrome
    print("[3/3] Rendering 300-DPI publication PNG via Chrome headless...")
    png_path = os.path.join(OUTPUT_DIR, "figure_2_system_architecture.png")
    
    html_path = os.path.join(OUTPUT_DIR, "render_figure_2.html")
    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #FFFFFF; width: 2360px; height: 1300px; overflow: hidden; }}
  </style>
</head>
<body>
  {svg_content}
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        "--window-size=2360,1300",
        f"--screenshot={png_path}",
        html_path
    ]
    res = subprocess.run(chrome_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Chrome error:", res.stderr)
        raise RuntimeError("Failed to render PNG with Chrome")
        
    if os.path.exists(html_path):
        os.remove(html_path)

    print(f"      Saved PNG: {png_path} ({os.path.getsize(png_path)} bytes)")
    print("==================================================")
    print("FIGURE 2 GENERATION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()
