# Perceptual Metric Reproduction & Failure Analysis on KADID-10k: report draft

> **DRAFT (B-26, P-010).** The structure follows the six required sections of the spec (`doc/references/HW3.pdf` §6).
> Every number is reproduced by `notebooks/02`–`05` from the committed `results/`.
> The perceptual explanations in §4–§5 started as AI hypotheses (AGENTS.md). Lines marked *(verified)* were checked numerically.
> When @ALLEN07056 approves the PR, the team takes ownership of the text (D-26). Before the PDF: cut to ≤ 4 pages (B-19) and convert (D-08).

## Background: Image Quality Metrics, Explained Simply (ELI5)

> From the team's notes (2026-11-02), kept in their original form. Topics: PSNR, SSIM, LPIPS, DISTS, pooling, MS-SSIM, the KADID-10k benchmark, SRCC/PLCC, and backbones. How each idea maps onto our setup is in the table at the end of this section. Condense into §1 for the PDF (B-19).

---

### 1. PSNR (Peak Signal-to-Noise Ratio)

A score for how close a damaged or compressed picture is to the original.

**ELI5:** Photocopy a drawing and the copy comes out a bit fuzzy. PSNR is a number saying how fuzzy the copy is compared to the original.

- **Higher = more like the original**
- **Lower = messier**

**Steps:**
1. Compare original and copy pixel by pixel.
2. Average the squared differences (this is the MSE, mean squared error).
3. Convert to a log scale in decibels:

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$

MAX is the brightest possible pixel value (255 for 8-bit images).

**Rules of thumb:**
- 30-50 dB: good quality (above ~40 dB is very hard to tell from the original)
- 20-30 dB: noticeable loss
- Below 20 dB: clearly degraded
- Infinite: identical images

**Catch:** It only counts raw pixel differences, not what humans notice. A tiny shift of the whole image hurts PSNR a lot but looks fine.

---

### 2. SSIM (Structural Similarity Index)

Scores how similar two images *look*, based on how the eye judges them rather than raw pixel differences.

**ELI5:** PSNR asks "how different is each dot?" SSIM asks "does this still look like the original?" It checks whether patterns, shapes, and edges survived.

**Compared in small patches:**
1. **Luminance:** is the patch equally bright?
2. **Contrast:** is the light/dark difference the same?
3. **Structure:** do patterns and edges line up? (the key part)

These are multiplied into one score per patch, then averaged over the image.

$$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

μ = average brightness, σ² = variance (contrast), σ_xy = how the two patches vary together (structure).

**Reading it:** 1.0 = identical; ~0.9+ = very similar; ~0.5-0.8 = visibly different; near 0 = no structural similarity.

**Why better than PSNR:** Brighten a photo slightly and PSNR may drop a lot while SSIM stays high. Blur the edges and PSNR may barely change while SSIM drops.

**Limits:** Not a true model of human vision. Fooled by shifts, rotations, and heavy distortions.

**Is SSIM hand-crafted?** Yes. A researcher designed the formula by hand from assumptions about vision. Nothing is learned from data. Only small constants (c1, c2) and the window size (e.g. 11x11 Gaussian) are chosen by hand.

---

### 3. LPIPS (Learned Perceptual Image Patch Similarity)

A neural network judges how different two images *look*, trained to agree with human judgments.

**ELI5:** PSNR counts differing dots, SSIM checks whether patterns survived, LPIPS asks a trained "robot eye" whether two pictures *feel* the same. That robot eye learned from thousands of human "which is closer to the original?" judgments.

**Steps:**
1. Feed both images into a pretrained deep network (VGG, AlexNet, SqueezeNet).
2. Grab internal features from several layers (early = edges/textures, deep = shapes/objects).
3. Compare features layer by layer using learned weights.
4. Average into one number.

**Reading it:** 0 = identical; ~0.1 = very similar; ~0.5+ = clearly different. **Lower is better** (the reverse of PSNR/SSIM).

**Why useful:** Tolerates things humans ignore (tiny shifts, similar-looking textures) and punishes things humans notice (blur, odd artifacts, wrong textures). Standard for generated images, super-resolution, and style transfer.

**Limits:** Depends on the backbone, so scores aren't comparable across backbones. Can be fooled by adversarial noise. Only an approximation of human perception.

**Is it deep-learning based?** Yes, with two learned parts:
- **Backbone:** trained for ImageNet classification, not quality assessment, yet its features match human perception surprisingly well.
- **Linear calibration layers:** trained on human similarity judgments (BAPPS dataset). The paper also found that even untouched, uncalibrated features beat SSIM and PSNR.

---

### 4. DISTS (Deep Image Structure and Texture Similarity)

A learned metric that treats **structure** and **texture** differently. Sits between SSIM and LPIPS: deep features like LPIPS, SSIM-style comparison.

**ELI5:** Two photos of the same grassy field have every blade in a different spot, so a pixel comparison says "totally different!", but your eye says "same grass." DISTS agrees with your eye: layout matters (structure), exact placement of repeating texture doesn't.

**Steps:**
1. Run both images through a pretrained VGG, with max-pooling replaced by L2 pooling (so small shifts matter less).
2. At each layer and channel, compute:
   - **Texture term:** compare the *means* of the feature maps (overall texture statistics, ignoring position)
   - **Structure term:** compare the *correlation* between feature maps (spatial layout)
3. Combine with learned weights across layers.

Think of it as **"SSIM on deep features, with texture tolerance."**

**Reading it:** 0 = identical; higher = more different. **Lower is better.**

**Why useful:** Tolerant of texture resampling (grass, bricks, fur, water). Good for GAN outputs and super-resolution. Agrees well with human ratings.

**Limits:** Depends on the VGG backbone, can be fooled by adversarial changes, and its learned weights may not generalize to every image type.

---

### 5. Pooling: max-pooling vs L2 pooling

**Pooling** shrinks a grid of numbers by summarizing small windows (say 2x2) into one number each.

**Max-pooling: "Who's the loudest?"** Keep the biggest number, discard the rest.

```
[1  3]
[2  9]   ->  9
```

- Keeps the strongest signal (e.g. "an edge is somewhere here")
- Loses *where* in the window it was
- Jumpy: a 1-pixel shift can change which value is the max

**L2 pooling: "What's the overall energy?"** Square every number, average, take the square root (root-mean-square).

```
[1  3]
[2  9]   ->  sqrt((1^2 + 3^2 + 2^2 + 9^2) / 4) = sqrt(95/4) ~ 4.9
```

- Every value contributes, louder ones count more
- Smoother and more shift-tolerant

| | Max-pooling | L2 pooling |
|---|---|---|
| Rule | Keep the biggest value | Root of the average of squares |
| Uses all values? | No | Yes |
| Shift sensitivity | Higher (jumpy) | Lower (smooth) |

DISTS swaps in L2 pooling so the metric doesn't change sharply from tiny shifts that humans wouldn't notice.

---

### 6. MS-SSIM (Multi-Scale SSIM)

SSIM applied at several zoom levels, then combined into one score.

**ELI5:** A painting looks different from 2 inches away (brushstrokes) than from across the room (overall shapes). SSIM looks from one distance only. MS-SSIM looks from several and combines them.

**Steps:**
1. Start with both images at full size.
2. Compute contrast and structure comparisons (brightness only at the coarsest scale).
3. Blur and downsample both by 2, then repeat, typically for 5 scales.
4. Multiply the per-scale scores, each raised to a weight found from experiments with human viewers (middle scales matter most).

$$\text{MS-SSIM} = \left[l_M\right]^{\alpha_M} \cdot \prod_{j=1}^{M} \left[c_j\right]^{\beta_j} \left[s_j\right]^{\gamma_j}$$

**Reading it:** 1.0 = identical; closer to 0 = more different. Higher is better.

**Why useful:** Adapts to viewing distance and resolution, matches human ratings better than single-scale SSIM, still hand-crafted so fast and deterministic. Common in compression research and as a training loss.

**Limits:** Needs images large enough to downsample several times. Still weak on geometric shifts and heavy generative distortions.

---

### 7. KADID-10k benchmark

A dataset for testing how well quality metrics agree with human opinion.

**ELI5:** Take 81 clean photos and wreck each one in 25 ways at 5 strengths, giving 10,125 damaged images. People rated how bad each looks (a mean opinion score, MOS). A good metric ranks images the same way humans do.

Distortion groups include blur, noise, color changes, compression (JPEG, JPEG2000), brightness changes, spatial distortions, and sharpness/contrast changes.

**Results (SRCC / PLCC, from the MR-IQA paper's Table 2):**

| Metric | SRCC | PLCC |
|---|---|---|
| PSNR | 0.676 | 0.675 |
| SSIM | 0.724 | 0.717 |
| MS-SSIM | 0.826 | 0.820 |
| LPIPS | 0.843 | 0.839 |
| DISTS | 0.887 | 0.886 |

Ordering: PSNR < SSIM < MS-SSIM < LPIPS < DISTS, tracking each metric's perceptual awareness.

**Caveats:**
- Other papers report slightly different numbers (e.g. SSIM 0.783 SRCC and MS-SSIM 0.838 in JND-SalCAR; PIQ reports ~0.72 for SSIM and ~0.88 for DISTS). Differences come from implementation, color vs. grayscale, and resizing.
- LPIPS values depend on the backbone.
- KADID-10k is mostly traditional distortions (blur, noise, compression), not generative-model artifacts, where DISTS and LPIPS tend to shine more.
- Even state-of-the-art full-reference metrics are far from perfect on KADID-10k.

---

### 8. SRCC, PLCC, and backbone

**PLCC (Pearson Linear Correlation Coefficient)**
Plot each image as a dot (human score vs. metric score). PLCC asks whether the dots form a straight line. 1.0 = perfect line, 0 = random cloud, -1.0 = perfectly opposite. It cares about actual values, so a curved relationship lowers it (papers often fit a curve first).

**SRCC (Spearman Rank-order Correlation Coefficient)**
Line up images best to worst by humans and by the metric, then ask whether the lineups match. 1.0 = identical ranking, 0 = unrelated, -1.0 = reversed. It cares only about order, so it fairly compares metrics with different scales. "Lower is better" metrics (LPIPS, DISTS) give negative values, so the absolute value is usually reported.

| | Asks | Sensitive to |
|---|---|---|
| SRCC | "Same ranking?" | Order only |
| PLCC | "Same straight-line relationship?" | Actual values |

High SRCC with lower PLCC means the metric ranks correctly but its scale is warped relative to human scores.

**Backbone**
The foundation and frame of a house: you don't rebuild it, you add rooms on top. In deep learning, it's a big pretrained network (VGG, AlexNet) that already learned to see by training on millions of images (ImageNet), turning an image into features (edges, textures, shapes, objects).

For LPIPS and DISTS, the backbone extracts features and a small learned part on top turns feature differences into a score.
- Different backbones give different scores ("LPIPS-VGG" is not directly comparable to "LPIPS-AlexNet").
- Bigger backbones are usually more accurate but slower.
- The metric inherits the backbone's strengths and blind spots.

---

### Summary comparison

| Metric | Type | Key idea | Scales | Better |
|---|---|---|---|---|
| PSNR | Hand-crafted | Pixel error | Pixel level | Higher |
| SSIM | Hand-crafted | Brightness, contrast, structure | Single | Higher |
| MS-SSIM | Hand-crafted | SSIM across zoom levels | Multiple | Higher |
| LPIPS | Learned | Deep feature distance, calibrated on human judgments | Multiple (deep layers) | Lower |
| DISTS | Learned | Structure + texture on deep features | Multiple (deep layers) | Lower |

---

### How this maps onto our project

| Metric | In our setup | Our KADID-10k SRCC (literature, Table in §7) | Where it shows up |
|---|---|---|---|
| PSNR | RGB; identical images capped at 80 dB (D-18) | 0.676 (0.676) | — |
| SSIM | pyiqa: **luma (Y) only**, no downsampling | 0.619 (0.724); **0.752** with Wang's downsampling (B-22) | B-1: color change invisible; C-1: relative luminance term over-penalises dark regions |
| MS-SSIM | luma only, weights 0.045 / 0.286 / 0.300 / 0.236 / 0.133 | 0.826 (0.826) | A-1: coarse scales catch large, smooth damage |
| LPIPS | **VGG16** v0.1 (D-23); AlexNet kept as supplementary | 0.720 VGG / 0.822 AlexNet (0.843) | A-2, B-2, C-1: spatial map and per-layer values |
| DISTS | VGG16; weights **trained on KADID-10k** (D-22), so in-domain | 0.814 (0.887); gap not explained yet (TODO) | A-2, B-2: per-stage texture/structure terms (no spatial map) |

SRCC/PLCC: instead of reporting |SRCC| for LPIPS/DISTS, we flip their sign before correlating (§2). PLCC is computed after the 5-parameter logistic fit.

**Sources:** [MR-IQA](https://arxiv.org/pdf/2302.13770) · [Full-reference IQA on KADID-10k](https://arxiv.org/pdf/1907.02096) · [JND-SalCAR](https://arxiv.org/pdf/1902.05316) · [PIQ documentation](https://piq.readthedocs.io/en/latest/overview.html)

## 1. Metrics and Dataset

**Dataset: KADID-10k** (Lin et al., QoMEX 2019).
- 81 pristine reference images (512 × 384), each distorted by **25 distortion types × 5 levels**, giving **10,125** distorted images.
- The types fall into 7 groups: blur (3), color (5), compression (2), noise (5), brightness change (3), spatial (5), sharpness and contrast (2).
- Subjective scores come from crowdsourced degradation-category ratings (about 30 per image), reported on a 1–5 scale where **higher = better** (column `dmos`, which is a MOS despite its name). The per-image rater variance (`var`) is also provided.
- We use the **whole dataset**, so no subset sampling is needed. The data is gitignored; download instructions are in `README.md` (D-14).

**Metrics and their perceptual assumption.**

| metric | type | what it compares | perceptual assumption |
|---|---|---|---|
| PSNR | traditional (fidelity) | mean squared error of RGB pixels, $10\log_{10}(255^2/\text{MSE})$ | every pixel error is equally visible, regardless of where it is or what it looks like |
| SSIM (Wang et al. 2004) | traditional (structural) | local luminance, contrast and structure (correlation) in 11×11 Gaussian windows on luma | the visual system extracts *structure*: structural change hurts more than a uniform shift |
| MS-SSIM (Wang et al. 2003) | traditional (structural) | SSIM's contrast-structure term at 5 dyadic scales, luminance only at the coarsest | the same, but whether a change is visible depends on viewing distance and scale |
| LPIPS-VGG (Zhang et al. 2018) | learned (deep features) | channel-normalised VGG16 features at 5 layers, squared difference weighted by linear weights learned from human 2AFC judgments on 64×64 patches (BAPPS), averaged over space | deep features trained for recognition capture perceptual similarity; humans judge local appearance |
| DISTS (Ding et al. 2022) | learned (deep features) | VGG16 features: a *texture* term (global means per channel) and a *structure* term (global correlations), with weights learned on KADID-10k + texture data | quality = structure + texture similarity, so it tolerates texture *resampling* (same texture, different pixels) |

LPIPS-VGG and DISTS share the same VGG16 backbone, so differences between them come from the **formula**, not the features (D-23). LPIPS with an AlexNet backbone is kept as a supplementary column.
**Caveat (D-22, verified in Ding et al. §II-D):** DISTS's weights were trained on KADID-10k ("We chose the large-scale IQA dataset KADID-10k as the training set"), so DISTS is evaluated *in-domain* here, and its correlation is optimistic compared with the other metrics.

## 2. Experimental Setup

**Implementation.**
- Every metric comes from `pyiqa` 0.1.16 with its default, official weights (`psnr`, `ssim`, `ms_ssim`, `lpips-vgg` v0.1, `dists`, plus `lpips` (Alex)), running on torch 2.15 and Python 3.11. The environment is pinned in `uv.lock` (macOS + Linux).
- Images are loaded as RGB in [0, 1] at full resolution; there is no resizing or cropping.
- PSNR uses RGB (capped at 80 dB for identical images).
- SSIM and MS-SSIM use the YIQ luma channel ($Y = 0.299R + 0.587G + 0.114B$, ×255, rounded), computed in float64 without downsampling.
- LPIPS and DISTS receive RGB.
- Scoring all pairs took 40 min on an M2 Pro: deep metrics on MPS, traditional metrics on CPU (D-17).

**Correctness checks** (`tests/`, 24 tests):
- An identical pair gets the perfect score.
- PSNR matches `skimage` within 5e-5 dB.
- SSIM matches a textbook implementation (which itself matches `skimage` to 1e-6), apart from one documented difference: pyiqa clamps a negative contrast-structure term to 0 (D-20, effect quantified in §3).
- CPU and MPS agree within 1e-3 relative.
- The committed scores reproduce on a second machine.
- **Pixel-identical pairs:** 320 of KADID's 10,125 images are pixel-identical to their reference (e.g. mean shift level 3). They are kept for the correlations and removed for the failure mining (D-18). Excluding them changes SRCC by at most 0.04 (`results/tables/overall_excl_identical.csv`).

**Score direction.**

KADID's `dmos` column is a mean opinion score where **higher = better quality** (1 = bad, 5 = excellent).
Every metric is first turned into an *aligned* score $\tilde s_m$ that is also higher = better:

$$
\tilde s_m(i) = \begin{cases} s_m(i) & m \in \{\text{PSNR, SSIM, MS-SSIM}\}\ \text{(similarity: higher = better)}\\ -\,s_m(i) & m \in \{\text{LPIPS-VGG, DISTS}\}\ \text{(distance: lower = better)}\end{cases}
$$

**Claim: after alignment, every metric's direction agrees with the MOS direction.** Evidence:

| metric | native direction | SRCC(raw $s_m$, MOS) | SRCC($\tilde s_m$, MOS) | PLCC after logistic |
|---|---|---|---|---|
| PSNR | higher = better | +0.676 | +0.676 | 0.680 |
| SSIM | higher = better | +0.619 | +0.619 | 0.628 |
| MS-SSIM | higher = better | +0.826 | +0.826 | 0.823 |
| LPIPS-VGG | lower = better | **−0.720** | +0.720 | 0.729 |
| DISTS | lower = better | **−0.814** | +0.814 | 0.814 |

- The raw SRCC is negative exactly for the two distance metrics. That is the expected sign, so these columns must be flipped, and after flipping every correlation is positive. *(verified, `results/tables/overall.csv`)*
- Every metric also scores a pixel-identical pair as perfect: PSNR = 80 dB (the cap), SSIM = 1, LPIPS ≈ 0, DISTS ≈ 0.
- `pyiqa`'s own `lower_better` flag is asserted against our registry when each model is created. *(verified, `tests/test_sanity.py`, `tests/test_evaluate.py`)*
- PLCC is computed on $\tilde s_m$ after a *monotonically increasing* VQEG 5-parameter logistic $f_m$, so the fit cannot reverse the direction.


**Evaluation procedure and formulas.**

Let $N$ be the number of pairs and $y_i$ the MOS of pair $i$.

**Percentile rank** (used for every disagreement score, so dB, [0,1] similarities and feature distances become comparable):

$$
r_m(i) = \frac{\operatorname{rank}\big(\tilde s_m(i)\big)}{N} \in (0,1], \qquad r_{MOS}(i) = \frac{\operatorname{rank}(y_i)}{N}
$$

**SRCC** is Pearson's correlation between the two rank vectors. Without ties it equals

$$
\text{SRCC}_m = 1 - \frac{6\sum_i d_i^2}{N(N^2-1)}, \qquad d_i = \operatorname{rank}(\tilde s_m(i)) - \operatorname{rank}(y_i) = N\,\big(r_m(i) - r_{MOS}(i)\big)
$$

So the image with the largest $|r_m(i) - r_{MOS}(i)|$ costs the metric the most SRCC. That is the score used for case B-1.

**PLCC** is computed after the VQEG logistic, fitted on all 10,125 pairs with $z$-scored input:

$$
f_m(x) = \beta_1\Big(\tfrac12 - \tfrac{1}{1+e^{\beta_2 (x-\beta_3)}}\Big) + \beta_4 x + \beta_5, \qquad
\text{PLCC}_m = \frac{\operatorname{cov}\big(f_m(\tilde s_m), y\big)}{\sigma_{f_m(\tilde s_m)}\,\sigma_y}
$$

The residual $e_m(i) = f_m(\tilde s_m(i)) - y_i$ is in MOS units, so it can be compared across metrics.
The pairs with the largest $|e_m(i)|$ cost the metric the most PLCC. That is the score used for case B-2.

**Disagreement scores for the five cases** (each case takes the top-1; if an image was already used by an earlier case, the next candidate is taken):

| case | what disagrees | score | sign meaning |
|---|---|---|---|
| A-1 | SSIM vs MS-SSIM, same image | $g_{A}(i) = \lvert r_{SSIM}(i) - r_{MS\text{-}SSIM}(i)\rvert$ | – |
| A-2 | LPIPS-VGG vs DISTS, same image | $\lvert r_{LPIPS}(i) - r_{DISTS}(i)\rvert$ | – |
| B-1 | any metric vs MOS, rank view (SRCC) | $\lvert r_m(i) - r_{MOS}(i)\rvert$, maximised over $(m, i)$ | $>0$: metric too optimistic |
| B-2 | any metric vs MOS, value view (PLCC) | $\lvert e_m(i)\rvert$, maximised over $(m, i)$ | $>0$: metric too optimistic |
| C-1 | SSIM vs LPIPS-VGG *ranking* of two images $i, j$ of the same reference | $\min(\lvert\Delta r_{SSIM}\rvert, \lvert\Delta r_{LPIPS}\rvert)$ where $\Delta r = r^{ref}(i) - r^{ref}(j)$ and the two $\Delta r$ have opposite signs | humans side with whichever metric has the same sign as $\Delta MOS$; pairs with $\lvert\Delta MOS\rvert < 0.5$ are ties and are skipped |

$r^{ref}$ is the percentile rank *within one reference image*, so content difficulty cannot create a fake ranking conflict.
Full candidate lists: `results/failures/candidates_{A,B,C}.csv`. Selection: `results/failures/selected_report5.csv`.


**Failure-case selection** (D-24, D-25): quantitative first, then visual inspection (spec §7).
- Images whose rater variance is above the 90th percentile can't be selected, so no case is built on a MOS that the raters themselves disagree on.
- Each of the five cases takes the top-1 candidate by the score in the table above; the full candidate lists are committed.

## 3. Quantitative Results

| (all 10,125 pairs) | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS* | *(LPIPS-Alex)* |
|---|---|---|---|---|---|---|
| SRCC | 0.676 | 0.619 | **0.826** | 0.720 | 0.814 | *0.822* |
| PLCC (logistic) | 0.680 | 0.628 | **0.823** | 0.729 | 0.814 | *0.823* |
| KRCC | 0.488 | 0.447 | 0.635 | 0.531 | 0.625 | *0.630* |

\*DISTS was trained on KADID-10k (in-domain).

![scatter](../results/figures/quant/scatter.png)
*Raw metric (native direction) vs MOS, coloured by distortion group; black = the fitted logistic. The negative slope of LPIPS/DISTS is the expected lower-is-better direction.*

**SRCC per distortion group** (`results/tables/by_group.csv`):

| group | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS |
|---|---|---|---|---|---|
| blur | 0.888 | 0.911 | 0.950 | 0.954 | **0.965** |
| brightness | **0.900** | 0.856 | 0.879 | 0.867 | 0.892 |
| color | 0.683 | **0.310** | **0.743** | 0.484 | 0.651 |
| compression | 0.827 | 0.855 | **0.918** | 0.906 | 0.917 |
| noise | 0.820 | 0.768 | **0.858** | 0.825 | 0.838 |
| sharpness & contrast | 0.872 | 0.866 | **0.923** | 0.906 | 0.918 |
| spatial | 0.681 | **0.386** | **0.713** | 0.531 | 0.632 |

![SRCC by type](../results/figures/quant/srcc_by_type.png)
![SRCC by type and level](../results/figures/selected/srcc_type_level.png)
*SRCC inside each (distortion type, level) cell, 81 images per cell. x-axis: distortion level L1 (mildest) … L5 (strongest).*

**Interpretation (draft, B-10):**
1. **No metric wins everywhere.** MS-SSIM has the best overall SRCC (0.826) and wins 5 of 7 groups. DISTS wins blur and PSNR wins brightness. On blur, compression, and sharpness/contrast every metric is within 0.08 of the best.
2. **Within type vs across types.** Within a single distortion type, every metric ranks images well: the median SRCC per type is 0.83–0.88. The overall SRCC is much lower because a metric must also put *different* distortion types on one common scale. For example, SSIM reaches 0.88 within color diffusion but only 0.31 on the whole color group. LPIPS-VGG reaches 0.94 within jitter but 0.53 on spatial distortions *(verified)*. Most of the gap between metrics is therefore **cross-type calibration**, which is exactly where §4's failure cases come from.
3. **Color and spatial distortions are hard for everyone.** On color, the luma-only SSIM collapses to 0.31, while PSNR (RGB) and MS-SSIM (coarse scales) do much better. On spatial distortions (non-eccentricity patch, pixelate, jitter, color block), the pixel-aligned metrics fall to SRCC 0.29–0.48 on non-eccentricity patch and pixelate, while LPIPS and DISTS reach 0.65–0.78.
4. **Formula vs backbone.** LPIPS-VGG (0.720) falls behind DISTS (0.814) even though both use the same VGG16. Within types they are close, so DISTS's global texture/structure statistics, plus its training on KADID, calibrate better across types. Changing the LPIPS backbone to AlexNet gives 0.822 (`lpips_backbone.csv`), so the backbone matters as much as the formula.
5. **Logistic fit.** PLCC improves over raw PLCC mostly for PSNR (0.556 → 0.680) and MS-SSIM (0.680 → 0.823), whose relation to MOS is strongly non-linear (scatter); DISTS is almost linear (0.806 → 0.814).

**Monotonicity check (B-06; `results/tables/monotonicity_by_type.csv`).** For each (reference, type) we test whether the aligned score never improves as the level rises.
- Across all types the deep metrics are the most monotone: share LPIPS 0.90, DISTS 0.85, SSIM 0.83, MS-SSIM 0.82, PSNR 0.81. The mean Spearman(level, score) is between −0.89 and −0.92 for every metric.
- **MOS itself is monotone in only 62% of series**, because neighbouring levels often differ by less than the rating noise.
- Every metric and MOS score 0 on `mean_shift` and `contrast_change`. KADID's levels for these types go in both directions around the reference (mean-shift level 3 is the identity), so for them "level" is not a severity order.
- `pixelate` breaks the pixel-aligned metrics: PSNR/SSIM/MS-SSIM are monotone in ≤ 6% of series (the level-3 score is better than level 2), while LPIPS/DISTS are monotone in 100%. Block-grid alignment favours some block sizes at pixel level, but not perceptually.

**SSIM variants (B-22, D-20; `results/tables/ssim_variants.csv`).** Our SSIM (0.62) is low, so we changed one implementation choice at a time:

| SSIM variant | SRCC | PLCC | color group SRCC |
|---|---|---|---|
| pyiqa default (main): no downsampling, cs clamped | 0.619 | 0.628 | 0.310 |
| unclamped cs (textbook / skimage) | 0.619 | 0.628 | 0.310 |
| Wang's 2× downsampling (original MATLAB) | **0.752** | **0.748** | **0.704** |

- The cs clamp is irrelevant (the largest change on any image is 0.013).
- **Downsampling raises SRCC by 0.13**, almost entirely through the color group. Viewing scale therefore matters for SSIM: at 2× downsampling, the low-frequency luma side-effects of color distortions become visible to its 11×11 window. This agrees with case A-1 in §4.
- We keep the pyiqa default as the main score (D-26), because it is what the package computes for everyone, and report this sensitivity as a finding.

## 4. Failure Cases

**Reading the heatmaps:** the x-axis L1–L5 is the KADID-10k distortion level, from 1 (mildest) to 5 (strongest). The exceptions are mean shift and contrast change, whose levels go in both directions around the reference (mean-shift L3 = no change). Gray cells hold only pixel-identical pairs.


**Reading the case figures.** Each figure shows the reference, the distorted image, and **only the views of the metrics that disagree in that case**. Each view is computed from the metric itself, so its numbers add up to the metric's score (code: `src/humper/explain.py`):

| panel | what it shows | used in |
|---|---|---|
| \|ΔY\| luma / \|ΔIQ\| chroma | the YIQ split of the pixel change (0–255, one shared scale). SSIM and MS-SSIM only receive Y, so chroma change is invisible to them | A-1, B-1, C-1 |
| 1 − SSIM map | where SSIM's 11×11 window sees damage; its mean is 1 − SSIM | A-1, B-1, C-1 |
| 1 − MS-SSIM map | MS-SSIM's 5 per-scale maps, upsampled and combined with its weights (same scale as the SSIM map) | A-1 |
| LPIPS-VGG map | LPIPS's per-pixel distance; its mean is the LPIPS score | A-2, B-2, C-1 |
| VGG16 stage bars | LPIPS per layer, and DISTS per stage split into texture (global feature *means*) and structure (global feature *correlations*); each metric's bars sum to its score | A-2, B-2 |
| \|RGB diff\| | where the pixels changed (what PSNR averages) | B-2 |

DISTS has no spatial map, because it compares whole-image statistics; the stage bars are its honest equivalent. A-1 and B-1 also have a scale-by-scale MS-SSIM figure, and every case has a type × level heatmap: the left panel is the case's signed score averaged per cell over all references, the right panel is the selected image's reference, and the black box marks the selected image.

### 4.1 A-1: SSIM vs MS-SSIM: `I18_04_04` (color diffusion, level 4)

![A-1](../results/figures/selected/A-1_images.png)
![A-1 heatmap](../results/figures/selected/A-1_heatmap.png)
![A-1 MS-SSIM scales](../results/figures/selected/A-1_msssim_scales.png)
*MS-SSIM scale by scale. Row 1: the original image at each scale. Row 2: where MS-SSIM sees damage. The sun halo and the rock/sky edge become dominant at scales 3–5, where the weights are largest.*

| | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS | MOS |
|---|---|---|---|---|---|---|
| score | 10.27 dB | 0.962 | 0.843 | 0.466 | 0.299 | 1.40 |
| global rank $r$ | 0.02 | **0.75** | **0.10** | 0.12 | 0.09 | 0.07 |

- **Disagreement:** $g_A = 0.645$, the largest SSIM–MS-SSIM gap in the dataset. SSIM calls this image better than 75% of the dataset. MS-SSIM, every other metric and the humans put it in the bottom 10%. **MS-SSIM agrees with humans.**
- **Systematic or isolated:** the left heatmap shows the same red pattern for **color diffusion, color saturation and color block** (mean gap up to +0.44) across all references. This is a family-wide difference between the two metrics, and I18 is its most extreme instance.
- **Possible factors (draft):**
  - *Scale / spatial extent* *(verified)*. Both metrics see only luma (YIQ Y). Luma does change locally (mean |ΔY| = 16.6 on 0–255, up to 70 around the sun), but smoothly, and the changes cancel in the global mean (151.9 → 152.3). SSIM's luminance term barely penalises a smooth shift at this brightness, so its map lights up only at edges (mean damage 0.04). The MS-SSIM map is bright over the whole halo and the rock edge (mean 0.17). The structure-contrast term $cs$ is 0.97 at the finest scale (the 11×11 window that SSIM uses) and falls steadily to 0.89, 0.81 and 0.72 at the coarser scales, which MS-SSIM weights most (0.30, 0.24, 0.13). The luma damage is a **large-scale, low-frequency** change: the bright halo around the sun and the sky/rock boundary are smeared. A small window sees each patch as "almost unchanged", while a large window sees the change.
  - *Color.* Most of the visible damage, the sky turning cyan and the rock orange, is in **chroma**, which neither metric sees. Here MS-SSIM is right for a partly wrong reason: it catches the luma side-effect of the color change. Compare B-1, where the color change has no luma side-effect and MS-SSIM fails too.
  - *Cross-check with B-22 (verified).* With Wang's 2× downsampling, SSIM's rank for this image drops from 0.75 to 0.46, halving the gap. Viewing scale explains part of the disagreement.

### 4.2 A-2: LPIPS-VGG vs DISTS: `I38_19_04` (jitter, level 4)

![A-2](../results/figures/selected/A-2_images.png)
![A-2 heatmap](../results/figures/selected/A-2_heatmap.png)

| | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS | MOS |
|---|---|---|---|---|---|---|
| score | 24.20 dB | 0.923 | 0.983 | 0.171 | 0.300 | 1.67 |
| global rank $r$ | 0.50 | 0.58 | 0.69 | **0.55** | **0.09** | 0.14 |

- **Disagreement:** gap 0.453. Both metrics use the same VGG16 backbone (D-23), so the difference comes from the **formula**. LPIPS-VGG places the image mid-pack, while DISTS and the humans place it near the bottom. **DISTS agrees with humans.**
- **Systematic or isolated:** within jitter alone both metrics order images almost perfectly (SRCC 0.94 LPIPS, 0.96 DISTS) *(verified)*. The failure is again **cross-type calibration**: the heatmap is red for jitter L3–L4 and color shift L4–L5 across all references, so LPIPS gives jitter smaller distances than its visual impact deserves.
- **Possible factors (draft):**
  - *Where each formula finds the distance* *(verified, stage bars)*. The DISTS score of 0.300 comes mostly from its **texture term at the deepest stages**: relu4_3 contributes 0.109 and relu5_3 0.111 + 0.041 structure. Jitter makes every straight edge ragged, which changes the *global average activation* of deep feature channels (edge and contour detectors). LPIPS spreads its 0.171 over the layers (0.009 → 0.062), and its map shows the damage only along the jet's outline, a small fraction of the image, which the spatial average dilutes.
  - *Small geometric changes.* LPIPS compares normalised features **position by position**. Its weights were learned from 2AFC judgments on 64×64 patches, which may have taught it to discount small shifts.
  - *Salient region.* All the damage is on the object, the region viewers look at. Neither metric weights saliency, but this could explain why the human score is so low.

### 4.3 B-1: metric vs MOS, rank view (SRCC): SSIM on `I32_04_02` (color diffusion, level 2)

![B-1](../results/figures/selected/B-1_images.png)
![B-1 heatmap](../results/figures/selected/B-1_heatmap.png)
![B-1 MS-SSIM scales](../results/figures/selected/B-1_msssim_scales.png)
*MS-SSIM scale by scale: the damage is below 0.05 at every scale (note the colour-bar range). No scale sees the yellow cast, because it is in chroma.*

| | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS | MOS |
|---|---|---|---|---|---|---|
| score | 14.57 dB | **0.996** | 0.997 | 0.263 | 0.221 | 1.77 |
| global rank $r$ | 0.09 | **0.98** | 0.94 | 0.39 | 0.25 | 0.16 |

- **Disagreement:** $r_{SSIM} - r_{MOS} = +0.823$, the largest rank error of any metric on any image. SSIM is **too optimistic**: it rates the image better than 98% of the dataset, while humans rate it worse than 84%. In terms of $d_i$ this single image contributes $(0.823 N)^2$ to $\sum d^2$.
- **The same image tops the PLCC view too.** It has the largest logistic residual of all, $e_{SSIM} = +2.56$ MOS: SSIM's fitted MOS is 4.33, while humans gave 1.77. B-2 therefore shows the next distinct image.
- **Systematic or isolated:** systematic. SSIM is off by more than 0.5 rank for **20.7%** of color-diffusion images (and 21.7% of color saturation 1, 46.9% of color block). Yet within color diffusion SSIM's SRCC is 0.875 *(verified)*. It orders these images correctly but places all of them far too high relative to other distortion types. The heatmap is red along whole rows (color block, color diffusion, color saturation), not in isolated cells.
- **Possible factors (draft):**
  - *Color.* SSIM (and MS-SSIM) in pyiqa use **only the luma channel** (YIQ Y). Here the luma is almost unchanged: mean Y goes from 171.1 to 166.2, and $cs \ge 0.995$ at every scale *(verified)*. The distortion is a strong **yellow cast**, purely chromatic, so the metric cannot see it.
  - *Cross-check with B-22 (verified).* With 2× downsampling SSIM still ranks this image at 0.97, so scale is not the cause here; color is.
  - PSNR, which uses RGB, sees it (14.6 dB, rank 0.09), and so do LPIPS/DISTS, which use RGB features. The cue SSIM lacks is therefore color information, not some deeper perceptual model.

### 4.4 B-2: metric vs MOS, value view (PLCC): DISTS on `I81_20_05` (non-eccentricity patch, level 5)

![B-2](../results/figures/selected/B-2_images.png)
![B-2 heatmap](../results/figures/selected/B-2_heatmap.png)

| | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS | MOS |
|---|---|---|---|---|---|---|
| score | 27.54 dB | 0.930 | 0.956 | 0.103 | **0.051** | 1.33 |
| global rank $r$ | 0.71 | 0.60 | 0.44 | 0.68 | **0.79** | 0.05 |
| fitted MOS $f_m(\tilde s)$ | | | | | **3.79** | 1.33 |

- **Disagreement:** $e_{DISTS} = +2.46$ MOS. The logistic mapping turns DISTS's very small distance into a predicted MOS of 3.79 ("good"), while humans gave 1.33 ("bad"). The residual is 2.46 points on a 4-point scale. This is the third-ranked (metric, image) pair overall, and the first one whose image is not already used by B-1. Every metric is optimistic here (all ranks ≥ 0.44), but DISTS is the most optimistic.
- **Systematic or isolated:** isolated. Only 1.2% of non-eccentricity-patch images have a DISTS rank error above 0.5, and within this type DISTS's SRCC is 0.67 *(verified)*. On the I81 row of the heatmap, L2 and L5 are dark red while L3 is near 0. The failure depends on **where the patches land**, not on the distortion type. (Across all references, DISTS's large positive residuals are on **color block** and **color shift**.)
- **Possible factors (draft):**
  - *Spatial localization.* This distortion copies many small patches to nearby positions. The damage is local and sparse (see the |diff| map); most of the image is untouched.
  - DISTS compares **global** feature statistics (means and variances over the whole map). Moving patches around inside the same image leaves those statistics almost unchanged. DISTS was in fact designed to tolerate this kind of texture resampling. *(verified, stage bars)*: every DISTS term is tiny, at most 0.024 at relu5_3, for a total of 0.051. LPIPS, which compares position by position, finds about twice the distance (0.103), and its map lights up exactly at the moved patches.
  - *Salient / semantic content.* The patches create small discontinuities on the shirt, hair, phone and cup edges, near the face. Viewers notice discontinuities on a person and on object contours much more than the area fraction suggests.

### 4.5 C-1: SSIM vs LPIPS-VGG ranking: `I43_07_05` vs `I43_25_01` (reference I43)

![C-1](../results/figures/selected/C-1_images.png)
![C-1 heatmap](../results/figures/selected/C-1_heatmap.png)

| image | type, level | SSIM | LPIPS-VGG | MOS | $r^{ref}_{SSIM}$ | $r^{ref}_{LPIPS}$ | $r^{ref}_{MOS}$ |
|---|---|---|---|---|---|---|---|
| $i$ = I43_07_05 | color saturation 1, L5 | 0.978 | 0.703 | 1.87 | **0.87** | **0.01** | 0.08 |
| $j$ = I43_25_01 | contrast change, L1 | 0.780 | 0.092 | 3.80 | **0.32** | **0.71** | 0.62 |

- **Disagreement:** $\Delta r_{SSIM} = +0.55$ (SSIM prefers $i$), $\Delta r_{LPIPS} = -0.70$ (LPIPS prefers $j$), strength 0.55, the strongest such conflict with a clear human preference. Humans prefer $j$ by **1.93 MOS**, so **humans side with LPIPS-VGG**. Over all 78,769 SSIM–LPIPS conflicts, humans side with LPIPS in 43%, with SSIM in 24%, and 34% are ties.
- **Systematic or isolated:** systematic. The left heatmap (within-reference $r_{SSIM} - r_{LPIPS}$) is red for color saturation, color diffusion and quantization, and blue for brighten, darken, mean shift, contrast change and high sharpen. SSIM is lenient on color changes and harsh on luminance and contrast changes compared with LPIPS.
- **Possible factors (draft):**
  - *Color (image $i$).* The leaf becomes lavender-gray: hue and saturation change while the luma structure (veins, drops) is intact. SSIM, which sees luma only, scores 0.978. LPIPS uses RGB deep features, which respond strongly to color, and scores it the worst image of the reference.
  - *Luminance in dark regions (image $j$)* *(verified, SSIM map)*. The leaf looks "enhanced", and the SSIM map is almost black on it. **80% of SSIM's damage lies in the dark, out-of-focus background** (37% of the image), whose luma drops from 42.5 to 13.8. SSIM's luminance term $\frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}$ is *relative*: the same drop costs far more in dark regions (here l ≈ 0.6). Humans hardly care about a darker, blurred background. LPIPS's map is dark everywhere for $j$ (0.09), but bright over the whole frame for $i$ (0.70), where the chroma panel shows the large color change that SSIM cannot see.

---


## 5. Discussion

**Strengths.**
- *PSNR* is robust to global brightness changes; it has the best brightness-group SRCC, because a brightness shift is exactly an error in pixel values.
- *SSIM and MS-SSIM* are strong on blur, compression and noise, where structure is lost. MS-SSIM's multi-scale design makes it the best single metric on KADID overall.
- *LPIPS-VGG and DISTS* are the only metrics that stay monotone and reasonably correlated on spatial distortions (pixelate, jitter, patches), where pixel alignment breaks.

**Limitations.**
- *Calibration across types* is the main weakness of every metric (§3, point 2). A metric can order the five levels of one distortion correctly and still place that whole distortion at the wrong quality compared with another.
- *Color.* SSIM and MS-SSIM ignore chroma by design (luma only), which explains cases B-1 and C-1. LPIPS sees color but over-penalises some color changes relative to others (color group SRCC 0.48).
- *Scale.* Single-scale SSIM misses large, low-frequency changes (case A-1, and +0.13 SRCC with downsampling).
- *Global statistics.* DISTS pools features over the whole image, so sparse, local damage can go unnoticed (case B-2).

**Does any single metric consistently reflect human perception? No.**
- The best overall metric (MS-SSIM, 0.83) still fails on color saturation (0.49), non-eccentricity patch (0.44) and pixelate (0.48).
- The learned metrics win exactly there but lose to MS-SSIM overall, and the stronger learned one (DISTS) was trained on this dataset.
- The ranking of the metrics changes with the distortion type (heatmaps in §3).

**A perceptual factor not adequately captured:**
- **Chromatic change.** SSIM and MS-SSIM cannot see it, and the deep metrics do not scale it consistently against other distortions (color group SRCC 0.48–0.74 for every metric vs ≥ 0.89 on blur).
- A second candidate is **localized damage on salient content** (case B-2). No metric weights *where* the viewer looks.


| case | metric that fails | factor it ignores or over-weights | systematic? |
|---|---|---|---|
| A-1 | SSIM | large-scale (low-frequency) changes: single small window | yes (color types) |
| A-2 | LPIPS-VGG | ragged edges from small geometric jitter: diluted by spatial averaging, while DISTS's deep texture statistics catch them | yes (jitter, color shift) |
| B-1 | SSIM (and MS-SSIM) | **color**: luma-only input | yes (20.7% of color diffusion) |
| B-2 | DISTS | **spatial localization**: global statistics ignore sparse local patch moves | no (content-specific) |
| C-1 | SSIM | color (ignored); luminance change in dark, unimportant regions (over-penalised by its relative luminance term) | yes |

Candidate answer to the spec question *"a perceptual factor not adequately captured"*: **chromatic change.** It is invisible to SSIM and MS-SSIM (luma only) and only weakly scaled across types by LPIPS. A second candidate is **sparse, localized damage on salient content**, which DISTS's global statistics miss.


## Contribution Statement (draft, B-17)
- **@hwupu (Member A):** project lead (PO); planning, code with an AI coding agent (data loader, metric registry, scoring, evaluation, failure mining, figures), drafted the report, selected the five cases by quantitative criteria.
- **@ALLEN07056 (Member B):** code reviewer for all PRs, cross-platform verification on WSL, verification of results and failure-case explanations, report review.

**Generative AI disclosure (draft).**
- Claude Code (Claude Opus 5.5) was used to write code, run the experiments and draft the text.
- Every AI interaction that produced kept output is logged in `doc/PROMPT_LOG.md` (P-001 to P-010), and every design choice in `doc/DECISIONS.md`.
- Humans chose the case-selection rules, reviewed every PR, and own the final explanations.

## References
- H. Lin, V. Hosu, D. Saupe, "KADID-10k: A large-scale artificially distorted IQA database," QoMEX 2019.
- Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," IEEE TIP 2004.
- Z. Wang, E. P. Simoncelli, A. C. Bovik, "Multiscale structural similarity for image quality assessment," Asilomar 2003.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," CVPR 2018.
- K. Ding, K. Ma, S. Wang, E. P. Simoncelli, "Image quality assessment: unifying structure and texture similarity," IEEE TPAMI 2022.
- H. R. Sheikh, M. F. Sabir, A. C. Bovik, "A statistical evaluation of recent full reference image quality assessment algorithms," IEEE TIP 2006 (logistic mapping).
- C. Chen, J. Mo, "IQA-PyTorch (pyiqa)," https://github.com/chaofengc/IQA-PyTorch.
- MR-IQA: Mask Reference Image Quality Assessment, https://arxiv.org/pdf/2302.13770 (Table 2, literature numbers in Background).
- A comprehensive evaluation of full-reference IQA algorithms on KADID-10k, https://arxiv.org/pdf/1907.02096.
- JND-SalCAR, https://arxiv.org/pdf/1902.05316.
- PIQ documentation, https://piq.readthedocs.io/en/latest/overview.html.

## Remaining before submission (Release: B-18, B-19)
- [ ] Verify the literature numbers in the Background table against the cited papers, and find why our DISTS (0.814) is below the reported 0.887.
- [ ] Convert to PDF (D-08) and cut to ≤ 4 pages: keep 1 image figure per case, the scatter plot, one heatmap; move the rest to the repo.
- [ ] Clean-clone reproduction on WSL (B-18; macOS done).
- [ ] Submit the PDF + repository link (B-19).
