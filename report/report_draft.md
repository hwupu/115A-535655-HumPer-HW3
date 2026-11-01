# Perceptual Metric Reproduction & Failure Analysis on KADID-10k: report draft

> **DRAFT (B-26, P-010).** The structure follows the six required sections of the spec (`doc/references/HW3.pdf` §6).
> Every number is reproduced by `notebooks/02`–`05` from the committed `results/`.
> The perceptual explanations in §4–§5 started as AI hypotheses (AGENTS.md). Lines marked *(verified)* were checked numerically.
> When @ALLEN07056 approves the PR, the team takes ownership of the text (D-26). Before the PDF: cut to ≤ 4 pages (B-19) and convert (D-08).

## Background: the metrics and the evaluation, explained simply

> Primer adapted from the team's ELI5 notes (2026-11-02). It is longer than the PDF can hold: condense it into §1 for the final report (B-19).
> The *In this project* lines link each idea to our setup and results.

### Full-reference image quality metrics

A full-reference metric compares a distorted image with its pristine original and outputs one number. The question is how well that number agrees with what people see.

**PSNR (Peak Signal-to-Noise Ratio): "how different is each dot?"**
- It compares the two images pixel by pixel, averages the squared differences (MSE) and converts the result to decibels:
$$\text{PSNR} = 10\log_{10}\frac{\text{MAX}^2}{\text{MSE}}, \qquad \text{MAX} = 255 \text{ for 8-bit images}$$
- Higher = closer to the original.
- Rules of thumb:
  - above 40 dB: hard to tell from the original;
  - 30–50 dB: good;
  - 20–30 dB: noticeable loss;
  - below 20 dB: clearly degraded;
  - identical images: ∞.
- **Catch:** it counts raw pixel errors, not what humans notice. A tiny shift of the whole image costs a lot of PSNR but looks fine.
- *In this project:* computed on RGB. Identical images are capped at 80 dB (D-18).

**SSIM (Structural Similarity): "does it still look like the original?"**
- In small windows (an 11×11 Gaussian) it compares three things:
  - **luminance:** is the window equally bright? (μ)
  - **contrast:** is the light/dark spread the same? (σ)
  - **structure:** do the patterns and edges line up? (σ_xy)
- The three comparisons are multiplied into one local score, and the local scores are averaged over the image:
$$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
- 1 = identical, about 0.9 or more = very similar, 0.5–0.8 = visibly different.
- It is **hand-crafted**: nothing is learned from data, and only the constants and the window size are chosen.
- Compared with PSNR:
  - A slight brightening hurts PSNR a lot but SSIM little.
  - Blurred edges hurt SSIM more than PSNR.
- Shifts, rotations and heavy distortions can still fool it.
- *In this project:* pyiqa computes SSIM on the **luma (Y) channel only**, so a pure color change is invisible to it (case B-1). The luminance term is *relative*, so the same brightness drop costs more in dark regions (case C-1).

**MS-SSIM (Multi-Scale SSIM): SSIM from several viewing distances.**
- A painting shows brushstrokes from 5 cm away and overall shapes from across the room. SSIM looks from one distance only.
- MS-SSIM compares contrast and structure at 5 scales, halving the image size each time, and compares brightness only at the coarsest scale.
- It multiplies the per-scale results, each raised to a weight measured with human viewers (the middle scales matter most):
$$\text{MS-SSIM} = [l_M]^{\alpha_M}\prod_{j=1}^{M}[c_j]^{\beta_j}[s_j]^{\gamma_j}$$
- It is still hand-crafted, fast and deterministic. It needs images large enough to downsample several times, and it is still weak on geometric shifts.
- *In this project:* the weights are (0.045, 0.286, 0.300, 0.236, 0.133). Large, smooth damage that SSIM's small window misses shows up at the coarse scales (case A-1 and its scale-by-scale figure).

**Backbone: the pretrained network the learned metrics build on.**
- A large network such as VGG or AlexNet is trained for ImageNet classification. It turns an image into features: early layers capture edges and textures, deep layers capture shapes and objects.
- A learned metric adds a small part on top that turns feature differences into a score.
- The metric inherits the backbone's strengths and blind spots, and **different backbones give different, non-comparable scores**.
- *In this project:* LPIPS and DISTS both use **VGG16**, so differences between them come from the formula (D-23). Switching LPIPS to AlexNet changes its SRCC from 0.720 to 0.822 (§3).

**LPIPS (Learned Perceptual Image Patch Similarity): a trained "robot eye".**
- Both images go through the backbone. At each of 5 layers the features are normalised and their squared difference is weighted by **linear weights learned from human 2AFC judgments** ("which of two patches is closer to the original?", the BAPPS dataset, 64×64 patches). The result is averaged over space and summed over layers.
- 0 = identical, about 0.1 = very similar, 0.5 or more = clearly different. **Lower = better.**
- It tolerates things humans ignore (tiny shifts, similar textures) and punishes things they notice (blur, artifacts).
- The original paper found that even uncalibrated deep features beat SSIM and PSNR.
- It can be fooled by adversarial noise.
- *In this project:* LPIPS-VGG v0.1. Its spatial map and per-layer values appear in the case figures; they add up to the score.

**DISTS (Deep Image Structure and Texture Similarity): "SSIM on deep features, with texture tolerance".**
- Two photos of the same grass field have every blade in a different place: pixels say "totally different", the eye says "same grass". DISTS agrees with the eye.
- It runs both images through VGG16. At each layer and channel it compares:
  - **texture:** the global *means* of the feature maps, ignoring position;
  - **structure:** the global *correlation* between the feature maps.
- Learned weights combine the terms over layers. 0 = identical, **lower = better.**
- DISTS replaces max-pooling with **L2 pooling**:
  - max-pooling keeps only the largest value in each 2×2 window, so it jumps when content shifts by one pixel;
  - L2 pooling takes the root-mean-square of the window, so every value contributes and small shifts change little.
- It tolerates texture resampling (grass, bricks, fur, water) and does well on GAN and super-resolution outputs.
- *In this project:* DISTS's weights were **trained on KADID-10k**, our test set (D-22). Because it compares whole-image statistics, it has no spatial map; the case figures show its per-layer texture/structure terms instead. That same global view explains case B-2.

### KADID-10k

81 clean photos, each damaged in 25 ways at 5 strengths, giving 10,125 images rated by people (a mean opinion score, MOS, from 1 to 5, higher = better). The distortion groups are blur, color, compression (JPEG, JPEG2000), noise, brightness change, spatial distortions, and sharpness/contrast.
- These are mostly classical distortions, not the artifacts of generative models, where LPIPS and DISTS tend to shine more.

### How agreement with humans is measured

**SRCC (Spearman rank correlation): "same ranking?"**
- Rank all images by the humans and by the metric, then correlate the two rankings. 1 = identical order, 0 = unrelated, −1 = reversed.
- It looks at order only, so it compares metrics fairly across very different scales.
- *In this project:* instead of reporting |SRCC| for lower-is-better metrics, we flip their sign first (§2), so every metric reads "higher = agrees more".

**PLCC (Pearson linear correlation): "same straight line?"**
- It uses the actual values, so a curved metric–MOS relation lowers it. As is standard, we first fit a monotonic 5-parameter logistic curve (§2).
- **High SRCC with lower PLCC** means the metric ranks correctly but its scale is warped relative to human scores.

### Summary and what the literature reports

| metric | type | key idea | scales | better | KADID-10k SRCC: literature† | KADID-10k SRCC: ours |
|---|---|---|---|---|---|---|
| PSNR | hand-crafted | pixel error | pixel | higher | 0.676 | 0.676 |
| SSIM | hand-crafted | luminance, contrast, structure | single | higher | 0.724 | 0.619 (0.752 with Wang's downsampling) |
| MS-SSIM | hand-crafted | SSIM across zoom levels | 5 | higher | 0.826 | 0.826 |
| LPIPS | learned | calibrated deep-feature distance | 5 layers | lower | 0.843 | 0.720 VGG / 0.822 AlexNet |
| DISTS | learned | deep structure + texture statistics | 6 stages | lower | 0.887 | 0.814 |

†As reported in MR-IQA (Table 2). Other papers report different values, e.g. SSIM 0.783 and MS-SSIM 0.838 (JND-SalCAR), or SSIM ≈ 0.72 and DISTS ≈ 0.88 (PIQ). The differences come from implementation details: color vs grayscale, resizing, and the backbone.
- **PSNR and MS-SSIM** match the literature exactly.
- **SSIM** matches once Wang's original downsampling is used (B-22).
- **LPIPS** is close with the AlexNet backbone, which the literature number most likely uses.
- **DISTS** is 0.07 lower than reported, and we have not explained why yet (TODO: check whether the published numbers resize the images, or use a different weight version).

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
