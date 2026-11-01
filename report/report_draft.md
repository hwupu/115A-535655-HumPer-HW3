# Perceptual Metric Reproduction & Failure Analysis on KADID-10k

## Background on Image Quality Metrics

---

### 1. PSNR (Peak Signal-to-Noise Ratio)

A score for how close a damaged or compressed picture is to the original.

**Brief.** Photocopy a drawing and the copy comes out a bit fuzzy. PSNR is a number saying how fuzzy the copy is compared to the original.

- **Higher = more like the original**
- **Lower = messier**

**Steps.**
1. Compare original and copy pixel by pixel.
2. Average the squared differences (this is the MSE, mean squared error).
3. Convert to a log scale in decibels.

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$

MAX is the brightest possible pixel value (255 for 8-bit images).

**Rules of thumb.**
- 30-50 dB means good quality (above ~40 dB is very hard to tell from the original)
- 20-30 dB means noticeable loss
- Below 20 dB means clearly degraded
- Infinite means identical images

**Catch.** It only counts raw pixel differences, not what humans notice. A tiny shift of the whole image hurts PSNR a lot but looks fine.

---

### 2. SSIM (Structural Similarity Index)

Scores how similar two images *look*, based on how the eye judges them rather than raw pixel differences.

**Brief.** PSNR asks "how different is each dot?" SSIM asks "does this still look like the original?" It checks whether patterns, shapes, and edges survived.

**Compared in small patches.**
1. **Luminance.** is the patch equally bright?
2. **Contrast.** is the light/dark difference the same?
3. **Structure.** do patterns and edges line up? (the key part)

These are multiplied into one score per patch, then averaged over the image.

$$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

Here $\mu$ is the average brightness, $\sigma^2$ the variance (contrast), and $\sigma_{xy}$ how the two patches vary together (structure).

**Reading it.** 1.0 = identical; ~0.9+ = very similar; ~0.5-0.8 = visibly different; near 0 = no structural similarity.

**Why better than PSNR.** Brighten a photo slightly and PSNR may drop a lot while SSIM stays high. Blur the edges and PSNR may barely change while SSIM drops.

**Limits.** Not a true model of human vision. Fooled by shifts, rotations, and heavy distortions.

**Is SSIM hand-crafted?** Yes. A researcher designed the formula by hand from assumptions about vision. Nothing is learned from data. Only small constants (c1, c2) and the window size (e.g. 11x11 Gaussian) are chosen by hand.

---

### 3. LPIPS (Learned Perceptual Image Patch Similarity)

A neural network judges how different two images *look*, trained to agree with human judgments.

**Brief.** PSNR counts differing dots, SSIM checks whether patterns survived, LPIPS asks a trained "robot eye" whether two pictures *feel* the same. That robot eye learned from thousands of human "which is closer to the original?" judgments.

**Steps.**
1. Feed both images into a pretrained deep network (VGG, AlexNet, SqueezeNet).
2. Grab internal features from several layers (early = edges/textures, deep = shapes/objects).
3. Compare features layer by layer using learned weights.
4. Average into one number.

**Reading it.** 0 = identical; ~0.1 = very similar; ~0.5+ = clearly different. **Lower is better** (the reverse of PSNR/SSIM).

**Why useful.** Tolerates things humans ignore (tiny shifts, similar-looking textures) and punishes things humans notice (blur, odd artifacts, wrong textures). Standard for generated images, super-resolution, and style transfer.

**Limits.** Depends on the backbone, so scores aren't comparable across backbones. Can be fooled by adversarial noise. Only an approximation of human perception.

**Is it deep-learning based?** Yes, with two learned parts.
- **Backbone.** trained for ImageNet classification, not quality assessment, yet its features match human perception surprisingly well.
- **Linear calibration layers.** trained on human similarity judgments (BAPPS dataset). The paper also found that even untouched, uncalibrated features beat SSIM and PSNR.

---

### 4. DISTS (Deep Image Structure and Texture Similarity)

A learned metric that treats **structure** and **texture** differently. It sits between SSIM and LPIPS, using deep features like LPIPS and an SSIM-style comparison.

**Brief.** Two photos of the same grassy field have every blade in a different spot, so a pixel comparison says "totally different!", but your eye says "same grass." DISTS agrees with your eye. Layout matters (structure), but the exact placement of repeating texture doesn't.

**Steps.**
1. Run both images through a pretrained VGG, with max-pooling replaced by L2 pooling (so small shifts matter less).
2. At each layer and channel, compute two terms.
   - **Texture term.** compare the *means* of the feature maps (overall texture statistics, ignoring position)
   - **Structure term.** compare the *correlation* between feature maps (spatial layout)
3. Combine with learned weights across layers.

Think of it as **"SSIM on deep features, with texture tolerance."**

**Reading it.** 0 = identical; higher = more different. **Lower is better.**

**Why useful.** Tolerant of texture resampling (grass, bricks, fur, water). Good for GAN outputs and super-resolution. Agrees well with human ratings.

**Limits.** Depends on the VGG backbone, can be fooled by adversarial changes, and its learned weights may not generalize to every image type.

---

### 5. Pooling, max-pooling vs L2 pooling

**Pooling** shrinks a grid of numbers by summarizing small windows (say 2x2) into one number each.

**Max-pooling asks "who's the loudest?"** Keep the biggest number, discard the rest.

```
[1  3]
[2  9]   ->  9
```

- Keeps the strongest signal (e.g. "an edge is somewhere here")
- Loses *where* in the window it was
- Jumpy, since a 1-pixel shift can change which value is the max

**L2 pooling asks "what's the overall energy?"** Square every number, average, take the square root (root-mean-square).

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

**Brief.** A painting looks different from 2 inches away (brushstrokes) than from across the room (overall shapes). SSIM looks from one distance only. MS-SSIM looks from several and combines them.

**Steps.**
1. Start with both images at full size.
2. Compute contrast and structure comparisons (brightness only at the coarsest scale).
3. Blur and downsample both by 2, then repeat, typically for 5 scales.
4. Multiply the per-scale scores, each raised to a weight found from experiments with human viewers (middle scales matter most).

$$\text{MS-SSIM} = \left[l_M\right]^{\alpha_M} \cdot \prod_{j=1}^{M} \left[c_j\right]^{\beta_j} \left[s_j\right]^{\gamma_j}$$

**Reading it.** 1.0 = identical; closer to 0 = more different. Higher is better.

**Why useful.** Adapts to viewing distance and resolution, matches human ratings better than single-scale SSIM, still hand-crafted so fast and deterministic. Common in compression research and as a training loss.

**Limits.** Needs images large enough to downsample several times. Still weak on geometric shifts and heavy generative distortions.

---

### 7. KADID-10k benchmark

A dataset for testing how well quality metrics agree with human opinion.

**Brief.** Take 81 clean photos and wreck each one in 25 ways at 5 strengths, giving 10,125 damaged images. People rated how bad each looks (a mean opinion score, MOS). A good metric ranks images the same way humans do.

Distortion groups include blur, noise, color changes, compression (JPEG, JPEG2000), brightness changes, spatial distortions, and sharpness/contrast changes.

**Results (SRCC / PLCC, from the MR-IQA paper's Table 2).**

| Metric | SRCC | PLCC |
|---|---|---|
| PSNR | 0.676 | 0.675 |
| SSIM | 0.724 | 0.717 |
| MS-SSIM | 0.826 | 0.820 |
| LPIPS | 0.843 | 0.839 |
| DISTS | 0.887 | 0.886 |

The ordering PSNR < SSIM < MS-SSIM < LPIPS < DISTS tracks each metric's perceptual awareness.

**Caveats.**
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
Think of the foundation and frame of a house. You don't rebuild it, you add rooms on top. In deep learning, it's a big pretrained network (VGG, AlexNet) that already learned to see by training on millions of images (ImageNet), turning an image into features (edges, textures, shapes, objects).

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
| PSNR | RGB; identical images capped at 80 dB | 0.676 (0.676) | used as the pixel baseline |
| SSIM | pyiqa, **luma (Y) only**, no downsampling | 0.619 (0.724); **0.752** with Wang's downsampling | B-1 (color change is invisible), C-1 (relative luminance term over-penalises dark regions) |
| MS-SSIM | luma only, weights 0.045 / 0.286 / 0.300 / 0.236 / 0.133 | 0.826 (0.826) | A-1 (coarse scales catch large, smooth damage) |
| LPIPS | **VGG16** v0.1; AlexNet kept as supplementary | 0.720 VGG / 0.822 AlexNet (0.843) | A-2, B-2, C-1 (spatial map and per-layer values) |
| DISTS | VGG16; weights **trained on KADID-10k**, so in-domain | 0.814 (0.887); we could not explain this gap | A-2, B-2 (per-stage texture and structure terms, no spatial map) |

Instead of reporting |SRCC| for LPIPS and DISTS, we flip their sign before correlating (§2). PLCC is computed after the 5-parameter logistic fit.

**Sources.** [MR-IQA](https://arxiv.org/pdf/2302.13770) · [Full-reference IQA on KADID-10k](https://arxiv.org/pdf/1907.02096) · [JND-SalCAR](https://arxiv.org/pdf/1902.05316) · [PIQ documentation](https://piq.readthedocs.io/en/latest/overview.html)

## 1. Metrics and Dataset

We test five image quality metrics on KADID-10k and ask one question. **Does each metric judge picture quality the way people do?**

**Brief.** Show people thousands of damaged photos and ask "how bad does this look?". Then ask five robots the same question. A good robot gives the worst score to the photos people hated and the best score to the photos people liked.

**The dataset, KADID-10k**

| | |
|---|---|
| Clean photos | 81 (512 × 384 pixels) |
| Ways to damage them | 25 distortion types in 7 groups (blur, color, compression, noise, brightness, spatial, sharpness/contrast) |
| Strengths | 5 levels per type, from L1 (mild) to L5 (strong) |
| Damaged photos | 81 × 25 × 5 = **10,125** |
| Human score | MOS from 1 (bad) to 5 (excellent), **higher = better**, about 30 ratings per photo |

We use all 10,125 photos, so no sampling is needed.

**The five metrics and what each one believes about human vision**

| Metric | Type | What it compares | Its belief about vision |
|---|---|---|---|
| PSNR | hand-crafted | pixel errors (RGB) | every pixel error is equally visible |
| SSIM | hand-crafted | brightness, contrast and structure in small 11×11 windows, on brightness (luma) only | the eye cares about *structure*, so broken edges and patterns hurt more than a uniform change |
| MS-SSIM | hand-crafted | SSIM at 5 zoom levels | the same, and how visible a change is depends on viewing distance |
| LPIPS-VGG | learned | deep VGG16 features, position by position, weighted to match human judgments | features trained to recognise objects also capture what looks similar |
| DISTS | learned | deep VGG16 features, compared through whole-image statistics, texture (feature averages) and structure (feature correlations) | quality = structure + texture, so re-arranged texture (grass, gravel) is fine |

**Two things to keep in mind.**
- **Same backbone.** LPIPS and DISTS both use VGG16, so any difference between them comes from their *formula*, not from the network. We also kept LPIPS with an AlexNet backbone for comparison.
- **DISTS has seen this test before.** Its weights were trained on KADID-10k (Ding et al., Sec. II-D), so its scores here are optimistic, like a student who studied the exam questions.

## 2. Experimental Setup

**Brief.** Every robot gets the same photos, at the same size, through the same code. Before comparing robots with people, we make sure every robot "counts in the same direction" as people do.

**Steps.**
1. Load each damaged photo and its clean original at full size (no resizing, no cropping).
2. Score every pair with each metric, using the official implementations in the `pyiqa` library (v0.1.16).
   - PSNR uses the RGB pixels.
   - SSIM and MS-SSIM use only brightness (the Y channel of YIQ).
   - LPIPS and DISTS use RGB.
3. Align the score directions (below).
4. Compare the metric scores with the human scores (SRCC and PLCC, below).
5. Find the photos where metrics fail (Section 4).

Scoring all 10,125 pairs took 40 minutes on a laptop (Apple M2 Pro). The code, the scores and a test suite are in the project repository, and anyone can reproduce every number with one command per step.

**Sanity checks before trusting the numbers.**

| Check | Result |
|---|---|
| Identical images get a perfect score | PSNR 80 dB (its cap), SSIM 1, LPIPS ≈ 0, DISTS ≈ 0 |
| PSNR matches another library (scikit-image) | within 0.00005 dB |
| SSIM matches the textbook formula | yes, except that pyiqa clips negative structure values to 0, which changes nothing in practice (Section 3) |
| Same results on CPU and GPU, and on a second computer | yes |
| 320 photos are pixel-identical to their original (e.g. "mean shift" level 3) | kept for correlations (removing them changes SRCC by ≤ 0.04), skipped when hunting failures |

**Score direction, making every metric count the same way**

**Brief.** People give 5 to the best photo. PSNR, SSIM and MS-SSIM also give *high* numbers to good photos. LPIPS and DISTS are *distances*, so they give *low* numbers to good photos. If we compared them without fixing this, a perfect distance metric would look perfectly wrong.

So we flip the sign of the two distance metrics before comparing.

$$
\tilde s_m(i) = \begin{cases} s_m(i) & \text{PSNR, SSIM, MS-SSIM (higher = better)}\\ -\,s_m(i) & \text{LPIPS, DISTS (lower = better)}\end{cases}
$$

**Proof that the directions now match the human scores.**

| Metric | Native direction | SRCC before flipping | SRCC after flipping |
|---|---|---|---|
| PSNR | higher = better | +0.676 | +0.676 |
| SSIM | higher = better | +0.619 | +0.619 |
| MS-SSIM | higher = better | +0.826 | +0.826 |
| LPIPS-VGG | lower = better | **−0.720** | +0.720 |
| DISTS | lower = better | **−0.814** | +0.814 |

Only the two distance metrics start negative, exactly as expected. After flipping, every metric agrees in direction with the human scores.

**How we measure agreement**

**SRCC asks "do they put the photos in the same order?"** Rank all photos by the humans, rank them by the metric, and correlate the two rankings.

$$
\text{SRCC} = 1 - \frac{6\sum_i d_i^2}{N(N^2-1)}, \qquad d_i = \text{rank by metric} - \text{rank by humans}
$$

A photo that the metric ranks far away from the humans (a big $d_i$) costs the most SRCC.

**PLCC asks "do the numbers line up on a straight line?"** Metric scores and human scores are rarely related by a straight line, so we first bend the metric scores with the standard S-shaped (5-parameter logistic) curve $f_m$, then correlate.

$$
f_m(x) = \beta_1\Big(\tfrac12 - \tfrac{1}{1+e^{\beta_2 (x-\beta_3)}}\Big) + \beta_4 x + \beta_5, \qquad
\text{PLCC} = \text{corr}\big(f_m(\tilde s_m),\ \text{MOS}\big)
$$

$f_m(x)$ is "the human score the metric predicts". The gap $f_m(x) - \text{MOS}$ is the metric's error in human-score units, and the photos with the biggest gaps cost the most PLCC.

**Comparing metrics on different scales.** PSNR is in decibels, SSIM between 0 and 1, LPIPS a distance. To compare them on one photo we use **percentile ranks**, where $r = 0.9$ means "the metric thinks this photo is better than 90% of all photos". People get the same treatment ($r_{MOS}$).

## 3. Quantitative Results

**Overall agreement with people (all 10,125 photos, 1 = perfect).**

| | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS* | *(LPIPS-AlexNet)* |
|---|---|---|---|---|---|---|
| SRCC | 0.676 | 0.619 | **0.826** | 0.720 | 0.814 | *0.822* |
| PLCC | 0.680 | 0.628 | **0.823** | 0.729 | 0.814 | *0.823* |

\*DISTS was trained on this dataset.

![scatter](../results/figures/quant/scatter.png)
*Each dot is one photo, with the metric score on x and the human score on y,, coloured by distortion group. The black line is the fitted S-curve. LPIPS and DISTS slope downward because they are distances (lower = better).*

**Brief.** The best robot (MS-SSIM) agrees with people about 83% of the way. None is close to perfect, and the hand-crafted MS-SSIM beats both learned metrics overall.

**Agreement per distortion group (SRCC).**

| Group | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS |
|---|---|---|---|---|---|
| blur | 0.888 | 0.911 | 0.950 | 0.954 | **0.965** |
| brightness | **0.900** | 0.856 | 0.879 | 0.867 | 0.892 |
| color | 0.683 | 0.310 | **0.743** | 0.484 | 0.651 |
| compression | 0.827 | 0.855 | **0.918** | 0.906 | 0.917 |
| noise | 0.820 | 0.768 | **0.858** | 0.825 | 0.838 |
| sharpness & contrast | 0.872 | 0.866 | **0.923** | 0.906 | 0.918 |
| spatial | 0.681 | 0.386 | **0.713** | 0.531 | 0.632 |

![SRCC by type](../results/figures/quant/srcc_by_type.png)
![SRCC by type and level](../results/figures/selected/srcc_type_level.png)
*Agreement inside each distortion type (top) and inside each type and level (bottom, 81 photos per cell). L1 is the mildest level and L5 the strongest. Blue means agreement with people, red means reversed, gray means no data.*

**What the numbers tell us.**

1. **No metric wins everywhere.** MS-SSIM wins 5 of 7 groups, DISTS wins blur and PSNR wins brightness. On blur, compression and sharpness every metric is within 0.08 of the best, so those are easy.

2. **Ranking within one distortion is easy; comparing different distortions is hard.**
   **Brief.** A judge who can rank five cakes from best to worst may still not know whether the best cake beats the worst pizza.
   Inside one distortion type every metric does well (median SRCC 0.83–0.88). For example, SSIM scores 0.88 inside "color diffusion" but only 0.31 across all color distortions, and LPIPS scores 0.94 inside "jitter" but 0.53 across spatial distortions. Most of each metric's errors come from putting *different kinds* of damage on one common scale. This is also where the failure cases in Section 4 come from.

3. **Color and spatial damage are hard for everyone.** SSIM collapses on color (0.31) because it only looks at brightness. On spatial damage (moved patches, pixelation), the pixel-by-pixel metrics drop to 0.29–0.48, while LPIPS and DISTS keep 0.65–0.78.

4. **Formula vs backbone.** LPIPS-VGG (0.720) loses to DISTS (0.814) with the *same* VGG16 network, so the formula matters. But LPIPS with AlexNet scores 0.822, so the backbone matters just as much.

5. **Some metrics need the S-curve.** Without the curve, PLCC is 0.556 for PSNR and 0.680 for MS-SSIM; with it, 0.680 and 0.823. Their relation to human scores is bent (see the scatter plot). DISTS is almost straight already (0.806 → 0.814).

**Does a stronger distortion always get a worse score?**

**Brief.** Level 5 is stronger than level 4, so every metric should score it worse. We check this for each photo and distortion type.

| | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS | People (MOS) |
|---|---|---|---|---|---|---|
| Photo series that get worse at every level | 81% | 83% | 82% | **90%** | 85% | 62% |

- **People are the least consistent.** Neighbouring levels often look almost the same, and rating noise reverses them.
- **Two types are not "stronger with each level".** For mean shift and contrast change, KADID's levels go in *both* directions around the original (mean-shift level 3 is the original itself). Every metric and the humans "fail" there, so we ignore them.
- **Pixelation fools the pixel metrics.** PSNR, SSIM and MS-SSIM get worse at every level in ≤ 6% of series (level 3 often scores better than level 2), while LPIPS and DISTS do in 100%.

**Why is our SSIM lower than in other papers?**

Our SSIM (0.62) is below the commonly reported ~0.72. We changed one implementation detail at a time.

| SSIM version | SRCC | PLCC | Color group SRCC |
|---|---|---|---|
| pyiqa default (used in this report) | 0.619 | 0.628 | 0.310 |
| without clipping negative structure values | 0.619 | 0.628 | 0.310 |
| with the original 2× downsampling first | **0.752** | **0.748** | **0.704** |

**Brief.** The original SSIM first shrinks the image by half, as if you stepped back from the screen. From further away, the smooth brightness changes that come with color damage become visible to SSIM's small window. That one step adds 0.13 SRCC, almost all of it on color. We keep the library's default for the rest of this report, because it is what most people run, but **SSIM's score depends a lot on viewing scale.**

## 4. Failure Cases

We picked the five cases **by the numbers first**, then looked at the pictures (no cherry-picking). Each case is the single strongest example of one kind of disagreement.

| Case | Question | How we score the disagreement |
|---|---|---|
| A-1 | Where do SSIM and MS-SSIM disagree most? | $\lvert r_{SSIM} - r_{MS\text{-}SSIM}\rvert$ |
| A-2 | Where do LPIPS and DISTS disagree most? | $\lvert r_{LPIPS} - r_{DISTS}\rvert$ |
| B-1 | Which metric gets a photo's **rank** most wrong (what SRCC punishes)? | $\lvert r_m - r_{MOS}\rvert$ over all metrics |
| B-2 | Which metric predicts a photo's **score** most wrong (what PLCC punishes)? | $\lvert f_m(x) - \text{MOS}\rvert$ over all metrics |
| C-1 | Where do SSIM and LPIPS **order two damaged versions of the same photo** in opposite ways? | the smaller of the two rank differences, with people clearly preferring one photo |

Photos where the raters disagreed a lot (top 10% of rating variance) are skipped, and each case uses a different photo.

**How to read the figures.**
- **Case figure.** the original, the damaged photo, then only the views of the metrics that disagree. Each view adds up to the metric's own score.

| Panel | What it shows |
|---|---|
| \|ΔY\| brightness / \|ΔIQ\| color | how much brightness and color changed (same scale). SSIM and MS-SSIM only see brightness |
| 1 − SSIM map | where SSIM sees damage |
| 1 − MS-SSIM map | where MS-SSIM sees damage, all 5 zoom levels combined |
| LPIPS map | where LPIPS sees damage |
| VGG16 stage bars | how much each network layer adds to the LPIPS and DISTS scores; DISTS split into texture and structure. DISTS uses whole-image statistics, so it has no map |
| \|RGB diff\| | where the pixels changed |

- **Heatmap.** distortion type (rows) × level (columns, L1 = mildest … L5 = strongest). The left panel averages over all 81 photos, the right panel shows only this case's photo. The black box marks the case. Red = the first-named metric is more generous; blue = the second is.

### 4.1 A-1. SSIM vs MS-SSIM on photo 18 with color diffusion, level 4

![A-1](../results/figures/selected/A-1_images.png)
![A-1 MS-SSIM scales](../results/figures/selected/A-1_msssim_scales.png)
*MS-SSIM zoom level by zoom level. The top row is the original at each scale, the bottom row shows where MS-SSIM sees damage. The sun halo and the rock edge dominate at the coarse scales, which carry the most weight.*
![A-1 heatmap](../results/figures/selected/A-1_heatmap.png)

| | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS | People |
|---|---|---|---|---|---|---|
| score | 10.27 dB | 0.962 | 0.843 | 0.466 | 0.299 | MOS 1.40 |
| rank r | 0.02 | **0.75** | **0.10** | 0.12 | 0.09 | 0.07 |

**What happened.** SSIM says this photo is better than 75% of the dataset. MS-SSIM, every other metric and the people put it in the bottom 10%. **MS-SSIM is right.**

**Brief.** Stand close to a painting and you see that each brushstroke is fine. Step back and you see that the whole sky is the wrong color. SSIM only stands close.

**Why.**
- **Big, smooth damage.** Brightness changes a lot (on average 17 out of 255, up to 70 near the sun), but smoothly, over large areas. Inside SSIM's small 11×11 window, a smooth change looks like "almost nothing happened", so SSIM only flags the edges (average damage 0.04). MS-SSIM also looks at shrunken copies of the image, where the halo and the rock edge become sharp changes (average damage 0.17). Its structure score falls from 0.97 at full size to 0.72 at the coarsest scale.
- **Color is invisible to both.** Most of what people see (cyan sky, orange rock) is color, which neither metric measures. MS-SSIM is right partly by luck, because it catches the brightness side-effect of the color change.
- **Not a one-off.** The heatmap shows the same pattern for all color distortions, and with the original 2× downsampling SSIM's rank drops from 0.75 to 0.46.

### 4.2 A-2. LPIPS vs DISTS on photo 38 with jitter, level 4

![A-2](../results/figures/selected/A-2_images.png)
![A-2 heatmap](../results/figures/selected/A-2_heatmap.png)

| | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS | People |
|---|---|---|---|---|---|---|
| score | 24.20 dB | 0.923 | 0.983 | 0.171 | 0.300 | MOS 1.67 |
| rank r | 0.50 | 0.58 | 0.69 | **0.55** | **0.09** | 0.14 |

**What happened.** "Jitter" nudges pixels a few positions, so the jet's outline becomes ragged. LPIPS calls the photo average. DISTS and the people put it near the bottom. Both metrics use the same network, so **the formula is what differs. DISTS is right.**

**Brief.** LPIPS checks the picture spot by spot and averages. Most spots are sky and look fine, so the ragged outline gets diluted. DISTS asks "how much edge-like texture is in the whole picture?", and a jet whose every edge is ragged changes that answer a lot.

**Why.**
- **Where each formula finds the distance.** Most of DISTS's 0.300 comes from its *texture* term in the two deepest layers (0.109 and 0.111 + 0.041). These layers respond to edges and contours, and ragged edges change their overall activity. LPIPS spreads a smaller 0.171 over all layers, and its map shows the damage only along the outline, a small part of the image.
- **Small shifts were trained away.** LPIPS learned from people comparing small 64×64 patches, where tiny shifts barely matter.
- **The damage is on what people look at.** Every bit of it is on the jet, the main subject.
- **A type-wide effect.** Inside "jitter" alone, both metrics rank photos almost perfectly (0.94 and 0.96). LPIPS just gives all jittered photos too mild a score compared with other distortions (red jitter row in the heatmap).

### 4.3 B-1. A metric gets the rank wrong (SRCC), SSIM on photo 32 with color diffusion, level 2

![B-1](../results/figures/selected/B-1_images.png)
![B-1 MS-SSIM scales](../results/figures/selected/B-1_msssim_scales.png)
*MS-SSIM sees almost no damage at any zoom level (note the tiny color-bar range), because the yellow cast is pure color.*
![B-1 heatmap](../results/figures/selected/B-1_heatmap.png)

| | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS | People |
|---|---|---|---|---|---|---|
| score | 14.57 dB | **0.996** | 0.997 | 0.263 | 0.221 | MOS 1.77 |
| rank r | 0.09 | **0.98** | 0.94 | 0.39 | 0.25 | 0.16 |

**What happened.** This is the **biggest rank mistake of any metric on any photo.** SSIM says the photo is better than 98% of the dataset; people say it is worse than 84%. It is also the biggest score mistake, since SSIM's predicted human score is 4.33, the real one 1.77.

**Brief.** SSIM looks at the world in black and white. The whole photo turned yellow, but in black and white it looks the same.

**Why.**
- **Color blindness.** SSIM (and MS-SSIM) only measure brightness. Here brightness barely changes (171 → 166 on average), while color changes a lot (see the brightness vs color panels). The metric cannot see the damage.
- **Not a zoom problem.** With the original 2× downsampling SSIM still ranks the photo at 0.97, unlike in A-1.
- **The metrics that see color catch it.** PSNR (RGB pixels) ranks it 0.09, and LPIPS and DISTS (RGB features) rank it low too.
- **Systematic.** SSIM is badly wrong on 21% of color-diffusion photos, 22% of color-saturation photos and 47% of color-block photos (red rows in the heatmap). Inside color diffusion alone it still *orders* the photos well (0.875); it just rates all of them far too high.

### 4.4 B-2. A metric gets the score wrong (PLCC), DISTS on photo 81 with non-eccentricity patch, level 5

![B-2](../results/figures/selected/B-2_images.png)
![B-2 heatmap](../results/figures/selected/B-2_heatmap.png)

| | PSNR | SSIM | MS-SSIM | LPIPS-VGG | DISTS | People |
|---|---|---|---|---|---|---|
| score | 27.54 dB | 0.930 | 0.956 | 0.103 | **0.051** | MOS 1.33 |
| rank r | 0.71 | 0.60 | 0.44 | 0.68 | **0.79** | 0.05 |
| predicted MOS | | | | | **3.79** | 1.33 |

**What happened.** DISTS predicts a human score of 3.79 ("good"); people gave 1.33 ("bad"). That is 2.5 points wrong on a 1–5 scale. (The biggest score mistake overall is B-1's photo, so we show the next one.) Every metric is too generous here, and DISTS the most.

**Brief.** Cut small squares out of a photo and paste each one slightly off. The photo still has the same colors and the same textures, just in slightly wrong places. DISTS mostly checks *what* is in the picture, not *where*, so it barely notices. People immediately see the broken edges.

**Why.**
- **Small, scattered damage.** Many small patches are moved a little; most of the photo is untouched (pixel-change panel).
- **Whole-image statistics.** DISTS compares feature averages and correlations over the whole image, and moving patches inside the same image barely changes those. Every DISTS term is tiny (total 0.051). DISTS was designed to tolerate re-arranged texture, and here that tolerance is the wrong call.
- **A spot-by-spot metric notices.** LPIPS compares position by position, finds twice the distance (0.103), and its map lights up exactly at the moved patches.
- **Where the damage lands.** The breaks fall on a person, near the face, on the hair, the phone and the cup edges, where people look closely.
- **Mostly a one-off.** Only 1.2% of photos of this type fool DISTS this badly. On this photo's heatmap row, levels 2 and 5 are dark red while level 3 is fine, so it depends on where the patches land.

### 4.5 C-1. SSIM and LPIPS order two versions of photo 43 oppositely

![C-1](../results/figures/selected/C-1_images.png)
![C-1 heatmap](../results/figures/selected/C-1_heatmap.png)

| Photo | Damage | SSIM | LPIPS-VGG | People (MOS) | SSIM's rank* | LPIPS's rank* | People's rank* |
|---|---|---|---|---|---|---|---|
| i | color saturation, L5 | 0.978 | 0.703 | 1.87 | **0.87** | **0.01** | 0.08 |
| j | contrast change, L1 | 0.780 | 0.092 | 3.80 | **0.32** | **0.71** | 0.62 |

\*Rank among the 125 damaged versions of this same leaf photo.

**What happened.** SSIM prefers the purple-gray leaf (i). LPIPS prefers the high-contrast leaf (j). People prefer j by almost 2 points, so **LPIPS is right.** Over all 78,769 such SSIM vs LPIPS conflicts, people side with LPIPS 43% of the time, with SSIM 24%, and the rest are ties.

**Brief.** SSIM is color-blind *and* worries too much about dark corners. It does not notice the leaf turned purple, but it panics because the blurry dark background became darker.

**Why.**
- **Photo i, color.** The leaf turns lavender-gray, but its veins and water drops (the structure) are intact, so SSIM, which sees only brightness, scores it 0.978. The color panel shows the huge color change; LPIPS sees color and ranks this the worst version of the leaf.
- **Photo j, dark background.** The leaf itself looks "enhanced", and the SSIM map is almost black on it. **80% of SSIM's damage is in the dark, blurry background**, which goes from brightness 43 to 14. SSIM's brightness term is *relative*, so the same drop counts far more in a dark area than in a bright one. People hardly care about a darker, out-of-focus background.
- **Systematic.** Across all photos (left heatmap) SSIM is more generous than LPIPS on color distortions (red) and harsher on brightness and contrast changes (blue).

## 5. Discussion

**Brief.** Each metric is a different kind of judge. PSNR counts pixel mistakes. SSIM checks structure but is color-blind and short-sighted. MS-SSIM steps back to see the big picture. LPIPS looks closely, spot by spot. DISTS looks at the overall texture. Each is good at some damage and fooled by other damage, and none of them judges like a person every time.

**What each metric is good at.**

| Metric | Good at | Why |
|---|---|---|
| PSNR | brightness changes (best in that group, 0.900) | a brightness change is literally a pixel-value change |
| SSIM / MS-SSIM | blur, compression, noise | these destroy structure, which is what SSIM measures; MS-SSIM's zoom levels make it the best metric overall |
| LPIPS / DISTS | spatial damage (pixelation, jitter, moved patches) | deep features tolerate small misalignments that wreck pixel comparisons |

**Where they fail.**

| Case | Metric that fails | What it misses | One-off or pattern? |
|---|---|---|---|
| A-1 | SSIM | big, smooth changes (looks through one small window only) | pattern (color types) |
| A-2 | LPIPS | ragged edges, diluted by averaging spot by spot | pattern (jitter, color shift) |
| B-1 | SSIM, MS-SSIM | **color**, since they only see brightness | pattern (21–47% of color photos) |
| B-2 | DISTS | **where** the damage is, since whole-image statistics ignore small moved patches | one-off |
| C-1 | SSIM | color (ignored); darkening of unimportant dark areas (over-counted) | pattern |

**The biggest weakness shared by all metrics is comparing different kinds of damage.** Every metric ranks photos well *within* one distortion type. They fail when a blur has to be compared with a color shift. A metric can order the five levels of one distortion perfectly and still put that whole distortion at the wrong quality.

**Does any single metric consistently reflect human perception? No.**
- The best overall metric, MS-SSIM (0.83), still fails on color saturation (0.49), moved patches (0.44) and pixelation (0.48).
- The learned metrics win exactly on those, but lose to MS-SSIM overall, and the stronger one (DISTS) was trained on this very dataset.
- Which metric is best changes with the type of damage (Section 3 heatmaps).

**A perceptual factor the metrics do not capture well is color.**
- SSIM and MS-SSIM cannot see it at all (cases B-1, C-1).
- The deep metrics see it but do not weigh it consistently against other damage. Every metric scores 0.48–0.74 on the color group, against at least 0.89 on blur.

A second factor is **where people look**. Damage on a face or the main subject matters more than damage in a blurry background (cases A-2, B-2, C-1), but every metric treats all parts of the image the same.


## References
- H. Lin, V. Hosu, D. Saupe, "KADID-10k: A large-scale artificially distorted IQA database," QoMEX 2019.
- Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," IEEE TIP 2004.
- Z. Wang, E. P. Simoncelli, A. C. Bovik, "Multiscale structural similarity for image quality assessment," Asilomar 2003.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," CVPR 2018.
- K. Ding, K. Ma, S. Wang, E. P. Simoncelli, "Image quality assessment: unifying structure and texture similarity," IEEE TPAMI 2022.
- H. R. Sheikh, M. F. Sabir, A. C. Bovik, "A statistical evaluation of recent full reference image quality assessment algorithms," IEEE TIP 2006 (logistic mapping).
- C. Chen, J. Mo, "IQA-PyTorch (pyiqa)," https://github.com/chaofengc/IQA-PyTorch.
- MR-IQA: Mask Reference Image Quality Assessment, https://arxiv.org/pdf/2302.13770.
- A comprehensive evaluation of full-reference IQA algorithms on KADID-10k, https://arxiv.org/pdf/1907.02096.
- JND-SalCAR, https://arxiv.org/pdf/1902.05316.
- PIQ documentation, https://piq.readthedocs.io/en/latest/overview.html.
