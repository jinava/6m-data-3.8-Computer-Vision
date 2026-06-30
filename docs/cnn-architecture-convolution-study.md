# Study: `tiny_cnn_architecture.png` vs. Slide 12 — convolution operation ↔ TinyCNN architecture

> **Author:** [Opus 4.8] · **Date:** 2026-06-29
> **Scope:** Technical review and reconciliation of two L08 (Computer Vision) teaching artifacts —
> `docs/tiny_cnn_architecture.png` (the full TinyCNN architecture diagram) and **slide 12** of
> `docs/Lesson 3.8 - Computer Vision.pdf` (the "Convolution Operation" worked example).
> All numbers below were re-derived from the actual model in `notebooks/03_first_cnn.ipynb`.

---

## 0. Ground truth — what TinyCNN actually is

From `notebooks/03_first_cnn.ipynb` (verified by running the model):

```python
class TinyCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # block 1
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # block 2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),   # 1568 -> 64
            nn.Linear(64, n_classes),               # 64 -> 10
        )
```

**Trained on Fashion-MNIST** — 10 *clothing* classes:
`['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']`
(**not** the MNIST handwritten digits 0–9).

### Exact shape trace (input `1×28×28`)

| Stage | Op | Output shape | Why |
|---|---|---|---|
| input | — | `1 × 28 × 28` | grayscale |
| block 1 | `Conv2d(1,16,k3,**p1**)` | `16 × 28 × 28` | **padding=1 keeps 28** |
| | `ReLU` | `16 × 28 × 28` | — |
| | `MaxPool2d(2)` | `16 × 14 × 14` | halves H,W |
| block 2 | `Conv2d(16,32,k3,**p1**)` | `32 × 14 × 14` | padding=1 keeps 14 |
| | `ReLU` | `32 × 14 × 14` | — |
| | `MaxPool2d(2)` | `32 × 7 × 7` | halves H,W |
| head | `Flatten` | `1568` | 32·7·7 = 1568 |
| | `Linear(1568,64)+ReLU` | `64` | — |
| | `Linear(64,10)` | `10` | logits |

### Exact parameter breakdown (total **105,866**)

| Layer | Formula | Params |
|---|---|---|
| `Conv2d(1,16,k3)` | 16·(1·3·3) + 16 | **160** |
| `Conv2d(16,32,k3)` | 32·(16·3·3) + 32 | **4,640** |
| `Linear(1568,64)` | 1568·64 + 64 | **100,416** |
| `Linear(64,10)` | 64·10 + 10 | **650** |
| **Total** | | **105,866** |

> Note: ~95% of the parameters live in the **first Linear layer** (the flatten→64 dense), not the conv
> layers — a useful thing to point out when discussing where CNN parameters actually go.

---

## 1. Concerns — `tiny_cnn_architecture.png`

The diagram is **structurally correct**: the two conv blocks, the channel growth `1→16→32`, the spatial
shrink `28→14→7`, the `1568` flatten, `Dense(64)+ReLU`, and `Dense(10)` logits all match the code exactly.
Three issues remain:

### 1a. ⚠️ Semantic mismatch — digit input, but TinyCNN classifies clothing (Medium)
- The input image is a handwritten **"7"** and the output neurons are labelled **"0–9"** — i.e. it depicts
  **MNIST digit recognition**.
- The actual TinyCNN in this course is trained on **Fashion-MNIST** (clothing). The architecture is
  identical (28×28×1 in, 10 classes out), but the *semantics* are wrong: it implies the network outputs
  digit classes when it really outputs `T-shirt/top … Ankle boot`.
- **Fix:** use a Fashion-MNIST sample (e.g. the Coat used throughout NB 02/03) as the input thumbnail, and
  label the output `0 = T-shirt … 9 = Ankle boot` (or just "10 clothing classes").

### 1b. ⚠️ `padding=1` is shown in the dimensions but omitted from the annotation (Low–Medium)
- The Conv boxes are annotated **"Conv 3×3 (16 filters, stride 1)"** with **no mention of padding**.
- Yet the diagram correctly shows the output staying **28×28**. That is only true **because `padding=1`**.
  Without padding, `Conv 3×3, stride 1` on `28×28` yields `26×26` (`(28−3)/1+1`).
- So the picture is right but the label is **incomplete**, and it omits the *exact* parameter that makes the
  size-preserving behaviour hold — which is confusing next to slide 12, where padding=0 makes the size
  shrink. **Fix:** annotate `Conv 3×3, stride 1, padding 1` (`same` padding).

### 1c. Cosmetic clutter (Low)
- The output column shows stray labels **"0-9", "0-6", "0-9"** next to neurons — the "0-6" in particular is
  meaningless and looks like an auto-layout artifact.
- The **Block 2** caption is **duplicated** ("Block 2 (Convolutional Block)" appears twice) and an
  "Activation" label is loosely placed.
- **Fix:** clean the output labels to a single "10 logits (one per class)" and de-duplicate the Block-2 caption.

> **Verdict on the PNG:** technically faithful to the model's *shapes and parameters*; the only real
> correctness issue is the **digit-vs-clothing semantics** (1a). The padding annotation (1b) is an accuracy
> *completeness* gap, not an error.

---

## 2. Concerns — Slide 12 ("Convolution Operation")

Slide 12 shows a single numerical convolution: **Input 6×6, Kernel 3×3, Padding 0, Stride 1 → Output 4×4**,
filter `[[1,0,-1],[1,0,-1],[1,0,-1]]`, first output value `−5`.

- **Arithmetic verified correct.** Top-left 3×3 patch `[[3,0,1],[1,5,8],[2,7,2]]` · filter
  `= (3+1+2)·1 + (0+5+7)·0 + (1+8+2)·(−1) = 6 − 11 = −5`. ✓
- **Output-size formula correct:** `⌊(6 − 3 + 2·0)/1⌋ + 1 = 4`. ✓
- It is a **single-channel, single-filter, padding-0** illustration of the *operation* (slide → multiply →
  sum → one output pixel).
- The one teaching caveat (covered in the separate slide-deck review): the filter is a vertical-edge
  detector, and the deck's edge-kernel slide labels this family inconsistently with `02_convolutions_intuition.ipynb`.
  Slide 12 itself is **numerically correct**.

---

## 3. Are the PNG and slide 12 "the same thing"? — No: micro vs. macro

They sit at **two different levels of abstraction** and are **complementary**, not duplicates.

| Dimension | **Slide 12** (operation) | **PNG** (architecture) |
|---|---|---|
| **What it shows** | How *one* convolution computes *one* output value | How the *whole* TinyCNN is assembled end-to-end |
| **Level** | MICRO — the arithmetic inside a Conv box | MACRO — the stack of Conv/Pool/FC layers |
| **Input** | one toy `6×6` grid | one real `28×28` image |
| **Channels** | 1 in → **1 filter** → 1 feature map | 1 → **16** → **32** feature maps |
| **Padding** | **0** (size shrinks `6→4`) | **1** (size preserved `28→28`) |
| **Stride** | 1 | 1 (conv), 2 (pool) |
| **Output** | a single `4×4` feature map | `10` class logits |
| **Purpose** | teach the *mechanics* of convolution | teach the *composition* of a CNN |

**The relationship:** slide 12 is the **zoom-in** of a single "Conv 3×3" box in the PNG. The PNG's first
conv layer runs **16 instances** of the slide-12 operation in parallel (one per filter), and uses
**padding=1** instead of 0 — so each produces a `28×28` map and there are 16 of them (`16@28×28`).

> ⚠️ **The trap to flag:** the two artifacts use **different padding conventions** (0 vs 1). A learner who
> internalises slide 12 ("conv shrinks the image: 6→4") and then sees the PNG ("conv keeps 28→28") will be
> confused unless padding is made explicit in both. This is the single most important thing to reconcile.

---

## 4. Consolidation — can they be merged into one figure? Yes.

Because slide 12 is literally the operation inside the PNG's Conv box, a single **"From one convolution to
the whole TinyCNN"** figure can nest the micro view inside the macro view. Proposed design:

```
┌──────────────────────────  FROM ONE CONVOLUTION → THE WHOLE TinyCNN  ──────────────────────────┐
│                                                                                                 │
│  MICRO (zoom-in of ONE filter)                MACRO (full TinyCNN architecture)                 │
│  ┌───────────────────────────┐                                                                  │
│  │  3×3 patch    kernel       │       Coat        Block 1            Block 2        head         │
│  │  ┌─────┐    ┌─────┐        │     28×28×1   Conv3×3,p1 →16    Conv3×3,p1 →32   Flatten 1568    │
│  │  │ ··· │ ⊙  │ ··· │ = Σ →●│ ───►  ┌──┐    ReLU, Pool2 →     ReLU, Pool2 →     →Dense64+ReLU   │
│  │  └─────┘    └─────┘  one   │       │##│    16@28→16@14       32@14→32@7        →Dense10        │
│  │  slide·multiply·sum  pixel │       └──┘                                          ↓            │
│  │  ×16 filters → 16 maps     │                                              10 clothing logits  │
│  └───────────────────────────┘                                                                  │
│                                                                                                 │
│  Output size (both views):   out = ⌊ (in − k + 2p) / s ⌋ + 1                                     │
│     • slide-12 teaching example:  p=0 →  (28−3+0)/1+1 = 26   (shrinks)                           │
│     • TinyCNN actually uses:      p=1 →  (28−3+2)/1+1 = 28   (size preserved) ← what the PNG shows│
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### What the consolidated figure must reconcile

1. **Padding (the key fix).** Show the formula once and instantiate it both ways: `p=0 → 28→26` (the
   slide-12 "shrink" intuition) and `p=1 → 28→28` (what TinyCNN uses). This removes the 6→4-vs-28→28
   contradiction.
2. **Channels.** Make explicit that the micro view is **one filter → one feature map**, and TinyCNN's first
   conv has **16 filters → 16 feature maps** (`out_channels = #filters`). The slide-12 single map is one
   slice of the PNG's `16@28×28` stack.
3. **Scale.** Re-cast the micro example on a **28×28** input (or clearly label it as a 6×6 *teaching* toy)
   so the two halves share a coordinate system.
4. **Semantics.** Use a **Fashion-MNIST** input (a Coat) and label the 10 outputs as clothing classes — not
   a digit "7" / "0–9".
5. **Pooling.** Add the pool formula too (`MaxPool2d(2)` → `28→14`) so the full size story
   `28 → 28 → 14 → 14 → 7` is derivable from the same formula family.

### Suggested single-figure caption
> *"A convolution slides a small kernel over the image, multiplying and summing to produce one output pixel
> (left). TinyCNN stacks this operation: 16 such kernels (padding 1, so 28×28 is preserved) → ReLU →
> 2×2 max-pool (→14×14), then 32 kernels → pool (→7×7), flatten to 1568, and two dense layers to 10
> clothing-class logits."*

**Feasibility:** straightforward — it is one diagram with a "detail callout." The only content work is
choosing a single padding story (show both via the formula) and fixing the digit→clothing semantics.

---

## 5. Consolidated recommendations

| # | Artifact | Severity | Issue | Fix |
|---|---|---|---|---|
| 1 | PNG | **Medium** | Input "7" + "0–9" outputs imply digit recognition; TinyCNN classifies clothing | Use a Fashion-MNIST sample + clothing-class labels |
| 2 | PNG | Low–Med | Conv annotation omits `padding=1` (the param that preserves 28×28) | Annotate `Conv 3×3, stride 1, padding 1 (same)` |
| 3 | PNG | Low | Stray "0-6" output label; duplicated Block-2 caption | Clean to "10 logits"; de-duplicate |
| 4 | Slide 12 | — | Numerically correct (`−5`, output 4) | none |
| 5 | PNG ↔ Slide 12 | Medium | **Padding-convention clash** (0 vs 1) confuses learners | Reconcile via the output-size formula in a merged figure |

**Priority order:** PNG #1 (semantics) → the padding reconciliation #5 → PNG #2 → cosmetics #3.

---

## Appendix A — output-size formula (used throughout)

```
out = floor( (in − kernel + 2·padding) / stride ) + 1
```

| Case | in | k | p | s | out |
|---|---|---|---|---|---|
| Slide 12 (conv, p0) | 6 | 3 | 0 | 1 | **4** |
| TinyCNN conv (p1) | 28 | 3 | 1 | 1 | **28** |
| TinyCNN pool | 28 | 2 | 0 | 2 | **14** |
| TinyCNN conv2 (p1) | 14 | 3 | 1 | 1 | **14** |
| TinyCNN pool2 | 14 | 2 | 0 | 2 | **7** |

## Appendix B — sources cross-checked
- `notebooks/03_first_cnn.ipynb` — TinyCNN definition, param count (105,866), shape trace.
- `notebooks/02_convolutions_intuition.ipynb` — the convolution operation by hand; channels = #filters.
- `lesson.md` — Q4 output-shape tracing (1568 flatten), key takeaways.
- `docs/Lesson 3.8 - Computer Vision.pdf` slide 12 (convolution arithmetic) and slides 14–16
  (padding/stride/pooling formulas).
- `docs/lesson-3.8-slide-review.html` — companion review of the full slide deck (the Q3 / Sobel-label issues).
