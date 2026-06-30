# Study: `tiny_cnn_architecture.png` (TinyCNN) vs. L3.7 Slide 12 (Single-Perceptron Forward Pass)

> **Author:** [Opus 4.8] · **Date:** 2026-06-29
> **Scope:** Technical review and reconciliation of two teaching artifacts that sit at opposite ends of the
> same abstraction ladder — **L3.7 slide 12** "Single Perceptron Forward Pass" (`docs/Lesson 3.7 - Neural
> Networks & Deep Learning.pdf`, the *atomic* neuron) and **`docs/tiny_cnn_architecture.png`** (the *whole*
> TinyCNN, a multi-layer CNN). All numbers re-derived from `notebooks/03_first_cnn.ipynb` and the slide.
>
> Companion docs: `cnn-architecture-convolution-study.md` (PNG vs. L3.8 slide 12, the *convolution operation*)
> and `lesson-3.8-slide-review.html` (full L3.8 deck review).

---

## 0. Ground truth — the two artifacts

### A. L3.7 Slide 12 — one perceptron (the atom)
A single neuron computing a binary "checkout PASS/FAIL" decision from 3 tabular features (the L07 NorthStar story):

| Input | Value | Weight |
|---|---|---|
| X₁ Session Duration | 320 s | W₁ = 0.01 |
| X₂ Total Clicks | 8 | W₂ = 0.50 |
| X₃ Cart Value | $75 | W₃ = −0.02 |
| Bias | — | b = −0.5 |

```
z = (320×0.01) + (8×0.50) + (75×−0.02) + (−0.5)
  =   3.2      +   4.0     +   (−1.5)    +  (−0.5)   = 5.2          ✓ verified
Step:  f(z) = 1 if z ≥ 3.0 else 0   →   5.2 ≥ 3.0   →   OUTPUT = 1 (PASS)   ✓ verified
```
- **Parameters:** 3 weights + 1 bias = **4**.
- **Activation:** step (threshold) function.
- **Task:** binary classification on tabular features.

### B. PNG — TinyCNN (the whole network)
Verified against `notebooks/03_first_cnn.ipynb`:

```
1×28×28 image
 └ Conv2d(1,16,k3,p1)+ReLU+MaxPool2 → 16@28→16@14
 └ Conv2d(16,32,k3,p1)+ReLU+MaxPool2 → 32@14→32@7
 └ Flatten(1568) → Linear(1568,64)+ReLU → Linear(64,10) → 10 logits
```
- **Parameters:** 160 + 4,640 + 100,416 + 650 = **105,866**.
- **Activation:** ReLU (hidden) + raw logits → softmax at loss.
- **Task:** 10-class image classification (Fashion-MNIST *clothing*).

> **The key fact that links them:** the PNG's `Dense(64)` layer is **64 perceptrons** (100,416 params) and its
> `Dense(10)` output is **10 perceptrons** (650 params) — each one is exactly the slide-12 neuron, only with a
> ReLU/identity activation instead of a step. **One perceptron is the brick; TinyCNN is the building.**

---

## 1. Concerns — `tiny_cnn_architecture.png`

(Structurally faithful: channel growth `1→16→32`, spatial shrink `28→14→7`, `1568` flatten, `Dense(64)→Dense(10)`
all match the code and the 105,866 param count.)

| # | Severity | Issue | Fix |
|---|---|---|---|
| 1a | **Medium** | Input is a handwritten **"7"** and outputs are labelled **"0–9"** → implies MNIST *digit* recognition, but TinyCNN classifies **Fashion-MNIST clothing** (`T-shirt/top … Ankle boot`). | Use a clothing sample (the Coat) + clothing-class labels. |
| 1b | Low–Med | Conv boxes say "Conv 3×3, stride 1" but omit **`padding=1`** — the very parameter that keeps the map at 28×28 (without it, 3×3/s1 → 26×26). | Annotate `Conv 3×3, stride 1, padding 1 (same)`. |
| 1c | Low | Stray output label "**0-6**"; duplicated "Block 2" caption. | Clean to "10 logits"; de-duplicate. |

---

## 2. Concerns — L3.7 Slide 12 (single perceptron)

The slide's **arithmetic is correct** (`z = 5.2`, output 1). Three conceptual concerns:

### 2a. ⚠️ Bias and threshold roles are conflated (Medium)
The slide folds a bias `b = −0.5` into `z` **and** applies a step threshold at **3.0**. The standard
perceptron convention puts the **threshold at 0** and lets the **bias be the (learnable) threshold**:
`y = 1 iff w·x + b ≥ 0`. Using *both* a bias and a non-zero threshold is **redundant** and non-standard —
the effective decision here is `w·x ≥ 3.5`, so the bias of −0.5 is doing nothing a student can interpret.
- **Why it matters:** modern frameworks (PyTorch) have a bias but **no separate threshold** — the activation
  is applied at 0. A learner who memorises "threshold = 3.0" will be confused by every real neuron afterward.
- **Fix:** either (i) keep the bias and step at **0** (choose `b` so the boundary is meaningful), or
  (ii) drop the bias and use a threshold θ. Not both. Option (i) is preferred — it teaches *bias = threshold*.

### 2b. Step activation is non-differentiable (Low)
The step/threshold function has **zero gradient** almost everywhere (undefined at the jump), so it **cannot be
trained by gradient descent / backprop**. It's the historical Rosenblatt perceptron. Every trainable network
in L07–L08 (including TinyCNN in the PNG) uses **ReLU/sigmoid/softmax** *because* of this. Worth a one-line note
so students don't think real nets threshold-and-step.

### 2c. Values are illustrative, not learned (Low)
The weights/bias/threshold are hand-picked round numbers. In practice they are **learned** by gradient descent.
Not wrong, but unstated; a caption ("illustrative values; in training these are learned") would help.

---

## 3. Are the PNG and slide 12 "the same thing"? — No: atom vs. organism

They are the **two ends of the same abstraction ladder**, not duplicates.

| Dimension | **L3.7 Slide 12** (one perceptron) | **PNG** (TinyCNN) |
|---|---|---|
| **Level** | MICRO — a single neuron | MACRO — a full multi-layer network |
| **Unit count** | 1 neuron | thousands of neurons across 4 weight layers |
| **Parameters** | **4** | **105,866** |
| **Input modality** | tabular (3 numeric features) | image (1×28×28 = 784 pixels) |
| **Core op** | one weighted sum + bias | convolutions + pooling + dense layers |
| **Weights** | independent per input (fully connected) | **shared** in conv layers; independent in FC layers |
| **Activation** | **step** (non-differentiable) | **ReLU** + softmax (trainable) |
| **Task / output** | binary PASS/FAIL (1 value) | 10-class logits |
| **Trained by** | perceptron rule (historical) | gradient descent + backprop |

**The relationship (precise):**
- Each node in the PNG's `Dense(64)` and `Dense(10)` columns **is** a slide-12 perceptron — same `z = Σwᵢxᵢ + b`
  — differing only in the activation (ReLU/identity vs step).
- A **convolution filter** is *also* a perceptron, but restricted to a small **shared** receptive field
  (a 3×3 patch, the same weights reused at every location) — that weight-sharing is the one structural idea the
  bare perceptron does **not** show.
- So slide 12 explains the **arithmetic of a single unit**; the PNG explains **how thousands of those units are
  wired together** (and how conv layers share them) into an image classifier.

---

## 4. Consolidation — "From one neuron to a CNN" (a clean L07→L08 bridge)

Because the perceptron is literally the unit inside the PNG's dense layers, a single nested figure works well:

```
┌──────────────────────  FROM ONE NEURON → THE WHOLE TinyCNN  ───────────────────────┐
│                                                                                     │
│  MICRO: one perceptron (L3.7 s12)      MESO: a layer = many neurons    MACRO: TinyCNN│
│  ┌───────────────────────────┐         ┌───────────────┐                            │
│  │ x1─w1┐                     │         │ ● ● ● … (64)  │   Coat 28×28×1             │
│  │ x2─w2┼─Σ(+b)─►f(·)─► y     │  ×64 →  │ each = the    │   → [Conv p1+ReLU+Pool]×2  │
│  │ x3─w3┘   z=Σwx+b           │         │ micro neuron  │   → Flatten 1568           │
│  └───────────────────────────┘         └───────────────┘   → Dense64 (64 neurons)   │
│                                                            → Dense10 (10 neurons)    │
│  activation:  step → ReLU/softmax (trainable)              → 10 clothing logits      │
│  weights:     independent → SHARED in conv (3×3 slides everywhere)                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### What the consolidated figure must reconcile
1. **Activation.** Replace the slide's **step** with **ReLU** (hidden) / **softmax** (output), and state *why*:
   step has no usable gradient, so trainable nets need smooth/piecewise-linear activations.
2. **Weight sharing.** Show that FC neurons have independent weights (like the perceptron), but **conv filters
   reuse one small weight set across the whole image** — the new idea L08 adds on top of the L07 neuron.
3. **Modality & scale.** Tabular 3-vector (perceptron) → 784-pixel image (CNN); 4 params → 105,866 params.
4. **Bias/threshold.** Adopt the standard convention (bias absorbs the threshold; activation at 0) so the micro
   unit matches what the macro network actually uses in PyTorch.

### Suggested caption
> *"A neuron computes `z = Σ wᵢxᵢ + b` then applies an activation (left). Stack 64 of them and you get a dense
> layer; stack dense layers on top of weight-shared convolution layers and you get TinyCNN — 105,866 of these
> same weighted sums, trained end-to-end to map a 28×28 clothing image to 10 class scores."*

**Feasibility:** high. It's one three-panel "zoom-out" diagram. Pedagogically it's the cleanest possible bridge
from L07 (the neuron) to L08 (the CNN) — it answers "where did the perceptron go?" by pointing at the FC columns.

---

## 5. Consolidated recommendations

| # | Artifact | Severity | Issue | Fix |
|---|---|---|---|---|
| 1 | PNG | **Medium** | Digit "7"/"0–9" implies MNIST; TinyCNN classifies clothing | Fashion-MNIST sample + clothing labels |
| 2 | Slide 12 | **Medium** | Bias + non-zero threshold both present (redundant/non-standard) | Step at 0; let bias be the threshold |
| 3 | PNG | Low–Med | `padding=1` omitted from conv annotation | Annotate `padding 1 (same)` |
| 4 | Slide 12 | Low | Step is non-differentiable (can't train by GD) | Note ReLU/sigmoid are used in real nets |
| 5 | PNG | Low | Stray "0-6" label; duplicated Block-2 caption | Clean labels |
| 6 | both | Medium | Different abstraction levels confuse if shown apart | Merge as a "neuron → CNN" zoom figure (§4) |

**Priority:** Slide-12 #2 (bias/threshold) and PNG #1 (semantics) are the two real correctness/clarity issues;
the consolidation #6 is the high-value teaching improvement.

---

## Appendix A — verified arithmetic & the parameter ladder
```
Perceptron z = 320(0.01) + 8(0.5) + 75(-0.02) + (-0.5) = 3.2 + 4.0 - 1.5 - 0.5 = 5.2 ; 5.2≥3.0 → 1   ✓
Perceptron params: 3 + 1 = 4
TinyCNN Dense(64): 1568·64 + 64 = 100,416   (= 64 perceptrons, each 1568 inputs)
TinyCNN Dense(10):   64·10 + 10 =     650    (= 10 perceptrons, each 64 inputs)
TinyCNN total:                    105,866
```

## Appendix B — the one equation, at both scales
```
single neuron :  y = f( Σ_i w_i x_i + b )                      (slide 12; f = step)
dense layer   :  y = f( W x + b )         W is (units × in)     (PNG Dense(64), Dense(10); f = ReLU/identity)
conv layer    :  y[c,i,j] = f( Σ K_c · patch(i,j) + b_c )      (PNG conv; K_c shared across all i,j)
```

## Appendix C — sources cross-checked
- `docs/Lesson 3.7 - Neural Networks & Deep Learning.pdf` slide 12 (perceptron forward pass), slides 13–14 (MLP, NN-in-a-nutshell).
- `notebooks/03_first_cnn.ipynb` — TinyCNN definition, param count, shape trace.
- `notebooks/01_monday_morning.ipynb` — FlatMLP (the bridge: dense layers of neurons on flattened pixels).
- `lesson.md`, `reference.md` — course narrative and glossary.
- Companion: `docs/cnn-architecture-convolution-study.md`, `docs/lesson-3.8-slide-review.html`.
