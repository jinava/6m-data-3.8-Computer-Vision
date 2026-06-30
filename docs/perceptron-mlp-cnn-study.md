# Study: Neuron → MLP → CNN — `tiny_cnn_architecture.png` vs. L3.7 slides 11 & 12

> **Author:** [Opus 4.8] · **Date:** 2026-06-29
> **Scope:** Review and reconciliation of three teaching artifacts that form a single abstraction ladder:
> **L3.7 slide 11** "Single Perceptron Forward Pass" (the *neuron*), **L3.7 slide 12** "Multi-Layer Perceptron"
> (the *MLP*), and **`docs/tiny_cnn_architecture.png`** (TinyCNN, the *CNN*).
> All numbers re-derived from `notebooks/03_first_cnn.ipynb`, `notebooks/01_monday_morning.ipynb`, and the slides.
>
> Supersedes the 2-tier `perceptron-to-cnn-study.md`; companion to `cnn-architecture-convolution-study.md`
> (PNG vs. L3.8 convolution-operation slide) and `lesson-3.8-slide-review.html`.

---

## 0. Ground truth — the three rungs

### Rung 1 — L3.7 Slide 11: one perceptron (the atom)
NorthStar checkout PASS/FAIL from 3 tabular features:

```
inputs  X = (320s, 8, $75)     weights W = (0.01, 0.50, -0.02)     bias b = -0.5
z = 320(0.01) + 8(0.50) + 75(-0.02) + (-0.5) = 3.2 + 4.0 - 1.5 - 0.5 = 5.2        ✓
step:  f(z) = 1 if z ≥ 3.0 else 0   →   5.2 ≥ 3.0   →   OUTPUT 1 (PASS)            ✓
```
Params: **4** (3 weights + 1 bias). Activation: **step**. Task: binary, tabular.

### Rung 2 — L3.7 Slide 12: the MLP (layers of neurons)
"Inside the Neural Network: How AI sees shapes":
- **30×30 = 900-pixel** input (a triangle) → flattened to 900 signals.
- Hidden layer(s) of neurons with **ReLU**; **softmax** output over 3 shapes (Circle 10% / **Triangle 85%** / Square 5%).
- Claim: *"a single hidden layer with enough neurons + non-linear activations can approximate ANY continuous
  function arbitrarily well; in practice 2–3 hidden layers train better than one very wide layer."* (Universal
  Approximation Theorem.)
- This is structurally **NB 01's `FlatMLP`** (`784 → 256 → 128 → 10`, **235,146** params).

### Rung 3 — PNG: TinyCNN (the CNN), verified from `03_first_cnn.ipynb`
```
1×28×28  → [Conv2d(1,16,k3,p1)+ReLU+MaxPool2] → 16@14×14
         → [Conv2d(16,32,k3,p1)+ReLU+MaxPool2] → 32@7×7
         → Flatten(1568) → Dense(64)+ReLU → Dense(10) → 10 logits
```

| Part | Params | What it is |
|---|---|---|
| Conv **body** (`160 + 4,640`) | **4,800** | the **new L08 idea**: weight-shared *local* neurons (locality + translation invariance) |
| MLP **head** (`100,416 + 650`) | **101,066** | `Flatten → Dense64 → Dense10` — **this is an MLP** (= rung 2) on conv features |
| **Total** | **105,866** | trained on **Fashion-MNIST clothing** (10 classes) |

> **The spine of this whole study:** rung 1 is the unit, rung 2 stacks units into an MLP, and rung 3's TinyCNN
> is *"rung 2's MLP with a convolutional feature-extractor bolted in front."* Slide 12's MLP **reappears
> verbatim** as the PNG's head (`Flatten 1568 → Dense64 → Dense10`).

---

## 1. Concerns

### 1A. PNG (TinyCNN) — structurally faithful; three issues
| # | Severity | Issue | Fix |
|---|---|---|---|
| a | **Medium** | Input "7" + "0–9" labels imply MNIST *digits*; TinyCNN classifies Fashion-MNIST *clothing* | Coat sample + clothing-class labels |
| b | Low–Med | Conv annotation omits **`padding=1`** (the param that keeps 28×28; without it → 26×26) | Annotate `Conv 3×3, stride 1, padding 1 (same)` |
| c | Low | Stray "0-6" output label; duplicated Block-2 caption | Clean to "10 logits"; de-duplicate |

### 1B. Slide 11 (perceptron) — arithmetic correct; conceptual issues
- **(Medium) Bias & threshold conflated.** A bias `b=−0.5` is folded into `z` **and** a separate step threshold
  of **3.0** is applied. Standard practice steps at **0** and lets the **bias be the threshold**
  (`y = 1 iff w·x + b ≥ 0`). Both together is redundant and non-standard; the effective rule is `w·x ≥ 3.5`,
  so the bias is uninterpretable. PyTorch neurons have a bias but **no** separate threshold — students will trip.
- **(Low) Step is non-differentiable** → cannot be trained by gradient descent/backprop; it's the historical
  Rosenblatt perceptron. Every trainable net here (incl. TinyCNN) uses ReLU/softmax *because* of this.
- **(Low) Values illustrative, not learned** — unstated.

### 1C. Slide 12 (MLP) — mostly sound; terminology & consistency issues
- **(Medium) "Threshold functions like ReLU."** ReLU is **not** a threshold/step function — it is a *rectifier*
  (`max(0,x)`, piecewise-linear, differentiable almost everywhere). Calling it a "threshold function" one slide
  after the *actual* step function (slide 11) blurs the crucial contrast: **step ⇒ untrainable; ReLU ⇒ trainable.**
  The sub-label **"x>0 → non-linearity"** is also imprecise — ReLU is *linear* for x>0 (slope 1); the
  non-linearity is the **kink at 0**.
- **(Medium) Inconsistent with slide 11.** Slide 12 says *"biases adjust the activation threshold for each
  neuron"* — i.e. **bias = threshold**, the correct modern view — which **contradicts** slide 11's separate
  bias + 3.0 threshold. The two consecutive slides teach two different threshold mechanics.
- **(Low) Universal-approximation caveat.** "Approximate ANY continuous function arbitrarily well" should add
  *"on a bounded domain"* and note it is an **existence** result (may need exponentially many neurons; says
  nothing about *learnability*). The follow-up "2–3 layers train better than one wide layer" partly covers this.
- **(Note, not error) MLP-on-flattened-pixels is the very approach L08 improves on.** Slide 12 feeds 900 raw
  pixels to an MLP; NB 01 shows this explodes to ~38M params at 224² and ignores spatial structure. That's the
  intended L07→L08 setup — fine, as long as it's framed as the *baseline*, which TinyCNN (rung 3) beats.

---

## 2. Are they "the same thing"? — No: one ladder, three rungs

| Dimension | **Slide 11** perceptron | **Slide 12** MLP | **PNG** TinyCNN (CNN) |
|---|---|---|---|
| **Level** | the atom (1 neuron) | layers of neurons | conv body + MLP head |
| **Params** | 4 | ~235K (FlatMLP) | **105,866** (4,800 conv + 101,066 MLP head) |
| **Input** | 3 tabular features | 900 flattened pixels | 28×28 image (kept 2-D until flatten) |
| **Key op** | `z = Σwx + b` | full matrix `Wx + b` per layer | **convolution** (shared local weights) + dense |
| **Weights** | independent | independent (fully connected) | **shared** in conv; independent in head |
| **Activation** | **step** (untrainable) | **ReLU + softmax** | **ReLU + softmax** |
| **Spatial prior** | none | **none** (flattening discards it) | **yes** (locality + translation invariance) |
| **Trained by** | perceptron rule (historical) | gradient descent | gradient descent |

**Precise relationships**
- Every node in slide 12's hidden/output layers **is** a slide-11 perceptron — same `z = Σwx + b` — but with
  **ReLU/softmax** instead of a step (so it can be trained).
- The PNG's **head** (`Flatten → Dense64 → Dense10`) **is** a slide-12 MLP, operating on conv features rather
  than raw pixels.
- The PNG's **conv body** is the one genuinely new thing: a perceptron whose weights are a small kernel
  **shared** across all spatial positions — that weight-sharing is L08's whole contribution over L07's MLP.

So the three artifacts are **not** alternatives; they are **successive zoom-outs** of the same machinery.

---

## 3. Consolidation — "From one neuron to a CNN" (single 3-panel figure)

```
┌──────────────────  NEURON → MLP → CNN  (L07 → L08 in one picture)  ───────────────────┐
│                                                                                        │
│  (a) PERCEPTRON  [s11]        (b) MLP  [s12] = stack (a)        (c) CNN [PNG] = conv + (b)│
│  ┌────────────────────┐       ┌──────────────────┐             Coat 28×28×1             │
│  │ x1─w1┐             │       │  ●●●  hidden(ReLU)│  ── used    → [Conv p1 +ReLU+Pool]×2 │
│  │ x2─w2┼─Σ(+b)─►f─►y │ ×N →  │  ●●●              │     as ──►  →  Flatten 1568          │
│  │ x3─w3┘  z=Σwx+b    │       │  softmax → classes│   the head  →  Dense64 +ReLU         │
│  └────────────────────┘       └──────────────────┘             →  Dense10 → 10 logits   │
│   f: step → ReLU/softmax       independent weights              conv = SHARED local      │
│   bias = threshold (at 0)      on FLATTENED pixels              weights (locality)       │
│                                                                                          │
│   the MLP (b) is literally the CNN's head (c); the conv layers are the new L08 part      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Reconciliations the merged figure must make
1. **Activation:** replace slide 11's **step** with **ReLU/softmax**, stating *why* (step has no gradient).
2. **Bias/threshold:** adopt slide 12's correct framing (**bias = threshold, step at 0**) and fix slide 11 to match.
3. **Weight sharing:** show FC neurons have independent weights (rungs 1–2), but conv filters **reuse one small
   kernel everywhere** (rung 3) — the single new idea.
4. **Spatial prior:** slide 12 flattens pixels (discards 2-D structure); the CNN keeps it until the flatten,
   which is exactly why TinyCNN beats FlatMLP on images.
5. **Scale:** 4 params → ~235K (MLP) → 105,866 (CNN: 4,800 conv + 101,066 MLP head).

### Caption
> *"A neuron computes `z = Σwᵢxᵢ + b` then an activation (a). Stack many into layers and you get an MLP (b),
> which classifies flattened pixels. Put convolution layers — weight-shared local neurons — in front, and the
> same MLP now classifies learned spatial features: that is TinyCNN (c). 105,866 weighted sums, trained end-to-end."*

**Feasibility:** high — one three-panel zoom-out. It is the cleanest possible L07→L08 bridge: it answers
"where did the perceptron and the MLP go?" by pointing at the CNN's dense head.

---

## 4. Consolidated recommendations

| # | Artifact | Severity | Issue | Fix |
|---|---|---|---|---|
| 1 | Slide 11 | **Medium** | Bias + separate 3.0 threshold (redundant, non-standard) | Step at 0; bias = threshold |
| 2 | Slide 12 | **Medium** | "Threshold functions like ReLU" / "x>0 → non-linearity" mislabels ReLU | Call ReLU a rectifier; non-linearity is the kink at 0 |
| 3 | Slides 11↔12 | **Medium** | Contradict each other on bias-vs-threshold | Use the slide-12 (bias=threshold) convention in both |
| 4 | PNG | **Medium** | Digit "7"/"0–9" vs. Fashion-MNIST clothing | Clothing sample + labels |
| 5 | PNG | Low–Med | `padding=1` omitted from conv annotation | Annotate `padding 1 (same)` |
| 6 | Slide 12 | Low | UAT stated without "bounded domain / existence-not-learnability" | Add the caveat |
| 7 | PNG | Low | Stray "0-6" label; duplicate Block-2 caption | Clean labels |
| 8 | all three | **Medium (value)** | Shown apart, the abstraction ladder is invisible | Merge as the "neuron → MLP → CNN" figure (§3) |

**Priority:** #1–#4 are the real correctness/clarity items; **#8 is the high-value teaching win** — it ties
L07 (neuron, MLP) directly to L08 (CNN) and explains *why* the CNN exists.

---

## Appendix A — verified arithmetic & the parameter ladder
```
Perceptron : z = 320(0.01)+8(0.5)+75(-0.02)+(-0.5) = 5.2 ; 5.2≥3.0 → 1     ✓   (4 params)
MLP/FlatMLP: 784→256→128→10                                                = 235,146 params
TinyCNN    : conv body 160+4,640 = 4,800 ;  MLP head 100,416+650 = 101,066 ;  total = 105,866
```

## Appendix B — the one equation at all three rungs
```
neuron :  y    = f( Σ_i w_i x_i + b )                       (slide 11; f = step)
MLP    :  y    = f( W x + b )           per layer            (slide 12; f = ReLU then softmax)
conv   :  y[c,i,j] = f( Σ K_c · patch(i,j) + b_c )          (PNG body; K_c SHARED across i,j)
head   :  y    = softmax( W2 · ReLU(W1 · flatten + b1) + b2 ) (PNG head = an MLP)
```

## Appendix C — sources cross-checked
- `docs/Lesson 3.7 - Neural Networks & Deep Learning.pdf` slides 11 (perceptron), 12 (MLP), 13 (NN-in-a-nutshell).
- `notebooks/03_first_cnn.ipynb` (TinyCNN), `notebooks/01_monday_morning.ipynb` (FlatMLP = the MLP rung).
- `lesson.md`, `reference.md`; companions `cnn-architecture-convolution-study.md`, `lesson-3.8-slide-review.html`.
