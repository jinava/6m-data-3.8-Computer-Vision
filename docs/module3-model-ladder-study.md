# Module 3 (DSAI) — the model ladder: LINEAR → LOGISTIC → NEURON → MLP → CNN → EMBEDDINGS → ATTENTION

> **Author:** [Opus 4.8] · **Date:** 2026-06-29
> **Scope:** A study of the *whole* SCTP DSAI Module 3 (lessons **3.1–3.10**) that extends the
> "NEURON → MLP → CNN (L07 → L08)" picture **downward** (the simpler models that come first) and
> **upward** (embeddings, attention, transformers), and places the non-neural lessons as **side-rails**.
> Grounded in each lesson's `lesson.md`; complements `perceptron-mlp-cnn-study.md`,
> `cnn-architecture-convolution-study.md`, and `lesson-3.8-slide-review.html`.

---

## 0. The one picture — the extended ladder

```
              ┌──────────────────────  THE NorthStar STORY (Sarah Chen)  ──────────────────────┐
 FOUNDATIONS  │  L01 (3.1) What is ML? framing + 7-step workflow                                │
 (underpin    │  L02 (3.2) Probability & Statistics — distributions, CI, p-value, effect size  │
  everything) │            → the EVALUATION lens every rung is read through                     │
              └────────────────────────────────────────────────────────────────────────────────┘

 THE SUPERVISED / REPRESENTATION-LEARNING SPINE  (each rung = a richer use of  z = Σ wᵢxᵢ + b )

   rung 0          rung 1            rung 2         rung 3        rung 4         rung 5        rung 6
 ┌─────────┐    ┌──────────┐     ┌──────────┐   ┌────────┐   ┌─────────┐    ┌──────────┐   ┌──────────────┐
 │ LINEAR  │ →  │ LOGISTIC │  →  │ PERCEPTRON│→  │  MLP   │ → │   CNN   │ →  │ EMBEDDINGS│→ │ ATTENTION /  │
 │ REGRESS.│    │ REGRESS. │     │ = NEURON  │   │ (deep) │   │ (images)│    │ (meaning) │   │ TRANSFORMER  │
 │  L03    │    │   L03    │     │   L07     │   │  L07   │   │   L08   │    │   L09     │   │  L10 → LLM/RAG│
 └─────────┘    └──────────┘     └──────────┘   └────────┘   └─────────┘    └──────────┘   └──────────────┘
  w·x+b         σ(w·x+b)         f(w·x+b)        f(Wx+b)      f(K∗x+b)       Encoder(text)   softmax(QKᵀ/√d)V
  identity      sigmoid+         the named       stack +      weight-        dense vec,      DATA-DEPENDENT
  (no actv.)    cross-entropy    unit; trained   non-linear   SHARED, LOCAL  geometry =      weights; every
                trainable        by grad-desc    layers       (spatial bias) meaning         token attends all

   SIDE-RAILS (different paradigms the module also teaches — NOT weighted-sum stacks):
   • L04 (3.4) TREES & ENSEMBLES — axis-aligned splits; random forest / gradient boosting. *Tabular champion.*
   • L05 (3.5) UNSUPERVISED — PCA (variance axes), K-Means (distance clusters), Isolation Forest (anomalies). *No labels.*
   • L06 (3.6) TIME SERIES — STL, ETS/Holt-Winters, lag-feature GB. *Breaks the i.i.d. assumption.*
```

> **Read the arrows as "adds a capability", not "is better than".** A later rung is **not** automatically
> superior — trees still beat deep nets on tabular data (L04), classical ETS beats fancy models on clean
> series (L06), and logistic regression is the right call when you need interpretable coefficients (L03).
> The module's meta-lesson is **match the tool to the data**, not "always climb."

---

## 1. The unifying thread — one equation, evolving

Almost the entire spine is the **weighted sum `z = Σ wᵢxᵢ + b`** wrapped in progressively richer ways. What
changes from rung to rung is **(a) the activation**, **(b) where the weights live**, and **(c) the inductive bias**.

| Rung | Equation | Activation | Where the weights are |
|---|---|---|---|
| Linear reg (L03) | `ŷ = w·x + b` | identity | fixed, global |
| Logistic reg (L03) | `ŷ = σ(w·x + b)` | sigmoid (+ cross-entropy) | fixed, global |
| Perceptron (L07) | `ŷ = f(w·x + b)` | step (historical) | fixed, global |
| Neuron / dense (L07) | `a = f(Wx + b)` | ReLU / softmax | fixed, global |
| MLP (L07) | `y = softmax(W₂·ReLU(W₁x + b₁) + b₂)` | ReLU + softmax | fixed, global |
| Conv (L08) | `a[c,i,j] = f(Σ Kc · patch(i,j) + bc)` | ReLU | fixed, **shared & local** |
| Embedding (L09) | `e = Encoder(tokens) ∈ ℝ³⁸⁴` | (transformer inside) | pretrained |
| Attention (L10) | `Attn(Q,K,V) = softmax(QKᵀ/√dk)·V` | softmax | **dynamic — computed from the data** |

**The single most important progression** is column 3 — *where the weights come from*:

```
fixed & GLOBAL  (linear → logistic → MLP)
   └─► fixed & SHARED+LOCAL  (CNN: one small kernel reused everywhere = locality + translation invariance)
          └─► DYNAMIC, content-dependent  (attention: connection strengths are softmax(QKᵀ), recomputed per input)
```

That is the real story of the module's deep-learning half: weights go from *one fixed value per
input-pair*, to *one fixed kernel shared across space*, to *weights that are themselves a function of the input.*

---

## 2. The rungs in build-order (detail)

### Rung 0 — Linear regression *(L03, conceptual base)*
`ŷ = w·x + b`. A neuron with **no activation**. The purest weighted sum; the regression sibling of the
classifier below. (L03's concrete task is classification, so logistic is the built model; linear regression
is the identity-activation special case that anchors the whole ladder.)

### Rung 1 — Logistic regression *(L03 — the first model Sarah owns)*
`ŷ = σ(w·x + b)`, sigmoid output, **cross-entropy loss**, trained by **gradient descent**. This **is a single
trainable neuron** — the course builds the neuron's *mathematics* here (L03), four lessons before it *names*
it a neuron (L07). Key L03 ideas that persist up the whole ladder: preprocessing pipelines (no leakage),
train/validate discipline, metrics over accuracy (precision/recall/F1), and **threshold as a business decision**.

### Rung 2 — Perceptron = the neuron *(L07, slide 11)*
`f(w·x + b)` named as the atomic unit. Historically a **step** activation (non-differentiable → not trainable
by gradient descent); modern neurons use ReLU/sigmoid. *(See `perceptron-mlp-cnn-study.md` for the slide-11
bias/threshold concern.)*

### Rung 3 — MLP *(L07, slide 12)*
Stack neurons into layers with **non-linear activations**; **softmax** output. Universal approximator. This is
`notebooks/01_monday_morning.ipynb`'s `FlatMLP` (784→256→128→10, 235K params). Its weakness on images
(parameter blow-up, no spatial prior) motivates the next rung.

### Rung 4 — CNN *(L08)*
Replace the MLP's first layers with **convolutions** — a perceptron whose weights are a small kernel **shared
across every spatial location** (locality + translation invariance, far fewer parameters). TinyCNN =
conv body (`4,800` params) **+ an MLP head** (`Flatten 1568 → Dense64 → Dense10`, `101,066` params). *The MLP
of rung 3 literally reappears as the CNN's classifier head.* Transfer learning: reuse a pretrained backbone.

### Rung 5 — Embeddings *(L09)*
Map tokens/sentences → **dense vectors where geometric distance encodes meaning** (`all-MiniLM-L6-v2`, 384-dim).
The representation idea generalises from pixels to words: instead of one-hot tokens (equidistant, meaningless),
learn vectors where *frock ≈ dress*. Semantic search = embed corpus once + cosine-rank against the query.
**This is the "R" (retrieval) in RAG.**

### Rung 6 — Attention / Transformer → LLM / RAG *(L10)*
`Attn(Q,K,V) = softmax(QKᵀ/√dk)·V`. Each token emits a **query / key / value**; weights are the softmax of
query·key — i.e. **the network computes its own connection strengths from the data**, so every token can attend
to every other (no fixed locality like CNNs, no forgetting like RNNs). Stack attention + MLP blocks (residuals)
→ a transformer; pretrain as a next-token predictor → an **LLM**; ground it on your data with **RAG** =
embeddings-retrieval (L09) + LLM generation (L10). Scale changes capability, not the architecture.

---

## 3. The side-rails — paradigms off the weighted-sum spine

These are core to the module but are **not** stacks of weighted sums, so they sit beside the ladder, not on it.

### L04 (3.4) — Trees & ensembles *(the tabular champion)*
Decision trees split on **thresholds**, not weighted sums (so no scaling needed); they capture non-linear
interactions for free. **Random forest** = bagging (variance↓); **gradient boosting** = sequential residual
correction (accuracy ceiling↑, more tuning). On real tabular data these are **often the best model, full stop**
— a deliberate counterweight to "deep learning always wins."

### L05 (3.5) — Unsupervised *(no labels)*
Three jobs: **PCA** (linear projection onto variance-maximising axes — a cousin of the weighted sum, but
unsupervised), **K-Means** (distance-based clustering; assumes spherical clusters), **Isolation Forest**
(anomaly scores). Output needs human profiling to become named segments.

### L06 (3.6) — Time series *(breaks i.i.d.)*
Order matters: no shuffling, no k-fold. **STL** decomposes trend/seasonality/residual; **ETS/Holt-Winters** is
the pragmatic classical default; **lag features + gradient boosting** is the ML route (reuses the L04 toolkit).
Always beat a **naive/seasonal-naive baseline** first.

---

## 4. The foundation bar — under every rung

- **L01 (3.1) — What is ML?** Framing, the three categories, the 7-step workflow (60–70% is data work),
  evaluate on unseen data, models decay. *Knowing when **not** to use ML is half the skill.*
- **L02 (3.2) — Probability & Statistics.** Distributions, the CLT, **confidence intervals**, p-values,
  **effect size**. This is the **evaluation lens**: every accuracy number on every rung above should be
  reported as `84% (95% CI: 78–89%)`, not a bare point estimate. Statistical ≠ practical significance.

---

## 5. Per-lesson review notes (brief)

| Lesson | Strength | Watch-out / note |
|---|---|---|
| 3.1 Intro ML | Excellent "ML-fit" checklist; honest 60–70%-is-data framing | — |
| 3.2 Prob/Stats | The CI reporting habit is the best single discipline in the course | — |
| 3.3 Supervised | Pipeline/leakage discipline; threshold-as-business-decision | Logistic regression is *literally* the trainable neuron of L07 — worth making that link explicit |
| 3.4 Trees | Honest "highest F1 ≠ right ship"; tabular realism | — |
| 3.5 Unsupervised | "Pick the job before the algorithm"; no-label evaluation honesty | K-Means spherical-cluster assumption flagged well |
| 3.6 Time series | i.i.d.-breaks framing; baseline-first discipline | — |
| 3.7 NN | Clean neuron→MLP build | **slide 11 bias/threshold conflated; slide 12 mislabels ReLU as a "threshold function" and contradicts slide 11** (see `perceptron-mlp-cnn-study.md`) |
| 3.8 CV | Honest "transfer can lose" result; tight notebook alignment | **slide 5 Q3 over-gates transfer learning; Sobel Gx/Gy labels conflict with NB02; PNG digit-vs-clothing** (see `lesson-3.8-slide-review.html`) |
| 3.9 NLP | "Cosine is a ranking not a %"; hybrid-search realism | — |
| 3.10 GenAI | Q/K/V explained as one matmul+softmax; RAG grounding + eval-set discipline | "2–3 layers = deep" is loose; UAT-style claims could note caveats |

> **Cross-lesson consistency win:** the **bias = threshold** framing in L07 slide 12 is the *correct* one; the
> separate-threshold framing in L07 slide 11 is the outlier. Aligning slide 11 to it would also make the whole
> ladder's "activation applied at 0" story consistent from logistic regression (rung 1) upward.

---

## 6. How this extends the original "NEURON → MLP → CNN" figure

The L07→L08 picture is the **middle three rungs** of a seven-rung spine. Extending it:

- **Downward (where the neuron came from):** the neuron isn't new in L07 — Sarah *already trained one* in L03 as
  **logistic regression**. Adding rungs 0–1 turns "here is a neuron" into "you've used this since week 3."
- **Upward (where it goes):** after CNN, the representation idea moves from pixels to **meaning** (embeddings,
  L09), and the *fixed* weights become **dynamic** (attention, L10). RAG ties L09 + L10 together.
- **Sideways (honesty):** the tree/unsupervised/time-series rails stop the figure from implying a single
  hierarchy of superiority.

**Suggested master caption:**
> *"One idea — the weighted sum `z = Σwx + b` — runs the length of Module 3. Make it trainable (logistic
> regression), name it (neuron), stack it (MLP), share it across space (CNN), use it to encode meaning
> (embeddings), and finally let the data compute its own weights (attention). Everything else — trees,
> clustering, forecasting — is a different, equally valid tool for a different shape of data."*

---

## Appendix A — the equation at every rung
```
linear reg   :  ŷ = w·x + b
logistic reg :  ŷ = σ(w·x + b)                         loss = cross-entropy ; trained by gradient descent
perceptron   :  ŷ = step(w·x + b)                      (historical; not GD-trainable)
neuron/dense :  a = f(Wx + b)                          f = ReLU/softmax
MLP          :  y = softmax(W₂·ReLU(W₁x + b₁) + b₂)
conv         :  a[c,i,j] = f(Σ Kc·patch(i,j) + bc)     Kc SHARED across all i,j
embedding    :  e = Encoder(tokens) ∈ ℝ^384            geometric distance ≈ semantic distance
attention    :  Attn(Q,K,V) = softmax(QKᵀ/√dk)·V       weights are a FUNCTION of the input
transformer  :  x ← x + Attn(x);  x ← x + MLP(x)       (residual blocks, stacked)
LLM          :  p(next token) = softmax(transformer(tokens))
RAG          :  answer = LLM( prompt ⊕ topK(embed(query)) )
```

## Appendix B — the NorthStar arc (one storyline, ten lessons)
sentiment of 10k reviews (L01) → how sure are we? (L02) → churn classifier (L03, logistic) →
make it better (L04, boosting) → segment customers, no labels (L05) → forecast revenue (L06) →
neural nets (L07) → auto-tag product photos (L08, CNN) → semantic catalogue search (L09, embeddings) →
natural-language shopping assistant (L10, RAG). One analyst, one company, ten tools.

## Appendix C — sources
`lesson.md` of all ten repos `6m-data-3.1 … 3.10`; the L07/L08 slide decks; the TinyCNN notebook
(`03_first_cnn.ipynb`) and FlatMLP notebook (`01_monday_morning.ipynb`); companion docs
`perceptron-mlp-cnn-study.md`, `cnn-architecture-convolution-study.md`, `lesson-3.8-slide-review.html`.
