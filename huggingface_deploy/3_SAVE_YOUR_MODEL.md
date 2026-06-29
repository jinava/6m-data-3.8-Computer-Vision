# Make a New Deployable `.pt` File

So you trained a new or improved model and want to deploy *that one*. This shows
how to save it correctly and plug it into your Space.

The golden rule of deployment:

> **The model file and the `app.py` code must describe the *same* network.**
> A `.pt` file is just a bag of numbers (the weights). It has no idea what shape
> it's supposed to be — that lives in your `app.py`. If the two disagree, loading
> fails.

---

## Step 1 — Save the weights (not the whole model)

At the end of your training notebook, after the model is trained, add **one line**:

```python
torch.save(model.state_dict(), "tiny_cnn.pt")
print("Saved: tiny_cnn.pt")
```

- `model` is your trained network variable (in notebook 03 it's called `cnn`).
- `state_dict()` saves **only the learned numbers** — small, portable, the standard
  way to deploy.
- ⚠️ Do **not** use `torch.save(model, ...)` (the whole object). It looks easier but
  breaks easily across machines and library versions. Always save the `state_dict`.

> If your model is on a GPU, that's fine — `state_dict()` saves CPU-friendly numbers.
> When you reload in `app.py` we already use `map_location="cpu"`.

---

## Step 2 — Check what the saved file expects

The 30-second sanity check that saves hours of confusion. In a fresh cell:

```python
import torch
sd = torch.load("tiny_cnn.pt", map_location="cpu")
for name, tensor in sd.items():
    print(name, tuple(tensor.shape))
```

This prints every layer's name and size, e.g.:

```
features.0.weight (16, 1, 3, 3)
features.0.bias   (16,)
...
classifier.3.weight (10, 64)   <- last number = number of classes
```

Whatever this prints, your `app.py` `TinyCNN` class **must** produce the exact same
names and shapes. (For the original course model, it already does.)

---

## Step 3 — Update `app.py` only if you changed the architecture

You change `app.py` **only** if your new model is structurally different. Three
cases:

**A) Same architecture, just retrained / more epochs**
→ Change nothing. Just upload the new `tiny_cnn.pt` over the old one. Done.

**B) You changed layers** (added a conv block, changed sizes, e.g. the `DeeperCNN`
from the optional notebook)
→ Copy your **new** model class into `app.py`, replacing the old `TinyCNN`
definition. The class in `app.py` must be character-for-character the one you
trained with.

**C) You changed the number or meaning of classes**
→ Update the `CLASS_NAMES` list in `app.py` so it has exactly the right labels **in
the same order (0, 1, 2, …)** your training data used. The list length must equal the
last number from Step 2.

> Tip: if you used a different dataset, also fix the preprocessing in `app.py`
> (`Grayscale`, `Resize((28,28))`, `invert`) to match how you fed images during
> training. Whatever transforms you trained with, the app must do the same.

---

## Step 4 — Prove it reloads before you deploy

Always confirm the file loads into your `app.py` class **on a clean run** — this
catches mismatches on your laptop instead of in a failed build:

```python
# paste the SAME class you put in app.py, then:
m = TinyCNN()                 # or your new class name
m.load_state_dict(torch.load("tiny_cnn.pt", map_location="cpu"))
m.eval()
print("Loaded cleanly ✅")
```

- **No error** → you're safe to upload.
- `Error(s) in loading state_dict: Missing/Unexpected key(s)` → the class and the
  file disagree. Re-check Step 3. The error text names exactly which layers mismatch.

---

## Step 5 — Deploy the new file

Go to your Space → **Files** → upload the new `tiny_cnn.pt` (and the edited `app.py`
if you changed it) → **Commit**. The Space rebuilds automatically in a minute or two.
Refresh the **App** tab and test.

---

### Quick reference

| You changed… | Update `tiny_cnn.pt`? | Update `app.py`? |
|--------------|:---:|:---:|
| Just retrained / more epochs | ✅ | — |
| Layers / architecture | ✅ | ✅ (model class) |
| Number or names of classes | ✅ | ✅ (`CLASS_NAMES`) |
| Input type (size, colour, dataset) | ✅ | ✅ (preprocessing) |
