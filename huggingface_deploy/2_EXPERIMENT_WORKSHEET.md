# Experiment Worksheet — When Does Your Model Work?

Your model was trained on **Fashion-MNIST**: tiny 28×28 **grayscale** photos of
clothing, on a plain background, with the item centred and filling the frame.
It knows exactly **10 categories**:

> T-shirt/top · Trouser · Pullover · Dress · Coat · Sandal · Shirt · Sneaker · Bag · Ankle boot

A model can only be good at what it has seen. These experiments help you *feel*
that truth instead of just being told it.

For each experiment: make a prediction **first**, then upload, then write down what
actually happened and why you think so.

---

## Experiment 1 — The "home turf" test (should succeed)

Upload an image close to what the model trained on: a single clothing item,
centred, plain background. (You can download a real Fashion-MNIST sample, or take a
photo of one shoe on a white table.)

- My guess before uploading: ____________________
- Top prediction & confidence: ____________________
- Correct? ☐ Yes ☐ No
- **Why do you think it did well/badly?** ____________________

---

## Experiment 2 — A category it doesn't know (should fail interestingly)

Upload a clothing item that is **not** one of the 10 categories — e.g. a hat,
gloves, a scarf, socks, sunglasses.

- Top prediction: ____________________
- What did it *force* the image into? ____________________
- **Key idea:** the model has no "I don't know" option. It must pick one of 10.
  Write one sentence on what that means for trusting a model: ____________________

---

## Experiment 3 — Something completely different (should fail)

Upload a non-clothing photo: a face, a pet, food, a building, a landscape.

- Top prediction & confidence: ____________________
- Was the model still **confident** even though it was wrong? ☐ Yes ☐ No
- **Discuss:** why is a *confidently wrong* model dangerous? ____________________

---

## Experiment 4 — Same item, harder conditions (find the breaking point)

Take the SAME clothing item and photograph it several ways. Note where accuracy
drops off:

| Condition | Prediction | Still correct? |
|-----------|------------|----------------|
| Plain background, centred | | |
| Busy / cluttered background | | |
| Very colourful / patterned | | |
| Item small in a big frame | | |
| Rotated / sideways | | |
| Dark photo / bad lighting | | |

- Which condition broke it first? ____________________
- **Why?** Connect it back to what the training images looked like: ____________________

---

## Wrap-up — What did you learn?

1. In one sentence, when does your model work well?
   ____________________
2. In one sentence, when does it fail?
   ____________________
3. The model is only as good as its ____________________ (fill the blank).
4. If you wanted it to recognise *your* phone photos reliably, what would you need
   to change about the training? ____________________

---

### For the instructor — talking points
- **Generalisation gap:** great on test set ≠ good on real-world photos. The
  distribution shift (colour, resolution, background, framing) is the whole lesson.
- **No abstain option:** softmax always sums to 1, so the model always "decides".
  High confidence is not the same as being right.
- **Data is destiny:** every failure traces back to "the model never saw that".
  This motivates data collection, augmentation, and transfer learning (notebook 04).
