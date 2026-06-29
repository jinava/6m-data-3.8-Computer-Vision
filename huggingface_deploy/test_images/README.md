# Test Images

Drop these into your deployed app to see where the model works and where it breaks.

## `success_*` — should be recognised correctly
These are 10 real Fashion-MNIST images (one per category), shown as a normal-looking
dark item on a white background. The app flips light/dark for you, so the model sees
them exactly like its training data. Use these to confirm your app works — the
filename tells you the correct answer.

## `fail_*` — should fail (on purpose!)
- `fail_random_noise.png` — random colour static. Not clothing at all.
- `fail_red_circle.png` — a plain shape, like a simple real-world photo.
- `fail_face.png` — a cartoon face, standing in for "a photo of a person".

Watch what category the model forces these into, and how **confident** it is even
when it's completely wrong. That's the lesson — see `3_EXPERIMENT_WORKSHEET.md`.

> Best of all: upload your own phone photos too and compare!
