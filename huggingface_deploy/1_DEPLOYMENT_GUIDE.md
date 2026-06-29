# Deploy Your CNN to Hugging Face — Step by Step

In this project you'll put **your own trained model** online as a real web app.
Anyone with the link can upload an image and watch your CNN guess what it is.

You'll use **Hugging Face Spaces** — a free service that turns a few files into a
live website. No servers, no payment, no command line required.

**Time needed:** about 20–30 minutes.

---

## What's in this folder

You need to upload three files. They're already prepared for you here:

| File | What it is |
|------|------------|
| `app.py` | The web app. Loads your model and draws the "upload → predict" page. |
| `requirements.txt` | The list of libraries Hugging Face must install for you. |
| `tiny_cnn.pt` | **Your trained model's weights** — the thing you made in notebook 03. |

> 💡 If you retrained the model yourself, replace `tiny_cnn.pt` with your own
> saved file (keep the same filename, or update the filename inside `app.py`).

---

## Part 1 — Create a Hugging Face account

1. Go to **https://huggingface.co** and click **Sign Up**.
2. Register with your email and verify it (check your inbox).
3. Pick a username — this becomes part of your app's web address.

That's it. The free account is all you need.

---

## Part 2 — Create a Space

A "Space" is Hugging Face's word for a hosted web app.

1. Click your profile picture (top-right) → **New Space**.
   (Or go straight to **https://huggingface.co/new-space**.)
2. Fill in the form:
   - **Owner:** your username.
   - **Space name:** e.g. `my-clothing-recogniser`.
   - **License:** `mit` is fine.
   - **Select the Space SDK:** choose **Gradio**. *(This is important — it tells
     Hugging Face to run `app.py` as a Gradio web app.)*
   - **Hardware:** leave it on the free **CPU basic**. Our model is tiny.
   - **Visibility:** **Public** (so you can share the link).
3. Click **Create Space**.

You now have an empty Space with its own web page.

---

## Part 3 — Upload your three files

1. On your new Space's page, click the **Files** tab.
2. Click **+ Add file** → **Upload files**.
3. Drag in all three files from this folder:
   `app.py`, `requirements.txt`, and `tiny_cnn.pt`.
4. Scroll down and click **Commit changes to main**.

---

## Part 4 — Wait for it to build, then test

1. Click the **App** tab. You'll see a **"Building"** status.
   Hugging Face is installing PyTorch and friends — this takes **2–5 minutes**
   the first time. (You can watch progress under the **Logs** tab.)
2. When it says **Running**, your app appears: an upload box and a results box.
3. Upload an image and watch the top 3 guesses appear.

🎉 **You've deployed your model.** Your link looks like:
`https://huggingface.co/spaces/YOUR-USERNAME/my-clothing-recogniser`
Share it with anyone.

---

## Part 5 — Experiment (this is the real point!)

Your model is **not magic**. It only ever saw 28×28 grayscale clothing images
during training. Now go find out where it works and where it breaks.

Open `3_EXPERIMENT_WORKSHEET.md` and run the experiments there. Try clothing photos,
non-clothing photos, busy backgrounds, colourful images — and record what happens.
The goal is to *understand why* it succeeds or fails, not just to make it work.

---

## If something goes wrong

| Symptom | Likely cause & fix |
|---------|--------------------|
| Status stuck on **Building** for >10 min | Open the **Logs** tab to read the error. Usually a typo in `requirements.txt`. |
| **"Error"** / red screen | Check the filename: `app.py` must be named exactly that, and `tiny_cnn.pt` must be uploaded. |
| `Error(s) in loading state_dict` | The `TinyCNN` code in `app.py` doesn't match the model you saved. They must be identical. |
| Predictions look random | That's often **correct behaviour** — see the worksheet. The model guesses on anything. |

> Tip: after editing any file, the Space rebuilds automatically. You don't need to
> recreate it.
