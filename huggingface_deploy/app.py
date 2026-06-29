# app.py — Fashion-MNIST clothing recogniser (Hugging Face Space)
# ----------------------------------------------------------------
# This is the file Hugging Face runs to build your web app.
# It loads the model YOU trained (tiny_cnn.pt) and puts an
# "upload a photo -> get a prediction" page in front of it.

import torch
import torch.nn as nn
import torchvision.transforms as T
import gradio as gr

# ----------------------------------------------------------------
# 1) The model definition.
#    This MUST be the exact same TinyCNN from your training notebook,
#    otherwise the saved weights won't fit. (Copied from 03_first_cnn.)
# ----------------------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# The 10 classes Fashion-MNIST knows about — IN THIS ORDER (0..9).
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ----------------------------------------------------------------
# 2) Load the trained weights once, when the app starts up.
# ----------------------------------------------------------------
model = TinyCNN()
model.load_state_dict(torch.load("tiny_cnn.pt", map_location="cpu"))
model.eval()  # eval mode = "we're predicting, not training"

# ----------------------------------------------------------------
# 3) Turn ANY uploaded photo into what the model expects:
#    a 28x28 GRAYSCALE image.
#    Fashion-MNIST images are light clothing on a BLACK background,
#    so we invert (most phone photos are dark item on light background).
# ----------------------------------------------------------------
preprocess = T.Compose([
    T.Grayscale(num_output_channels=1),  # colour -> grey
    T.Resize((28, 28)),                  # shrink to 28x28
    T.functional.invert,                 # flip light/dark to match training data
    T.ToTensor(),                        # -> tensor with values 0..1
])


def predict(image):
    if image is None:
        return {}
    x = preprocess(image).unsqueeze(0)   # add a batch dimension: (1, 1, 28, 28)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]  # turn scores into probabilities
    # Gradio's Label widget wants {class_name: probability}
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


# ----------------------------------------------------------------
# 4) Build the web page.
# ----------------------------------------------------------------
description = (
    "Upload an image and my CNN will guess which of 10 clothing categories it is.\n\n"
    "⚠️ This model only ever saw tiny 28×28 grayscale clothing images during training "
    "(Fashion-MNIST). It will confidently guess on ANY image — even ones it should not "
    "recognise. Try different photos and notice when it succeeds and when it fails!"
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=3, label="Top guesses"),
    title="My Fashion-MNIST Clothing Recogniser",
    description=description,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
