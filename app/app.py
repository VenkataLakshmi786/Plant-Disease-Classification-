from pathlib import Path
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = BASE_DIR / "best_resnet_se_model.pth"

CLASS_NAMES = ["bacteria", "fungus", "healthy", "pests", "virus"]
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        weights = self.pool(x).view(batch_size, channels)
        weights = self.fc(weights).view(batch_size, channels, 1, 1)
        return x * weights


class ResNetSE(nn.Module):
    """ResNet-50 + SE attention model"""
    def __init__(self, num_classes=5):
        super().__init__()
        # Do not download pretrained weights: the saved checkpoint supplies the weights.
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.se1 = SEBlock(256)
        self.se2 = SEBlock(512)
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, num_classes))

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.se1(self.backbone.layer1(x))
        x = self.se2(self.backbone.layer2(x))
        x = self.se3(self.backbone.layer3(x))
        x = self.se4(self.backbone.layer4(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}. Copy best_resnet_se_model.pth next to app.py."
        )
    model = ResNetSE(num_classes=len(CLASS_NAMES))
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


MODEL = None
MODEL_ERROR = None
try:
    MODEL = load_model()
except Exception as exc:
    MODEL_ERROR = str(exc)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None
    error = MODEL_ERROR

    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            error = "Choose an image file first."
        elif not allowed_file(file.filename):
            error = "Unsupported file type. Upload PNG, JPG, JPEG, BMP, or WEBP."
        elif MODEL is None:
            error = f"The model could not be loaded: {MODEL_ERROR}"
        else:
            try:
                image = ImageOps.exif_transpose(Image.open(file.stream)).convert("RGB")
                filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
                save_path = UPLOAD_DIR / filename
                image.save(save_path, format="JPEG" if save_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else None)
                tensor = preprocess(image).unsqueeze(0).to(DEVICE)
                with torch.inference_mode():
                    logits = MODEL(tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]
                predicted_index = int(torch.argmax(probabilities).item())
                prediction = {
                    "label": CLASS_NAMES[predicted_index],
                    "confidence": float(probabilities[predicted_index].item()) * 100,
                    "scores": sorted(
                        [{"label": CLASS_NAMES[i], "score": float(probabilities[i].item()) * 100}
                         for i in range(len(CLASS_NAMES))],
                        key=lambda item: item["score"], reverse=True,
                    ),
                }
                image_url = f"uploads/{filename}"
                error = None
            except (UnidentifiedImageError, OSError):
                error = "The uploaded file is not a readable image."

    return render_template("index.html", prediction=prediction, image_url=image_url, error=error, device=str(DEVICE))


@app.errorhandler(413)
def file_too_large(_error):
    return render_template("index.html", prediction=None, image_url=None,
                           error="File is too large. Maximum upload size is 10 MB.", device=str(DEVICE)), 413


if __name__ == "__main__":
    # Local development only. For deployment, use a production WSGI server.
    app.run(host="127.0.0.1", port=5000, debug=False)
