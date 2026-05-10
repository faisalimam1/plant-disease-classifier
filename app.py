import gradio as gr
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import json
import numpy as np

# ── Load class names ──────────────────────────────────
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# ── Load model ─────────────────────────────────────────
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 38)
model.load_state_dict(torch.load('plant_disease_model.pth',
                                  map_location='cpu'))
model.eval()
print("Model loaded successfully ✅")

# ── Transform ──────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Treatment recommendations ──────────────────────────
treatments = {
    "scab":           "Apply fungicide (Captan/Mancozeb). Remove infected leaves. Improve air circulation.",
    "black_rot":      "Prune infected branches. Apply copper-based fungicide. Remove mummified fruits.",
    "rust":           "Apply sulfur fungicide in spring. Remove nearby juniper trees if possible.",
    "powdery_mildew": "Apply sulfur fungicide. Improve air circulation. Avoid overhead watering.",
    "blight":         "Apply fungicide immediately. Remove infected plants. Avoid overhead watering.",
    "spot":           "Remove infected leaves. Apply fungicide. Avoid wetting foliage.",
    "virus":          "No cure. Remove infected plants immediately. Control insect vectors.",
    "mold":           "Improve ventilation. Reduce humidity. Apply copper fungicide.",
    "mites":          "Spray water forcefully. Apply neem oil or miticide spray.",
    "healthy":        "Plant is healthy! Maintain regular watering, fertilization and monitoring.",
    "haunglongbing":  "No cure available. Remove infected trees immediately to prevent spread.",
    "esca":           "No cure available. Remove infected vines. Prevent wounds during pruning.",
}

def get_treatment(class_name):
    class_lower = class_name.lower()
    for key, advice in treatments.items():
        if key in class_lower:
            return advice
    return "Consult a local agricultural expert for specific treatment advice."

# ── Prediction function ────────────────────────────────
def predict(image):
    if image is None:
        return "Please upload a plant leaf image.", {}

    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert('RGB')
    else:
        img = image.convert('RGB')

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        top3_probs, top3_ids = probs.topk(3)

    top_class = class_names[top3_ids[0][0].item()]
    parts     = top_class.split('___')
    plant     = parts[0].replace('_', ' ')
    condition = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
    treatment = get_treatment(top_class)

    result  = f"🌿 Plant:      {plant}\n"
    result += f"🔍 Condition:  {condition}\n\n"
    result += f"💊 Treatment:\n{treatment}"

    confidence = {}
    for i in range(3):
        name = class_names[top3_ids[0][i].item()]
        name = name.replace('___', ' — ').replace('_', ' ')
        confidence[name] = float(top3_probs[0][i])

    return result, confidence

# ── Gradio Interface ───────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="📸 Upload Plant Leaf Image", type="pil"),
    outputs=[
        gr.Textbox(label="🩺 Diagnosis & Treatment", lines=8),
        gr.Label(label="Top 3 Predictions", num_top_classes=3)
    ],
    title="🌿 Plant Disease Classifier",
    description="""
**Developed by Faisal Imam** | AI Engineer

🔗 [GitHub](https://github.com/faisalimam1) | [Kaggle](https://www.kaggle.com/faisalimam19) | [LinkedIn](https://www.linkedin.com/in/faisalimam19)

Upload a photo of a plant leaf to instantly identify diseases and get treatment recommendations.

Trained on 70,000+ images across 38 plant disease categories.
**Model:** ResNet-18 (Transfer Learning) | **Accuracy:** 99.07%

*Built as part of a 30-day AI Engineer Roadmap*
    """
)

# ── Launch ─────────────────────────────────────────────
demo.launch(theme=gr.themes.Soft())
