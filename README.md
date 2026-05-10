# 🌿 Plant Disease Classifier

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/faisalimam19/plant-disease-classifier)
[![Model](https://img.shields.io/badge/Model-ResNet--18-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.07%25-brightgreen?style=for-the-badge)]()
[![Classes](https://img.shields.io/badge/Classes-38-blue?style=for-the-badge)]()

> **AI-powered plant disease detection for Indian farmers.**
> Upload a photo of a plant leaf → get instant disease diagnosis + treatment recommendation.

---

## 🎯 Problem Being Solved

Indian farmers lose **20-30% of crops annually** to diseases they cannot identify early enough. Early detection enables targeted treatment and saves crops — but rural farmers rarely have access to agricultural experts.

This classifier lets a farmer point their phone camera at a leaf and instantly know:
- What disease is affecting the plant
- How confident the model is
- What treatment to apply immediately

---

## 🚀 Live Demo

**Try it here:** [huggingface.co/spaces/faisalimam19/plant-disease-classifier](https://huggingface.co/spaces/faisalimam19/plant-disease-classifier)

![Sample Predictions](predictions.png)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Images | 70,295 |
| Validation Images | 17,572 |
| Total Classes | 38 |
| Best Validation Accuracy | **99.07%** |
| Training Epochs | 5 |
| Training Time | ~16 minutes (T4 GPU) |

![Training Curve](training_curve.png)

---

## 🌱 Supported Plants & Diseases

| Plant | Diseases Covered |
|-------|-----------------|
| Apple | Scab, Black Rot, Cedar Rust, Healthy |
| Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Peach | Bacterial Spot, Healthy |
| + More | Blueberry, Cherry, Orange, Raspberry, Soybean, Squash, Strawberry |

---

## 🛠️ Tech Stack
