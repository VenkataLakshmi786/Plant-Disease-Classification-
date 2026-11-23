# Plant Disease Classification Using Deep Learning

## Overview
This project focuses on classifying plant leaf images into five categories: Bacteria, Fungus, Healthy, Pests, and Virus. It explores deep learning and hybrid classification approaches to identify plant health conditions and support intelligent agricultural diagnosis.

## Dataset
- **Dataset:** [Plant Pathogens – Kaggle](https://www.kaggle.com/datasets/sujallimje/plant-pathogens)
- **Total Images:** 176,744
- **Classes:** 5
- **Image Resolution:** 224 × 224 × 3

## Project Workflow
1. **Dataset Analysis:** Perform statistical analysis and exploratory data analysis to understand image characteristics and class distribution.
2. **Preprocessing:** Prepare and resize images, apply the required transformations, and normalize image inputs.
3. **Dataset Splitting:** Divide the dataset into 80% training, 10% validation, and 10% testing.
4. **Model Development:** Implement and evaluate:
   - EfficientNet-B0
   - ResNet-SE (ResNet-50 with Squeeze-and-Excitation attention)
   - CNN-DT (CNN–Decision Tree hybrid model)
5. **Model Evaluation:** Analyse Accuracy, Precision, Recall, F1-Score, Log Loss, ROC-AUC, and other evaluation visualizations.
6. **Deployment:** Develop a Flask web application using the trained ResNet-SE model for plant pathogen classification.

## Model Performance

| Model | Accuracy | F1-Score |
|---|---:|---:|
| EfficientNet-B0 | 88.69% | 88.67% |
| ResNet-SE | 95.59% | 95.58% |
| CNN-DT | 69.26% | 69.26% |

## Technologies Used
Python, PyTorch, Torchvision, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, and Flask.

## Project Structure
```text
Plant Disease Classification/
│
├── Code.ipynb
├── Results/
│   ├── EDA/
│   └── Model Evaluation/
│
└── app/
    ├── app.py
    ├── best_resnet_se_model.pth
    ├── requirements.txt
    ├── README.txt
    ├── templates/
    │   └── index.html
    └── static/
        └── uploads/
```

## Objective
To develop and evaluate deep learning models for plant pathogen classification and deploy an image-based prediction system using Flask.
