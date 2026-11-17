
# PneumoVision AI

## Deep Learning-Based Pneumonia Detection from Chest X-Ray Images Using CNN and Explainable AI

PneumoVision AI is an academic deep learning prototype that classifies chest X-ray images into Normal and Pneumonia categories using a custom Convolutional Neural Network (CNN).

It also uses Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize image regions that positively influence the model's Pneumonia score.

## Features

- Upload chest X-ray images in JPG, JPEG, or PNG format.
- CNN-based image classification.
- Pneumonia probability display.
- Validation-selected classification threshold.
- Grad-CAM heatmap generation.
- Grad-CAM overlay visualization.
- Streamlit web interface.

## Model Architecture

- Input: 224 × 224 grayscale image
- Five convolutional blocks
- Convolution, Batch Normalization, ReLU and Max Pooling
- Global Average Pooling
- Dense feature layer
- Sigmoid output layer

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pillow
- Matplotlib
- Streamlit

## Project Files

- `app.py` — Streamlit application
- `pneumovision_cnn_best.keras` — trained CNN checkpoint
- `threshold_config.json` — classification threshold
- `requirements.txt` — application dependencies

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Dataset

The model was developed using the Chest X-Ray Images (Pneumonia) dataset by Paul Mooney, available on Kaggle.

The dataset contains pediatric chest X-ray images labelled Normal and Pneumonia.

## Research Limitations

- The dataset is pediatric and does not establish performance on adult populations.
- Filename-based grouping was used to reduce potential overlap between training and evaluation splits. This is not a verified patient-level split.
- Model performance may be affected by dataset-specific characteristics and distribution differences.
- Grad-CAM visualizations do not prove clinical reasoning or identify confirmed disease locations.

## Disclaimer

This project is developed for academic and educational purposes only. It is not a medical device and must not be used for diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical assessment.
