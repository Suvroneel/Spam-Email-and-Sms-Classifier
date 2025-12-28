# Spam Email Classifier  
### End-to-End Machine Learning Pipeline

![Image](https://github.com/user-attachments/assets/439129bc-b593-4926-9faa-0d5fe90c7e57)

---

## Overview

This project implements an end-to-end **spam email classification system** using NLP and machine learning techniques.  
It covers the full lifecycle — preprocessing, feature extraction, model training, evaluation, and web-based inference.

The system is deployed as a **Streamlit application**, with ongoing work to integrate a **CNN-based deep learning model** and improve production readiness.

---

## Tech Stack

- **Languages & Libraries:** Python, scikit-learn, pandas  
- **Deployment:** Streamlit  
- **Utilities:** Google Sheets API, Pickle serialization  

---

## Core Features

### Text Preprocessing
- Lowercasing
- Tokenization
- Special character removal
- Stemming

![Image](https://github.com/user-attachments/assets/ec0fa2e2-74ae-4002-9b42-b36d17f02930)

---

### NLP Pipeline
- TF-IDF vectorization
- Multinomial Naïve Bayes classifier
- Scikit-learn pipelines for reproducible preprocessing and inference

---

### Exploratory Data Analysis

**Spam Word Cloud**

![Image](https://github.com/user-attachments/assets/197f20ed-fe33-4267-b060-551b21fdacef)

**Ham Word Cloud**

![Image](https://github.com/user-attachments/assets/96ca117e-85f1-4c41-ac03-2eb909f5e688)

---

### Deployment & Logging
- Streamlit-based web interface for real-time predictions
- User inputs and predictions logged via Google Sheets API
- Serialized models for consistent inference

![Image](https://github.com/user-attachments/assets/4807df2a-7687-42a1-b4a0-aa972b17a490)

---

## Business Impact

- High-accuracy spam detection on validation data  
- Continuous data collection for iterative model improvement  
- Modular pipeline design for future extensibility  

---

## Ongoing Work

- Integrating a **CNN-based spam classifier** using learned embeddings  
- Exploring **character-level CNN architectures (Char-CNN)**  
- Comparing deep learning performance against classical NLP baselines  
- Improving deployment robustness and inference latency  

---

## Notes

- Model training and experimentation are documented in the provided notebook(s).
- Trained model weights are excluded due to size constraints and can be regenerated using the training pipeline.

---
