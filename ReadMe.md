# Skandia AI-Based Vision System for Solder Joint Quality Classification

This repository contains the complete implementation of Skandia's AI-based computer vision solution designed to automate the classification of solder joint quality in real-time. It includes code for preprocessing, training, testing, API serving, and dataset documentation.

---

## 📌 Overview

Skandia's vision system leverages advanced computer vision and machine learning to detect solder joint defects at the production stage. This system improves inspection efficiency and consistency by:

- Reducing material waste  
- Enhancing product yield  
- Minimizing post-production rework  
- Automating quality assurance in real-time

---

## 🗂️ Key Files and Directories

**API Components**  
- `api/api.py` - FastAPI server for predictions  
- `api/api_call.py` - Example API client  
- `api/results/` - Saved inference results  
- `api/temp_uploads/` - Temporary image uploads  

**Model Development**  
- `model/datasets/` - Training/validation data  
- `model/model_testing/testing.py` - Evaluation script  
- `model/preprocessing/` - Data preparation scripts  

**Documentation**  
- `model/dataset.md` - Dataset specifications  
- `model/model.md` - Architecture details  
- `model/preprocessing.md` - Preprocessing guide  

---

## 🚀 Getting Started

### 1. Install Required Packages

```bash
pip install -r requirements.txt

