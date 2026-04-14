# 🔍🤖 Defect Detection using AI

An **AI-powered defect detection system** built using deep learning to automatically identify defects from input data and store results for analysis.

This project implements a **complete machine learning pipeline** — from dataset configuration and model training to inference and result persistence — following a **modular, reproducible, and scalable design**.

---

## 🚀 Key Features

* ✅ Deep learning-based defect detection
* ✅ End-to-end ML pipeline (**data → training → inference → storage**)
* ✅ Modular and extensible Python architecture
* ✅ YAML-driven dataset configuration
* ✅ Lightweight and easy to deploy
* ✅ SQLite-based result storage system

---

## 🧠 System Overview

The system follows a standard **AI pipeline architecture**:

1. **Data Ingestion**

   * Dataset paths and labels defined in `data.yaml`
   * Flexible dataset configuration

2. **Model Training**

   * Learns defect patterns using labeled data
   * Optimizes weights via backpropagation

3. **Inference Engine**

   * Loads trained model (`best.pt`)
   * Performs prediction on new data

4. **Result Storage**

   * Stores predictions in SQLite database
   * Enables tracking and analysis

---

## 📄 Project Report

A detailed technical report explaining architecture, implementation, and results:

👉 [View Full Report](https://drive.google.com/file/d/1UkB7i2odoNtBnfpAPt34XK_-ABe598-X/view?usp=sharing)

---

## 📁 Project Structure

```bash
Defect-Detection-using-AI/
│
├── app.py             # Inference pipeline (prediction + storage)
├── train.py           # Model training pipeline
├── database.py        # SQLite database operations
├── data.yaml          # Dataset configuration
├── requirements.txt   # Dependencies
└── README.md          # Documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Sankhacse/Defect-Detection-using-AI.git
cd Defect-Detection-using-AI
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

### 3️⃣ Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Train the Model

```bash
python train.py
```

**Internally:**

* Loads dataset from `data.yaml`
* Trains deep learning model
* Saves best weights → `best.pt`

---

### 🔹 Run Inference

```bash
python app.py
```

**Pipeline:**

* Load model
* Run prediction
* Store results in database

---

## 🧠 Model & AI Concepts

* Supervised Learning
* Feature Extraction
* Forward Propagation
* Backpropagation

### 📊 Recommended Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 🗄️ Database

* Uses **SQLite**
* Automatically created
* Stores predictions and metadata

---

## ⚡ Design Highlights

* 📦 Modular architecture
* 🔁 Reproducible via YAML config
* ⚙️ Scalable pipeline design
* 💡 Integration of AI + software engineering

---

## 🛠 Requirements

* Python **3.8+**
* Cross-platform (Windows / Linux / macOS)

---

## 📊 Future Improvements

* 🚀 Real-time detection
* 📊 Visualization dashboard
* 📈 Advanced metrics
* 🌐 Deployment (Web/API)
* ☁️ Cloud integration

---

## 🤝 Contributing

```
1. Fork the repository
2. Create a new branch
3. Commit changes
4. Submit PR
```

---

## ⭐ Support

If you find this useful:

👉 Star the repo
👉 Share it
👉 Contribute

---

## 💡 Best Practices

* Use well-labeled dataset
* Avoid class imbalance
* Tune hyperparameters
* Validate on unseen data

---

