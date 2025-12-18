# Defect Detection using AI 🔍🤖

This project is an AI-powered defect detection system that uses deep learning to identify defects from input data and store detection results for further analysis. The system supports model training, inference, and result storage in a clean and reproducible workflow.

---

## 🚀 Key Features

- Automated defect detection using a trained deep learning model
- Model training and inference support
- Lightweight and modular Python codebase
- Configurable dataset using YAML
- Detection results stored using a local database

---

## 📂 Project Structure

├── app.py # Run inference / detection
├── train.py # Train the defect detection model
├── database.py # Database operations
├── data.yaml # Dataset and configuration
├── requirements.txt # Project dependencies

---

## ⚙️ Setup Instructions

1️⃣ Clone the repository: 
 - git clone https://github.com/Sankhacse/Defect-Detection-using-AI.git
 - cd Defect-Detection-using-AI

2️⃣ Create and activate a virtual environment:
 - python -m venv venv
 - Windows: venv\Scripts\activate
 - macOS / Linux: source venv/bin/activate

3️⃣ Install dependencies:
 - pip install -r requirements.txt

4️⃣ Usage:
 - Train the model - python train.py
 - Run - python app.py

---

## 🧠 Model Information

 - The trained model (best.pt) is generated automatically during training
 - Model files are intentionally excluded from version control
 - It can be retrained using own dataset and configuration defined in data.yaml

---

## 🗄️ Database

 - Detection results are stored in a local SQLite database
 - Database files are auto-generated at runtime
 - Database files are excluded from version control to keep the repository clean
 - The database can be safely deleted and regenerated if required

---

## 🛠 Requirements

 - Python 3.8 or higher
 - Required Python packages listed in requirements.txt
 - Compatible with Windows, Linux, and macOS

---

## 📈 Future Enhancements

 - Real-time defect detection support
 - Visualization of detection results
 - Model performance evaluation metrics
 - Deployment as a web or desktop application

---

## 👨‍💻 Author
Sankha Subhra Mandal<br>
Computer Science & Engineering<br>
IIT (BHU) Varanasi

