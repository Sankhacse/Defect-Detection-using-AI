# 🔍🤖 Defect Detection using AI

An **AI-powered defect detection system** built using deep learning to automatically identify defects from input data and store results for analysis.
Designed with a **modular, reproducible, and scalable workflow**, this project supports both training and inference.

---

## 🚀 Features

✅ Automated defect detection using deep learning
✅ End-to-end pipeline (training → inference → storage)
✅ Clean and modular Python architecture
✅ YAML-based dataset configuration
✅ Lightweight and easy to extend
✅ SQLite-based local result storage

---

## 📁 Project Structure

```
Defect-Detection-using-AI/
│
├── app.py            # Run inference / detection
├── train.py          # Train the defect detection model
├── database.py       # Handle database operations
├── data.yaml         # Dataset configuration
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```

---

## ⚙️ Installation & Setup

Follow these steps carefully:

### 1️⃣ Clone the Repository

```
git clone https://github.com/Sankhacse/Defect-Detection-using-AI.git
cd Defect-Detection-using-AI
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
```

---

### 3️⃣ Activate Virtual Environment

**Windows:**

```
venv\Scripts\activate
```

**macOS / Linux:**

```
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Train the Model

```
python train.py
```

* Loads dataset from `data.yaml`
* Trains the model
* Saves the best model as `best.pt`

---

### 🔹 Run Inference (Detection)

```
python app.py
```

* Loads trained model
* Performs defect detection
* Stores results in database

---

## 🧠 Model Details

* Model file: `best.pt`
* Automatically generated after training
* Not included in GitHub (keeps repo lightweight)
* Can be retrained using custom datasets via `data.yaml`

---

## 🗄️ Database

* Uses **SQLite** for storing detection results
* Automatically created at runtime
* Safe to delete (will regenerate automatically)

---

## 🛠 Requirements

* Python **3.8+**
* Windows / Linux / macOS

Dependencies are listed in:

```
requirements.txt
```

---

## 📊 Future Improvements

🚀 Real-time defect detection
📊 Visualization dashboard (graphs/UI)
📈 Model performance metrics (accuracy, precision, recall)
🌐 Deployment (Web App / Desktop App)
☁️ Cloud integration

---

## 🤝 Contributing

Contributions are welcome!

```
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request
```

---

## 📜 License

This project is open-source and available under the MIT License.

---

## ⭐ Support

If you find this project useful:

👉 Star the repository
👉 Share with others
👉 Contribute improvements

---

## 💡 Tip

For best results:

* Use a well-labeled dataset
* Tune hyperparameters in `train.py`
* Adjust configuration in `data.yaml`

---

🔥 *Built with passion for AI & problem solving*
