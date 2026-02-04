# 🚗 Car Price Prediction

A machine learning web application built with **FastAPI** (backend) and **HTML/JavaScript** (frontend) that predicts the selling price of used cars based on features like brand, year, fuel type, seller type, kilometers driven, transmission, and ownership history.

---

## ✨ Features
- FastAPI backend with trained **Linear Regression** model.
- REST API endpoint `/predict` for car price prediction.
- Frontend form built with HTML, CSS, and JavaScript.
- Swagger UI (`/docs`) for testing API requests.
- Handles unseen labels gracefully with safe encoding.

---

## 🛠️ Tech Stack
- **Python** (FastAPI, Pandas, Scikit-learn)
- **Frontend**: HTML, CSS, JavaScript
- **Model**: Linear Regression
- **Tools**: Git, VS Code

---

## 📂 Project Structure
car_price_prediction/
│
├── main.py               # FastAPI backend

├── car_dataset.csv      # Training dataset

├── index.html            # Frontend UI

├── script.js             # Frontend logic

├── style.css             # Styling

└── README.md             # Project documentation


## 🚀 Get Started to Run this

### 1. Clone the repository
git clone https://github.com/Ansarali156/car_price_prediction.git
cd car_price_prediction

### 2. Install dependencies
pip install fastapi uvicorn pandas scikit-learn

### 3. Run the backend
uvicorn main:app --reload
Backend will start at: http://127.0.0.1:8000

### 4. Test API
Open Swagger docs:
http://127.0.0.1:8000/docs

### 5. Run frontend
Open index.html in your browser (or use VS Code Live Server).
