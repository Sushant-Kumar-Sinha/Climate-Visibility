# Climate Visibility Prediction Using Machine Learning 🌤️✈️

An AI-powered weather visibility prediction system designed to enhance safety across critical sectors by forecasting exact atmospheric visibility levels based on multi-vector weather metrics.

[![Deployment Status](https://img.shields.io/badge/Deployment-Hugging%20Face-blue)](https://huggingface.co/spaces/Sushant-Kumar-Sinha/climate-visibility)

---

## 📌 Project Overview

Weather visibility directly dictates operational safety across key global infrastructures. This project addresses the complex, non-linear challenges of visibility forecasting through a robust, high-performance **Two-Stage Machine Learning Prediction Engine**. 

### Critical Impacts & Sectors Served:
* **Aviation:** Supporting flight safety and landing decisions.
* **Road Transportation:** Preventing multi-vehicle accidents due to sudden low visibility.
* **Maritime:** Assisting in safe navigation and port operations.
* **Public Events:** Enhancing security for outdoor public gatherings and hiking.

---

## ⚙️ Architecture & Methodology

The system breaks away from standard single-model designs by introducing a **Modular Two-Stage Engine** to overcome severe data skew (93:7 class imbalance) and provide precise results.

### 🧠 Two-Stage Prediction Engine
1. **Stage 1 (Binary Classifier):** Determines if the visibility condition is **Clear** or **Reduced** based on a strict threshold of **8.0 km**.
   * **Algorithm:** Random Forest Classifier / XGBoost (100 Estimators, Max Depth: 15)
   * **Strategy:** Balanced class weights are utilized to successfully handle data skew.
2. **Stage 2 (Regressor):** Operates only when Stage 1 indicates a reduced scenario, computing the exact visibility mapping in kilometers.
   * **Algorithm:** Random Forest Regressor / XGBoost (100 Estimators, Max Depth: 15)
   * **Normalization:** `StandardScaler` applied across all features.

### 📊 Dataset Specifications
* **Training Data:** 75,000+ Atmospheric Samples
* **Data Split:** 80% Training | 20% Stratified Testing Split
* **Feature Set:** 8 Core Weather Parameters (including Temperature, Humidity, Wind Speed, and Atmospheric Pressure)

---

## 📈 Model Performance & Validation

The system achieves high robustness even when stress-tested across extreme weather conditions like dense fog, thunderstorms, and heavy snow.

* **Overall Classification Accuracy:** 94%
* **Reduced Condition Accuracy:** 85% (Achieved despite severe 93:7 class imbalance)
* **Mean Absolute Error (MAE):** 0.86 km
* **Root Mean Square Error (RMSE):** 2.02 km
* **R² Score:** 0.67

### Confusion Matrix
| Metric | Clear (Predicted) | Reduced (Predicted) |
| :--- | :--- | :--- |
| **True Clear** | 13,298 | 747 (False Reduced) |
| **True Reduced** | 143 (False Clear) | 829 |

---

## 💻 Tech Stack & Deployment

* **Frontend:** Responsive HTML5/CSS3 interface featuring a dark-themed UI with fluid JavaScript animations (**Vista-Animated Horizon** dashboard).
* **Backend:** Flask REST API equipped with CORS security and highly optimized inference endpoints.
* **Deployment:** Live on Hugging Face Spaces.
* **Production Interface Link:** [Access Live Space](https://huggingface.co/spaces/Sushant-Kumar-Sinha/climate-visibility)

---

## 🚀 Future Roadmap

* **Aviation & Safety Integration:** Direct pipeline coupling with flight planning software and real-time airport radar instruments.
* **Smart Mobility APIs:** Low-latency API architecture support for Autonomous Vehicles (AVs) navigating foggy routes.
* **Deep Learning Upgrade:** Transitioning the architecture to advanced Long Short-Term Memory (LSTM) time-series networks for 48-hour continuous forecasting.

---

## 👥 Team Members

* **Sushant Kumar Sinha**
* **Yash Kumar**
* **Tushar Kumar**
* **Sushant Kumar Sahu**
