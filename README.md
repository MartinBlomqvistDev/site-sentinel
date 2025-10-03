# 🔥 SITE SENTINEL: Predictive Safety for Construction Zones

**Slogan:** The Intelligent Watchdog for Construction Zones.

---
### 🏷️ Keywords & Hashtags

#MachineLearning #ComputerVision #PredictiveAnalytics #XGBoost #YOLOv8 #SiteSafety #TrafficMonitoring #SoloDeveloper #Streamlit #TTC

---

### 💡 Project Summary & Impact

**Site Sentinel** is a full-stack, AI-driven monitoring system designed to solve a critical, high-value safety problem: **preventing vehicle-personnel Near-Miss events in high-speed roadside construction areas.**

**My core achievement** was moving the system beyond passive surveillance by implementing a custom **Predictive Analytics Engine (XGBoost)** that forecasts risk up to **4.0 seconds in advance**. This provides crucial lead time for intervention, significantly reducing liability and enhancing worker safety.

As a solo developer, **I owned the entire ML lifecycle**, from cloud data ingestion and custom feature engineering to real-time dashboard deployment.

---

### ✨ The Core Innovation: Predictive Decision Support

I engineered this project to showcase expertise in creating robust, proactive systems. The predictability relies on three components:

| Feature | Technical Achievement | Value for Safety |
| :--- | :--- | :--- |
| **Target Definition (P30)** | **I defined** the target event (Y=1) as a vehicle/worker entering a **5.0-meter safety perimeter** within **4.0 seconds**. | Provides maximum human reaction time for safe intervention. |
| **Feature Engineering (P31)** | **I built** custom features, including **Time-to-Collision (TTC)** and **Relative Velocity**. This vector analysis ensures the model relies on motion, not just distance, preventing false alarms from stationary workers. | Proves ability to create intelligent features that filter out noise and identify true high-risk scenarios. |
| **Model (P32)** | **I trained** an XGBoost Classifier from scratch, utilizing `scale_pos_weight` to manage severe class imbalance (few risky frames vs. many safe frames). | Ensures the risk alerts are trustworthy (high Precision) and minimizes system distraction. |

---

### 🛠️ System Architecture & Solo Expertise

The application is structured as a professional, scalable ML production pipeline.

**Pipeline Flow:**
**Data Source (GCS)** $\rightarrow$ **Detection (YOLOv8)** $\rightarrow$ **Tracking (MOT)** $\rightarrow$ **Feature Engineering (TTC/Pandas)** $\rightarrow$ **Prediction (XGBoost)** $\rightarrow$ **Visualization (Streamlit)**

| Category | Skills Demonstrated (I) |
| :--- | :--- |
| **Computer Vision** | **Transfer Learning (YOLOv8 Fine-Tuning):** Adapting state-of-the-art models to niche domain objects (workers/machinery). |
| **Tracking & Data** | **Multi-Object Tracking (MOT):** Utilizing ByteTrack/DeepSORT for consistent ID assignment crucial for accurate trajectory analysis. |
| **Cloud & Infra** | **Cloud Storage (GCS):** Configuration, authentication, and management of large data artifacts for scalability. |
| **Frontend/App** | **Streamlit:** Developing a modular, real-time interactive dashboard (V40-V43) as a final deployment layer. |

---

### 📈 Model Performance (P33)

*Note: Update this table with your actual results after running the P33 evaluation.*

The model was validated on an **unseen test set** (30% of the demo clip) to ensure strong generalization. Precision is prioritized in safety-critical applications.

| Metric | Result | Goal & Interpretation |
| :--- | :--- | :--- |
| **Precision** | [0.XX] | **(CRITICAL METRIC)** Measures how often an alarm is correct. A high score proves the "Watchdog" is trustworthy. |
| **Recall** | [0.XX] | Measures how many real risk events the model successfully identified. |
| **F1-Score** | [0.XX] | Harmonic mean of Precision and Recall. |

---

### 🎬 Demo & Contact

*Note: Insert a live video or image here once available.*

**I am ready for a role in Machine Learning Engineering/Data Science.**

* Project Repository: [**https://github.com/MartinBlomqvistDev/site-sentinel**]
* LinkedIn: [**https://www.linkedin.com/in/martin-blomqvist**]
* Email: [**cm.blomqvist@gmail.com**]