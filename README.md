# 🏥 Healthcare Insurance Cost Prediction System

An end-to-end Machine Learning project that analyzes healthcare insurance data and predicts medical insurance costs based on demographic and health-related features.

This project includes:
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Model Building
- ML Pipeline Creation
- Streamlit Web Application Deployment

---

## 📂 Project Structure

```
Healthcare_Project/
│
├── Dataset/
│   └── insurance.csv
│
├── Notebooks/
│   ├── 1. HealthCare(Cleaning).ipynb
│   ├── 2. HealthCare(EDA).ipynb
│   └── 3. Healthcare-Final.ipynb
│
├── Pickle File/
│   └── ML_HealthCare_model.pkl
│
├── Healthcare_Streamlit/
│   ├── health.py        # Insurance Cost Prediction App
│   └── project.py       # EDA Dashboard App
│
├── Images/              # Project screenshots
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

The dataset contains the following features:

- **Age** – Age of the individual
- **Gender** – Male/Female
- **BMI** – Body Mass Index
- **Children** – Number of dependents
- **Smoker** – Smoking status
- **Region** – Residential region
- **Charges** – Medical insurance cost (Target Variable)

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Cleaning
- Removed inconsistencies
- Standardized column names
- Prepared dataset for modeling

### 2️⃣ Exploratory Data Analysis
- Univariate Analysis
- Bivariate Analysis
- Correlation Study
- Smoking vs Charges Analysis
- BMI & Age Impact Study

### 3️⃣ Model Building
- Feature Engineering
- Categorical Encoding
- Numerical Scaling
- Pipeline Implementation
- Model Training using Scikit-learn
- Model saved as `.pkl` file

---

## 💻 Web Applications

### 🔹 Insurance Cost Prediction App

Location: `Healthcare_Streamlit/health.py`

Features:
- Sidebar user input
- Automatic BMI calculation
- Real-time cost prediction
- Professional dashboard UI
- Bill-style output display

Run using:

```bash
streamlit run Healthcare_Streamlit/health.py
```

---

### 🔹 EDA Dashboard App

Location: `Healthcare_Streamlit/project.py`

Features:
- Interactive visualizations
- Data insights dashboard
- Univariate & Bivariate plots
- Clean professional layout

Run using:

```bash
streamlit run Healthcare_Streamlit/project.py
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Pickle

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone <your-repository-link>
```

### Step 2: Navigate to Project Directory

```bash
cd Healthcare_Project
```

### Step 3: Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run Streamlit App

```bash
streamlit run Healthcare_Streamlit/health.py
```

---

## 📈 Key Insights

- Smokers incur significantly higher insurance charges.
- Higher BMI is associated with increased medical expenses.
- Age shows a positive correlation with insurance costs.
- Machine Learning enables fair and data-driven premium estimation.

---

## 🚀 Future Enhancements

- Deploy on Streamlit Cloud
- Add PDF download for insurance bill
- Improve model accuracy with advanced algorithms
- Add risk-level classification system

---

## 👤 Author

**Venkatesh Karnure**  
Machine Learning & Data Analytics Enthusiast  

---

## 📄 License

This project is developed for academic and educational purposes.
