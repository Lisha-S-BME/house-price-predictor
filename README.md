# 🏠 House Price Prediction

An end-to-end machine learning project that predicts house prices with **93.39% accuracy** and deploys it as an interactive web application.

**Live Demo:** [Click Here to Try the App](https://house-price-predictor-lisha-s-bme.streamlit.app)

---

## 📌 **What is This Project?**

This project solves a common real-world problem: **estimating the fair market value of a property** based on its features. Instead of guessing or using outdated comparisons, this model provides data-driven price predictions.

For **real estate businesses**, this means:
- Confident pricing decisions
- Reduced time on market
- Competitive advantage

This is a **complete data science pipeline** from raw data to production deployment.

---

## 🛠️ **Technology Stack**

| Category | Tools & Libraries |
|----------|-------------------|
| **Programming Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (Linear Regression, Random Forest, XGBoost) |
| **Model Serialization** | Joblib |
| **Web Framework** | Streamlit |
| **Deployment** | Streamlit Cloud |
| **Version Control** | GitHub |

---

## 🚀 **How It Works**

### 1. **Data Loading & Cleaning**
- Loaded 545 property records with 13 features
- No missing values found
- Converted categorical variables using one-hot encoding
- Engineered new features: `area_per_room`, `price_per_sqft`

### 2. **Exploratory Data Analysis**
- Visualized price distribution
- Created correlation heatmap to identify key features
- Analyzed feature relationships with target variable

### 3. **Model Building**
Trained and compared multiple models:

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Linear Regression | 88.67% | $497,561 | $756,632 |
| Ridge Regression | 87.96% | $528,663 | $775,524 |
| Lasso Regression | 74.63% | $679,453 | $1,072,162 |
| **Random Forest** | **93.39%** | **$289,233** | **$578,142** |

**Best Model:** Random Forest

### 4. **Feature Importance Analysis**

| Feature | Importance |
|---------|------------|
| Area | 51.22% |
| Bathrooms | 1.87% |
| Parking | 1.11% |
| Bedrooms | 0.46% |

**Key Insight:** Property area alone predicts more than half of the price variation!

### 5. **Deployment**
The trained model is saved as `model.joblib` and served through a Streamlit web app with interactive sliders.

---

## 🧠 **Challenges Faced & Solutions**

| Challenge | Solution |
|-----------|----------|
| **Categorical Data Encoding** | Used `pd.get_dummies()` for one-hot encoding |
| **Feature Mismatch in App** | Aligned all 15 features in the prediction pipeline |
| **Pickle Version Conflict** | Switched from `pickle` to `joblib` for better cross-version compatibility |
| **Cloud Deployment Errors** | Added `requirements.txt` to ensure correct library versions |
| **Missing `price_per_sqft` in App** | Added as a placeholder with value 0 during prediction |

### Code Breaking in `app.py`
**Problem:** The model expected 15 features, but only 14 were passed.
**Fix:** Added `price_per_sqft = 0` as the 15th feature in the input array.

```python
features = np.array([[
    area, bedrooms, bathrooms, stories, parking,
    mainroad_yes, guestroom_yes, basement_yes,
    hotwaterheating_yes, ac_val, prefarea_val,
    furnishingstatus_semi, furnishingstatus_unfurnished,
    area_per_room, price_per_sqft  # 15th feature
]])
