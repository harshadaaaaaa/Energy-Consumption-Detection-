# =========================================
# Title: Energy Consumption Prediction
# Models: Ridge Regression & Random Forest
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Load + Preprocess Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Energy_consumption.csv")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
    df['Hour'] = df['Timestamp'].dt.hour
    df['Month'] = df['Timestamp'].dt.month

    df['HVACUsage'] = df['HVACUsage'].map({'On':1, 'Off':0})
    df['LightingUsage'] = df['LightingUsage'].map({'On':1, 'Off':0})
    df['Holiday'] = df['Holiday'].map({'Yes':1, 'No':0})

    df = pd.get_dummies(df, columns=['DayOfWeek'], drop_first=True)
    df = df.drop('Timestamp', axis=1)

    return df

df = load_data()

# -----------------------------
# 2. Prepare Data
# -----------------------------
X = df.drop('EnergyConsumption', axis=1)
y = df['EnergyConsumption']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Train Models
# -----------------------------
ridge = Ridge()
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -----------------------------
# 4. Predictions
# -----------------------------
pred_ridge = ridge.predict(X_test)
pred_rf = rf.predict(X_test)

# -----------------------------
# 5. Evaluation Function
# -----------------------------
def get_metrics(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred)
    )

# -----------------------------
# 6. UI Title
# -----------------------------
st.title("⚡ Energy Consumption Prediction System")

# -----------------------------
# 7. Data Analysis Section
# -----------------------------
st.header("📊 Data Analysis")

# Energy vs Hour (cleaned)
hourly_avg = df.groupby('Hour')['EnergyConsumption'].mean()

fig1 = plt.figure()
plt.plot(hourly_avg.index, hourly_avg.values, marker='o')
plt.xlabel("Hour")
plt.ylabel("Average Energy")
plt.title("Average Energy Consumption vs Hour")
st.pyplot(fig1)

# Temperature vs Energy (with trend line)
fig2 = plt.figure()
plt.scatter(df['Temperature'], df['EnergyConsumption'], alpha=0.4)

z = np.polyfit(df['Temperature'], df['EnergyConsumption'], 1)
p = np.poly1d(z)
plt.plot(df['Temperature'], p(df['Temperature']), linewidth=2)

plt.xlabel("Temperature")
plt.ylabel("Energy")
plt.title("Temperature vs Energy with Trend Line")
st.pyplot(fig2)

# Correlation Heatmap
# fig3 = plt.figure(figsize=(10,6))
# sns.heatmap(df.corr(), cmap="coolwarm")
# plt.title("Correlation Heatmap")
# st.pyplot(fig3)

st.markdown("---")

# -----------------------------
# 8. Model Selection
# -----------------------------
model_choice = st.selectbox(
    "Choose Model",
    ["Ridge Regression", "Random Forest"]
)

# -----------------------------
# 9. Model Performance
# -----------------------------
if model_choice == "Ridge Regression":
    mae, rmse, r2 = get_metrics(y_test, pred_ridge)
    model = ridge
    preds = pred_ridge
else:
    mae, rmse, r2 = get_metrics(y_test, pred_rf)
    model = rf
    preds = pred_rf

st.header("📈 Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.2f}")

# Actual vs Predicted
fig4 = plt.figure()
plt.scatter(y_test, preds)
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("Actual vs Predicted")
st.pyplot(fig4)

st.markdown("---")

# -----------------------------
# 10. Feature Importance
# -----------------------------
if model_choice == "Random Forest":
    st.header("🌳 Feature Importance")

    importance = rf.feature_importances_
    features = X.columns

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True)

    # Table
    st.subheader("📋 Feature Importance Table")
    st.dataframe(imp_df)

    # Graph
    st.subheader("📊 Feature Importance Graph")
    fig5 = plt.figure()
    plt.barh(imp_df['Feature'], imp_df['Importance'])
    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")
    st.pyplot(fig5)

st.markdown("---")

# -----------------------------
# 11. Prediction Section
# -----------------------------
st.header("🔮 Predict Energy Consumption")

temp = st.slider("Temperature", 0.0, 50.0)
humidity = st.slider("Humidity", 0.0, 100.0)
sqft = st.slider("Square Footage", 100, 5000)
occupancy = st.slider("Occupancy", 0, 100)

hvac = st.selectbox("HVAC Usage", ["On", "Off"])
light = st.selectbox("Lighting Usage", ["On", "Off"])
holiday = st.selectbox("Holiday", ["Yes", "No"])

time_input = st.time_input("Select Time")

hour = time_input.hour

month_dict = {
    "January":1, "February":2, "March":3, "April":4,
    "May":5, "June":6, "July":7, "August":8,
    "September":9, "October":10, "November":11, "December":12
}
month_name = st.selectbox("Month", list(month_dict.keys()))
month = month_dict[month_name]

# Convert categorical
hvac = 1 if hvac == "On" else 0
light = 1 if light == "On" else 0
holiday = 1 if holiday == "Yes" else 0

# Align input
input_dict = dict.fromkeys(X.columns, 0)
input_dict.update({
    'Temperature': temp,
    'Humidity': humidity,
    'SquareFootage': sqft,
    'Occupancy': occupancy,
    'HVACUsage': hvac,
    'LightingUsage': light,
    'Holiday': holiday,
    'Hour': hour,
    'Month': month
})

input_df = pd.DataFrame([input_dict])

# Predict
if st.button("Predict Energy"):
    prediction = model.predict(input_df)
    st.success(f"⚡ Predicted Energy Consumption: {prediction[0]:.2f}kWh")