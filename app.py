# fraud_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# --- App Title & Banner ---
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
st.image("https://i.ibb.co/m0X0G6g/fraud-detection-banner.jpg", use_column_width=True)
st.title("💳 Credit Card Fraud Detection System")
st.markdown("An interactive tool to detect fraudulent transactions in real time.")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("/content/drive/MyDrive/firstproject/creditcard.csv")
    column_mapping = {f"V{i}": f"Feature_{i}" for i in range(1, 29)}
    df.rename(columns=column_mapping, inplace=True)
    return df

df = load_data()

# --- Show Dataset ---
st.subheader("📊 Dataset Overview")
if st.checkbox("Show raw data"):
    st.write(df.head())

# --- Data Info ---
st.subheader("🔍 Class Distribution")
class_counts = df['Class'].value_counts()
st.bar_chart(class_counts)

# --- Train Model ---
@st.cache_resource
def train_model(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(df)

# --- Model Evaluation ---
st.subheader("📈 Model Performance")
y_pred = model.predict(X_test)
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# --- User Input for Prediction ---
st.subheader("📝 Check a Transaction")
user_data = {}
for col in df.columns[:-1]:  # Skip Class column
    user_data[col] = st.number_input(f"Enter {col}", value=0.0)

if st.button("Predict Fraud"):
    input_df = pd.DataFrame([user_data])
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("🚨 This transaction is **Fraudulent**!")
    else:
        st.success("✅ This transaction is **Legit**.")

# --- Save model ---
joblib.dump(model, "fraud_model.pkl")
