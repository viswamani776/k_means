import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="K-Means Clustering App")

st.title("🧠 K-Means Customer Segmentation")
st.write("Clustering based on numeric customer features")

# ================= LOAD MODEL =================
if not os.path.exists("kmeans_model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("❌ Model or scaler file not found")
    st.stop()

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.success("✅ Model loaded successfully")

st.divider()

# ================= DYNAMIC INPUTS =================
n_features = scaler.n_features_in_

st.subheader("Enter Feature Values")

inputs = []
for i in range(n_features):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

# ================= PREDICTION =================
if st.button("🔍 Predict Cluster"):
    try:
        data = np.array(inputs).reshape(1, -1)
        data_scaled = scaler.transform(data)
        cluster = kmeans.predict(data_scaled)

        st.success(f"✅ Customer belongs to Cluster: {cluster[0]}")
    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
