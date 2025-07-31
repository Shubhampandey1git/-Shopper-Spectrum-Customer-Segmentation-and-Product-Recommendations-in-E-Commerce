import streamlit as st
import joblib
import numpy as np

# Load models
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
cluster_labels = joblib.load('cluster_labels.pkl')
similarity_df = joblib.load('similarity_df.pkl')
product_names = joblib.load('product_names.pkl')

# Helper function
def get_similar_product_names(product_code, top_n=5):
    if product_code not in similarity_df.columns:
        return []
    similar_items = similarity_df[product_code].sort_values(ascending=False)[1:top_n+1]
    result = []
    for code, score in similar_items.items():
        name = product_names.get(code, "Unknown Product")
        result.append((code, name, round(score, 3)))
    return result

# App Layout
st.set_page_config(page_title="Shopper Spectrum", layout="centered")
st.title("🛒 Shopper Spectrum")

module = st.sidebar.selectbox("Select Module", ["Product Recommendation", "Customer Segmentation"])

# 🎯 Module 1: Product Recommendation
if module == "Product Recommendation":
    st.header("🔍 Product Recommendation")
    input_name = st.text_input("Enter Product Name (e.g., HEART T-LIGHT HOLDER)").strip().lower()

    # Find best match from product descriptions
    selected_code = None
    if input_name:
        matches = product_names[product_names.str.lower().str.contains(input_name)]
        if not matches.empty:
            selected_code = matches.index[0]
            st.markdown(f"🔍 Matched Product: **{matches.iloc[0]}** (Code: `{selected_code}`)")
        else:
            st.warning("No matching product found.")

    if st.button("Get Recommendations"):
        if selected_code:
            recommendations = get_similar_product_names(selected_code)
            if recommendations:
                st.success("Recommended Products:")
                for code, name, score in recommendations:
                    st.markdown(f"- **{name}** (Code: `{code}`, Similarity: {score})")
            else:
                st.warning("No similar products found.")
        else:
            st.error("Please enter a valid product name.")


# 🎯 Module 2: Customer Segmentation
elif module == "Customer Segmentation":
    st.header("📊 Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
    monetary = st.number_input("Monetary (total spend ₹)", min_value=0.0, value=500.0)

    if st.button("Predict Segment"):
        input_scaled = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(input_scaled)[0]
        segment = cluster_labels.get(cluster, "Unknown Segment")
        st.success(f"🧠 Predicted Segment: **{segment}**")