"""
Step 5: Polished layout.

Same prediction logic as Step 4 -- this step only changes how it's
presented, so it reads well to someone who's never seen the project.

To run this:
    streamlit run app.py
"""

import streamlit as st
import joblib
import datetime
import pandas as pd

# st.set_page_config() controls browser-tab-level settings. It MUST be
# the very first Streamlit command called in the script, before any
# other st.* call -- Streamlit enforces this and will error otherwise.
st.set_page_config(page_title="Late Delivery Predictor", layout="centered")

# Files (model, encoders, feature list) are expected to sit in the
# same folder as this script. Using a relative path (just the
# filename, no folder prefix) means this works identically whether
# run locally on your machine or deployed on Streamlit Community
# Cloud -- a hardcoded local path like "C:/Users/..." would only ever
# work on your own computer, since that folder won't exist on a
# deployment server.

st.title("📦 Late Delivery Predictor")
st.write(
    "Estimate the likelihood that an order will be delivered late, "
    "based on shipping and order details."
)

# ------------------------------------------------------------------
# @st.cache_resource: load once, reuse across every rerun -- see
# Step 4 notes for why this matters for a 45MB model file.
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("late_delivery_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, encoders, feature_cols

model, encoders, feature_cols = load_artifacts()

# ------------------------------------------------------------------
# Sidebar -- context that isn't an input. Anything called on
# st.sidebar.xxx() renders in the left panel instead of the main page.
# ------------------------------------------------------------------
with st.sidebar:
    st.header("About this tool")
    st.write(
        "This model was trained on historical order data from a "
        "single company's supply chain operations (the DataCo "
        "Smart Supply Chain dataset). It reflects that company's "
        "specific shipping carriers, regions, and product catalog -- "
        "it is **not** a general-purpose delivery predictor."
    )
    st.write(
        "The strongest driver of predicted risk, by a wide margin, "
        "is **shipping mode** -- in the training data, orders shipped "
        "'First Class' were late almost every time."
    )

# ------------------------------------------------------------------
# Inputs, laid out in two side-by-side columns instead of one long
# vertical stack. st.columns(2) returns two column objects; anything
# called on col1 renders in the left half, col2 in the right half.
# ------------------------------------------------------------------
st.subheader("Order Details")

col1, col2 = st.columns(2)

with col1:
    shipping_mode = st.selectbox("Shipping Mode", encoders["shipping_mode"].classes_)
    category_name = st.selectbox("Product Category", encoders["category_name"].classes_)
    customer_segment = st.selectbox("Customer Segment", encoders["customer_segment"].classes_)
    order_date = st.date_input("Order Date", value=datetime.date.today())

with col2:
    order_region = st.selectbox("Order Region", encoders["order_region"].classes_)
    quantity = st.number_input("Quantity", min_value=1, max_value=100, value=1)
    product_price = st.number_input("Product Price ($)", min_value=0.0, value=100.0, step=10.0)

order_month = order_date.month
order_dayofweek = order_date.weekday()

# ------------------------------------------------------------------
# Prediction (same logic as Step 4)
# ------------------------------------------------------------------
st.subheader("Prediction")

if st.button("Predict Delivery Risk", type="primary"):
    input_dict = {
        "shipping_mode": encoders["shipping_mode"].transform([shipping_mode])[0],
        "order_region": encoders["order_region"].transform([order_region])[0],
        "category_name": encoders["category_name"].transform([category_name])[0],
        "customer_segment": encoders["customer_segment"].transform([customer_segment])[0],
        "quantity": quantity,
        "product_price": product_price,
        "order_month": order_month,
        "order_dayofweek": order_dayofweek,
    }
    input_df = pd.DataFrame([input_dict])[feature_cols]
    late_probability = model.predict_proba(input_df)[0][1]

    st.metric("Estimated Late Delivery Risk", f"{late_probability:.0%}")

    if late_probability >= 0.7:
        st.error(f"High risk of late delivery with **{shipping_mode}** shipping. Consider a slower-but-more-reliable mode if timing is flexible, or a faster mode if this is a hard deadline.")
    elif late_probability >= 0.4:
        st.warning("Moderate risk of late delivery. Worth flagging to the customer as a possibility.")
    else:
        st.success("Low risk of late delivery.")
