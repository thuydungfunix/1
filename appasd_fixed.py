# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

# --- Thử import shap ---
try:
    import shap
    shap_installed = True
except ImportError:
    shap_installed = False

# ⚙️ Cấu hình trang
st.set_page_config(page_title="ASD Screening", page_icon="🧠", layout="centered")

# 📂 Hàm tải mô hình
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_asd_model.cbm")  # thay bằng file model đã train
    return model

model = load_model()

# --- Giao diện nhập liệu ---
st.title("🧠 Autism Screening App (CatBoost)")
st.subheader("Nhập thông tin để sàng lọc")

# Bộ 10 câu hỏi AQ-10
aq_questions = []
for i in range(1, 11):
    ans = st.radio(f"Câu hỏi {i}", ["No", "Yes"], key=f"q{i}")
    aq_questions.append(1 if ans == "Yes" else 0)

# Thông tin khác
age = st.number_input("Tuổi", min_value=1, max_value=100, value=18)
gender = st.selectbox("Giới tính", ["male", "female"])
jaundice = st.radio("Có bị vàng da lúc sinh?", ["No", "Yes"])
autism = st.radio("Gia đình có người tự kỷ?", ["No", "Yes"])
relation = st.selectbox("Người trả lời bảng khảo sát", 
                        ["Self", "Parent", "Relative", "Health care professional", "Others"])
used_app_before = st.radio("Đã dùng app trước đây?", ["No", "Yes"])

# Convert đầu vào thành DataFrame
input_data = pd.DataFrame([aq_questions + [
    age,
    1 if autism == "Yes" else 0,
    1 if jaundice == "Yes" else 0,
    1 if used_app_before == "Yes" else 0,
    gender,
    relation
]], columns=[f"A{i}" for i in range(1, 10+1)] + 
         ["age", "autism", "jaundice", "used_app_before", "gender", "relation"])

# --- Dự đoán ---
if st.button("🔍 Dự đoán"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    # Hiển thị kết quả
    if pred == 1:
        st.error(f"⚠️ Nguy cơ **cao** mắc ASD (xác suất: {proba:.2f})")
        st.write("👉 Khuyến nghị: Bạn nên tham khảo ý kiến bác sĩ chuyên khoa để được đánh giá chi tiết hơn.")
    else:
        st.success(f"✅ Nguy cơ **thấp** mắc ASD (xác suất: {proba:.2f})")
        st.write("👉 Khuyến nghị: Tiếp tục theo dõi và hỗ trợ phát triển hành vi xã hội cho cá nhân.")

    # 🔎 Giải thích kết quả
    st.subheader("📊 Yếu tố ảnh hưởng đến kết quả")

    if shap_installed:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
    else:
        st.warning("⚠️ SHAP chưa được cài đặt. Hiển thị Feature Importance thay thế.")
        feature_importances = model.get_feature_importance()
        feat_df = pd.DataFrame({
            "Feature": input_data.columns,
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(feat_df.set_index("Feature"))
