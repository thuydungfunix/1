# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

# --- Thử import shap ---
try:
    import shap
    shap_installed = True
except ImportError:
    shap_installed = False

# ⚙️ Cấu hình trang
st.set_page_config(page_title="Ứng dụng Sàng lọc Tự kỷ", page_icon="🧠", layout="centered")

# 📂 Hàm tải mô hình
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")  # thay bằng file model đã train
    return model

model = load_model()

# --- Giao diện ---
st.title("🧠 Ứng dụng Sàng lọc Tự kỷ (CatBoost)")
st.subheader("Vui lòng trả lời 10 câu hỏi AQ-10 và thông tin cơ bản")

# 📋 Bộ 10 câu hỏi AQ-10 (tiếng Việt)
questions_vi = [
    "1. Người được đánh giá có thường tránh giao tiếp bằng mắt không?",
    "2. Người đó có thích chơi một mình hơn là cùng người khác không?",
    "3. Người đó có hay lặp lại từ/ngôn ngữ không?",
    "4. Người đó có khó khăn khi hiểu cảm xúc của người khác không?",
    "5. Người đó có khi nào không phản hồi khi được gọi tên không?",
    "6. Người đó có nhạy cảm quá mức với âm thanh không?",
    "7. Người đó có ít chia sẻ hứng thú/thành tích với người khác không?",
    "8. Người đó có hành vi lặp đi lặp lại không?",
    "9. Người đó có gặp khó khăn khi thay đổi thói quen hoặc môi trường không?",
    "10. Người đó có khó hiểu các quy tắc xã hội cơ bản không?"
]

aq_answers = []
for i, q in enumerate(questions_vi, 1):
    ans = st.radio(q, ["Không", "Có"], key=f"q{i}")
    aq_answers.append(1 if ans == "Có" else 0)

# 🧑‍💻 Thông tin khác (hiển thị tiếng Việt nhưng mapping sang English/number cho model)
age = st.number_input("Tuổi", min_value=1, max_value=100, value=18)

gender_vi = st.selectbox("Giới tính", ["Nam", "Nữ"])
gender = "male" if gender_vi == "Nam" else "female"

jundice_vi = st.radio("Có bị vàng da lúc sinh không?", ["Không", "Có"])
jundice = 1 if jundice_vi == "Có" else 0

autism_vi = st.radio("Gia đình có người tự kỷ không?", ["Không", "Có"])
autism = 1 if autism_vi == "Có" else 0

relation_vi = st.selectbox("Người trả lời bảng khảo sát", 
                           ["Bản thân", "Cha/mẹ", "Người thân", "Chuyên gia y tế", "Khác"])
relation_map = {
    "Bản thân": "Self",
    "Cha/mẹ": "Parent",
    "Người thân": "Relative",
    "Chuyên gia y tế": "Health care professional",
    "Khác": "Others"
}
relation = relation_map[relation_vi]

used_app_vi = st.radio("Đã từng dùng ứng dụng này trước đây chưa?", ["Không", "Có"])
used_app_before = 1 if used_app_vi == "Có" else 0

# 👉 Tạo DataFrame
input_data = pd.DataFrame([aq_answers + [
    age,
    gender,
    jundice,
    autism,
    used_app_before,
    relation
]], columns=[f"A{i}" for i in range(1, 10)] + 
         ["age", "gender", "jundice", "autism", "contry_of_res", "used_app_before", "relation"])

# Hiển thị lại dữ liệu đầu vào
st.subheader("📋 Dữ liệu đầu vào")
st.write(input_data)

# --- Dự đoán ---
if st.button("🔍 Dự đoán"):
    cat_features = ["gender",  "relation"]

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"⚠️ Nguy cơ **cao** mắc ASD (xác suất: {proba:.2f})")
        st.write("👉 Khuyến nghị: Tham khảo ý kiến bác sĩ chuyên khoa để được đánh giá chi tiết.")
    else:
        st.success(f"✅ Nguy cơ **thấp** mắc ASD (xác suất: {proba:.2f})")
        st.write("👉 Khuyến nghị: Tiếp tục theo dõi và hỗ trợ phát triển hành vi xã hội.")

    # 📊 Giải thích kết quả
    st.subheader("📊 Yếu tố ảnh hưởng đến kết quả")
    if shap_installed:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
    else:
        st.warning("⚠️ SHAP chưa cài đặt. Hiển thị Feature Importance thay thế.")
        feat_df = pd.DataFrame({
            "Đặc trưng": input_data.columns,
            "Tầm quan trọng": model.get_feature_importance()
        }).sort_values(by="Tầm quan trọng", ascending=False)
        st.bar_chart(feat_df.set_index("Đặc trưng"))


