
# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import csv
import glob
import os
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Sàng Lọc Tự Kỷ - Fixed", page_icon="🧠", layout="centered")

# ------------------------- Utilities -------------------------
def find_model_file():
    # try common names then glob for .cbm in cwd and /mnt/data
    candidates = ["catboost_asd.cbm", "catboost_model.cbm", "model.cbm", "catboost.cbm"]
    for c in candidates:
        if Path(c).exists():
            return str(Path(c))
        if Path("/mnt/data") / c:
            p = Path("/mnt/data") / c
            if p.exists():
                return str(p)
    # glob search
    for d in [".", "/mnt/data"]:
        for f in glob.glob(os.path.join(d, "*.cbm")):
            return f
    return None

@st.cache_resource
def load_model():
    model_path = find_model_file()
    if model_path is None:
        st.warning("Không tìm thấy file model (.cbm). Vui lòng đặt file model trong thư mục chạy app (ví dụ: catboost_model.cbm).")
        return None
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

def load_arff(path="Autism-Adult-Data.arff"):
    """
    Minimal ARFF parser to get attribute names and the dataset as a pandas DataFrame.
    Expects that the ARFF file structure is similar to the provided file.
    """
    p = Path(path)
    if not p.exists():
        return None, None
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    attrs = []
    data_start = None
    for i,l in enumerate(lines):
        ls = l.strip()
        if ls.lower().startswith('@attribute'):
            # get attribute name (second token)
            try:
                parts = ls.split()
                name = parts[1]
            except:
                continue
            attrs.append(name.strip())
        if ls.lower().startswith('@data'):
            data_start = i+1
            break
    # read data lines with csv.reader to handle quoted values
    data_lines = lines[data_start:]
    reader = csv.reader(data_lines, quotechar="'", skipinitialspace=True)
    rows = list(reader)
    # if number of columns matches attributes, create dataframe
    if len(rows)>0 and len(rows[0]) == len(attrs):
        df = pd.DataFrame(rows, columns=attrs)
        # strip quotes/spaces
        df = df.applymap(lambda x: x.strip().strip("'") if isinstance(x,str) else x)
        # convert numeric where appropriate
        for col in df.columns:
            if col.startswith('A') or col in ['age','result']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return attrs, df
    else:
        return attrs, None

# ------------------------- Prepare model + encoders -------------------------
model = load_model()

attrs, df_dataset = load_arff("Autism-Adult-Data.arff")
if attrs is None:
    attrs = []

# We assume the original model was trained on the first 16 attributes in the ARFF file (A1..contry_of_res)
# This matches the uploaded dataset structure used when fixing the app.
feature_attrs = attrs[:16] if len(attrs) >= 16 else None

# build label encoders for categorical columns used by the model (based on ARFF)
cat_cols = []
encoders = {}
if df_dataset is not None and feature_attrs is not None:
    # identify categorical columns among the first 16 (non-numeric)
    for c in feature_attrs:
        if df_dataset[c].dtype == object:
            cat_cols.append(c)
    # fit encoders (LabelEncoder behaves like typical training pipelines that encode categories to ints)
    for c in cat_cols:
        le = LabelEncoder()
        vals = df_dataset[c].fillna("NA").astype(str).tolist()
        le.fit(vals)
        encoders[c] = le

# ------------------------- UI -------------------------
st.title("🔎 Ứng dụng Sàng Lọc Tự Kỷ (ASD) — Fixed")
st.markdown("Ứng dụng này dùng model CatBoost (file `.cbm`) và ánh xạ các đặc trưng dựa trên bộ dữ liệu ARFF đã tải lên.")

st.header("📝 Vui lòng trả lời các câu hỏi sau:")

# 10 câu hỏi A1 - A10 (0/1)
questions = [
    "1. Có khi nào người được đánh giá tránh giao tiếp bằng mắt?",
    "2. Người đó có thích chơi một mình?",
    "3. Người đó có hay lặp lại từ/ngôn ngữ không?",
    "4. Người đó có khó khăn khi hiểu cảm xúc người khác?",
    "5. Có khi nào người đó không phản hồi khi được gọi tên?",
    "6. Người đó có nhạy cảm với âm thanh không?",
    "7. Có khi nào người đó không chia sẻ hứng thú hoặc thành tích với người khác?",
    "8. Người đó có hành vi lặp đi lặp lại không?",
    "9. Người đó có gặp khó khăn khi thay đổi thói quen hoặc môi trường không?",
    "10. Có khi nào người đó không hiểu các quy tắc xã hội cơ bản không?"
]
a_scores = []
for i,q in enumerate(questions):
    ans = st.selectbox(q, ["Không", "Có"], key=f"A_{i+1}")
    a_scores.append(1 if ans == "Có" else 0)

# Các đặc trưng khác (những thuộc tính đầu tiên trong dataset)
age = st.number_input("11. Tuổi của người được đánh giá:", min_value=1, max_value=120, value=18)
gender_choice = st.selectbox("12. Giới tính:", ["Nam", "Nữ"])

# If dataset was loaded, provide selection lists for ethnicity and country using values from the ARFF.
if df_dataset is not None:
    ethnicity_options = sorted(df_dataset['ethnicity'].fillna("Unknown").unique().tolist())
    country_options = sorted(df_dataset['contry_of_res'].fillna("Unknown").unique().tolist())
else:
    # Fallback defaults (these should be replaced by your dataset if available)
    ethnicity_options = ["White-European", "Latino", "Others", "Black", "Asian", "Hispanic", "South Asian"]
    country_options = ["United States", "India", "Brazil", "Spain", "Egypt", "Sri Lanka", "New Zealand"]

ethnicity = st.selectbox("13. Ethnicity (dân tộc):", ethnicity_options)
jundice = st.selectbox("14. Người đó có bị vàng da sau sinh không?", ["Không", "Có"])
austim = st.selectbox("15. Gia đình có người từng bị tự kỷ không?", ["Không", "Có"])
# These two fields exist in dataset but may not be part of the model's features — keep for UI only
used_app_before = st.selectbox("16. Bạn đã từng sử dụng ứng dụng này chưa?", ["Chưa", "Rồi"])
relation = st.selectbox("17. Bạn là ai đối với người được đánh giá?", ["Bố", "Mẹ", "Bản thân", "Người thân khác"])

# ------------------------- Build features for prediction -------------------------
def build_input_dataframe(a_scores, age, gender_choice, ethnicity, jundice, austim, country, feature_attrs, encoders):
    if feature_attrs is None:
        st.error("Không có thông tin cấu trúc đặc trưng (feature attributes). Vui lòng kiểm tra file ARFF.")
        return None
    row = {}
    # A1..A10
    for i in range(10):
        row[feature_attrs[i]] = int(a_scores[i])
    # age
    if 'age' in feature_attrs:
        row['age'] = int(age)
    # gender: map VN -> dataset values ('m'/'f')
    gender_map = {'Nam': 'm', 'Nữ': 'f'}
    if 'gender' in feature_attrs:
        val = gender_map.get(gender_choice, gender_choice)
        if 'gender' in encoders:
            row['gender'] = int(encoders['gender'].transform([str(val)])[0])
        else:
            # fallback: try numeric
            row['gender'] = 1 if val == 'm' else 0
    # ethnicity
    if 'ethnicity' in feature_attrs:
        if 'ethnicity' in encoders:
            row['ethnicity'] = int(encoders['ethnicity'].transform([str(ethnicity)])[0])
        else:
            row['ethnicity'] = str(ethnicity)
    # jundice
    if 'jundice' in feature_attrs:
        val = 'yes' if jundice == 'Có' else 'no'
        if 'jundice' in encoders:
            row['jundice'] = int(encoders['jundice'].transform([str(val)])[0])
        else:
            row['jundice'] = val
    # austim (family history)
    if 'austim' in feature_attrs:
        val = 'yes' if austim == 'Có' else 'no'
        if 'austim' in encoders:
            row['austim'] = int(encoders['austim'].transform([str(val)])[0])
        else:
            row['austim'] = val
    # country (contry_of_res)
    if 'contry_of_res' in feature_attrs:
        if 'contry_of_res' in encoders:
            row['contry_of_res'] = int(encoders['contry_of_res'].transform([str(country)])[0])
        else:
            row['contry_of_res'] = str(country)
    # If some attributes from feature_attrs were not set above, fill with zeros
    for c in feature_attrs:
        if c not in row:
            # default numeric 0
            row[c] = 0
    # build dataframe ordered
    df_in = pd.DataFrame([[row[c] for c in feature_attrs]], columns=feature_attrs)
    return df_in

# Prediction button
if st.button("📊 Dự đoán khả năng tự kỷ"):
    if model is None:
        st.error("Model chưa được tải. Hãy đảm bảo file .cbm có trong thư mục chạy app.")
    else:
        df_in = build_input_dataframe(a_scores, age, gender_choice, ethnicity, jundice, austim, country_options[0] if len(country_options)>0 else 'Unknown', feature_attrs, encoders)
        # Note: above we accidentally used country_options[0] as default; override with selected country
        # Rebuild with real selected country
        df_in = build_input_dataframe(a_scores, age, gender_choice, ethnicity, jundice, austim, country, feature_attrs, encoders)
        if df_in is None:
            st.error("Không thể khởi tạo dữ liệu đầu vào.")
        else:
            try:
                probs = model.predict_proba(df_in)
                # get index of class '1' (positive label) if present in model.classes_
                classes = list(model.classes_)
                try:
                    idx_pos = classes.index(1.0)
                except ValueError:
                    # fallback: take second column
                    idx_pos = 1 if probs.shape[1] > 1 else 0
                proba = float(probs[0][idx_pos])
                pred = int(probs[0][idx_pos] >= 0.5)
                st.subheader("🔍 Kết quả sàng lọc:")
                st.write(f"👉 Xác suất mắc tự kỷ (ASD): **{proba:.2f}**")
                st.markdown("### 🧭 Gợi ý hành động tiếp theo:")
                if pred == 1:
                   st.error("""
⚠️ Nguy cơ cao mắc ASD.
🔹 Hãy liên hệ chuyên gia tâm lý hoặc cơ sở y tế để được tư vấn kỹ lưỡng hơn.
🔹 Ghi chép lại các biểu hiện thường gặp trong cuộc sống hàng ngày.
🔹 Có thể tham khảo tài liệu về ASD từ WHO, CDC hoặc các trung tâm hỗ trợ trong nước.
""")

                else:
                  st.success("""
✅ Nguy cơ thấp mắc ASD.
🔹 Bạn có thể yên tâm ở thời điểm hiện tại.
🔹 Nếu vẫn còn băn khoăn, hãy trao đổi thêm với chuyên gia.
""")

            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi dự đoán: {e}")

st.markdown(\"---\")
st.caption(\"Ghi chú: Ứng dụng này sử dụng file model CatBoost (.cbm) và ánh xạ các đặc trưng bằng LabelEncoder dựa trên bộ dữ liệu ARFF cung cấp. Nếu bạn muốn thay đổi pipeline tiền xử lý (ví dụ: giữ relation, used_app_before), vui lòng gửi file code training để chúng tôi tái tạo chính xác quy trình tiền xử lý.\") 
