# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import csv, glob, os
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Sàng Lọc ASD - CatBoost", page_icon="🧠", layout="centered")

def find_model_file():
    candidates = ["catboost_model.cbm", "catboost_asd.cbm", "model.cbm", "catboost.cbm"]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
        p2 = Path("/mnt/data") / c
        if p2.exists():
            return str(p2)
    for d in [".", "/mnt/data"]:
        for f in glob.glob(str(Path(d) / "*.cbm")):
            return f
    return None

@st.cache_resource
def load_model():
    path = find_model_file()
    if not path:
        st.warning("Không tìm thấy file model (.cbm). Vui lòng đặt file model trong thư mục chạy app.")
        return None
    model = CatBoostClassifier()
    model.load_model(path)
    return model

def load_arff(path="Autism-Adult-Data.arff"):
    p = Path(path)
    if not p.exists():
        return None, None
    text = p.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    attrs = []
    data_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.lower().startswith('@attribute'):
            parts = s.split()
            if len(parts) >= 2:
                name = parts[1].strip()
                if name.startswith("'") and name.endswith("'"):
                    name = name[1:-1]
                attrs.append(name)
        if s.lower().startswith('@data'):
            data_start = i+1
            break
    data_lines = lines[data_start:]
    reader = csv.reader(data_lines, quotechar="'", skipinitialspace=True)
    rows = list(reader)
    if len(rows) == 0:
        return attrs, None
    if len(rows[0]) == len(attrs):
        df = pd.DataFrame(rows, columns=attrs)
        df = df.applymap(lambda x: x.strip().strip("'") if isinstance(x, str) else x)
        for c in df.columns:
            if c.lower().startswith('a') and any(ch.isdigit() for ch in c):
                df[c] = pd.to_numeric(df[c], errors='coerce')
            if c == 'age':
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return attrs, df
    return attrs, None

# load resources
model = load_model()
attrs, df_arff = load_arff("Autism-Adult-Data.arff")

# Explicit feature order to match training: 10 A-scores, age, gender, jundice, austim, used_app_before, relation
FEATURE_ORDER = ["A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
                 "age","gender","jundice","austim","used_app_before","relation"]

st.title("🔎 Sàng Lọc ASD (CatBoost)")
st.markdown("Ứng dụng này dùng model CatBoost đã huấn luyện. Vui lòng nhập đầy đủ thông tin dưới đây.")

st.header("📝 Câu hỏi sàng lọc (A1 - A10)")
q_texts = [
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
for i, qt in enumerate(q_texts):
    ans = st.selectbox(qt, options=[("Không", 0), ("Có", 1)], index=0, key=f"A{i+1}")
    a_scores.append(int(ans[1]))

st.header("🧾 Thông tin bổ sung")
age = st.number_input("Tuổi:", min_value=1, max_value=120, value=18)
gender = st.selectbox("Giới tính:", options=[("Nam","m"), ("Nữ","f")], index=0)
jundice = st.selectbox("Vàng da sau sinh:", options=[("Không","no"), ("Có","yes")], index=0)
austim = st.selectbox("Gia đình có tiền sử tự kỷ:", options=[("Không","no"), ("Có","yes")], index=0)
used_app_before = st.selectbox("Đã từng sử dụng app này trước đây:", options=[("Chưa","no"), ("Rồi","yes")], index=0)

# Relation choices: prefer ARFF values if available
if df_arff is not None and 'relation' in df_arff.columns:
    relation_choices = sorted(df_arff['relation'].fillna('?').astype(str).unique().tolist())
else:
    relation_choices = ['Self', 'Parent', '?', 'Health care professional', 'Relative', 'Others']

relation = st.selectbox("Bạn là ai đối với người được đánh giá (relation):", options=relation_choices, index=0)

def build_input_df(a_scores, age, gender_val, jundice_val, austim_val, used_app_val, relation_val):
    # Prepare a row dict with FEATURE_ORDER keys
    row = {col: 0 for col in FEATURE_ORDER}
    # map A1..A10
    for i in range(1, 11):
        col = FEATURE_ORDER[i-1]
        row[col] = int(a_scores[i-1])
    # age
    if 'age' in FEATURE_ORDER:
        row['age'] = int(age)
    # binary mapping used in training
    binary_map = {'no': 0, 'yes': 1, 'NO': 0, 'YES': 1, 'f': 0, 'm': 1}
    if 'gender' in FEATURE_ORDER:
        row['gender'] = binary_map.get(gender_val, 0)
    if 'jundice' in FEATURE_ORDER:
        row['jundice'] = binary_map.get(jundice_val, 0)
    if 'austim' in FEATURE_ORDER:
        row['austim'] = binary_map.get(austim_val, 0)
    if 'used_app_before' in FEATURE_ORDER:
        row['used_app_before'] = binary_map.get(used_app_val, 0)
    # relation: training used LabelEncoder on 'relation' then kept encoded numbers
    if 'relation' in FEATURE_ORDER:
        if df_arff is None or 'relation' not in df_arff.columns:
            relation_map = {'Self':0, 'Parent':1, '?':2, 'Health care professional':3, 'Relative':4, 'Others':5}
            row['relation'] = relation_map.get(relation_val, 2)
        else:
            le = LabelEncoder()
            try:
                # fit encoder on ARFF relation values in the same way as training
                le.fit(df_arff['relation'].fillna('?').astype(str).tolist())
                row['relation'] = int(le.transform([str(relation_val)])[0])
            except Exception:
                relation_map = {'Self':0, 'Parent':1, '?':2, 'Health care professional':3, 'Relative':4, 'Others':5}
                row['relation'] = relation_map.get(relation_val, 2)
    # Build DataFrame in correct order
    df_in = pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return df_in

if st.button("Dự đoán"):
    if model is None:
        st.error("Model chưa được tải. Vui lòng đảm bảo file .cbm có trong thư mục chạy app.")
    else:
        df_in = build_input_df(a_scores, age, gender[1], jundice[1], austim[1], used_app_before[1], relation)
        try:
            probs = model.predict_proba(df_in)
            classes = list(model.classes_)
            idx_pos = None
            for i, c in enumerate(classes):
                try:
                    if float(c) == 1.0:
                        idx_pos = i
                        break
                except:
                    pass
            if idx_pos is None:
                idx_pos = 1 if probs.shape[1] > 1 else 0
            proba = float(probs[0][idx_pos])
            pred = int(proba >= 0.5)
            st.subheader("🔍 Kết quả sàng lọc")
            st.write(f"Xác suất mắc ASD: **{proba:.2f}**")
            if pred == 1:
                st.error("""⚠️ Nguy cơ cao mắc ASD.
Hãy liên hệ chuyên gia để được tư vấn kỹ hơn.
""")
            else:
                st.success("""✅ Nguy cơ thấp mắc ASD.
Nếu cần vẫn nên theo dõi và trao đổi với chuyên gia.
""")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi dự đoán: {e}")

st.markdown('---')
st.caption('Ghi chú: Ứng dụng mô phỏng pipeline tiền xử lý theo mã huấn luyện. Đảm bảo file ARFF và file model (.cbm) nằm trong thư mục chạy app.')
