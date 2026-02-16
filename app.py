import os
import warnings

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image

import severity
import classifier
import db_solution   # 🔥 DB 연동 모듈

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Dr.Plant", layout="centered")

st.title("🌿 AI Dr.Plant")
st.write("1TEAM by 장주연, 김건률, 김동윤, 한은호")
st.write("식물 잎 사진을 업로드하면 **식물 종류, 병명, 진행률, 심각도, 솔루션**을 분석합니다.")


@st.cache_resource
def load_classification_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir

    model_path = os.path.join(project_root, "training", "best_resnet50_20251127_2211.pth")
    train_dir = os.path.join(project_root, "training", "dataset", "train")

    class_names = classifier.load_class_names(train_dir)
    classifier.class_names = class_names

    model = classifier.get_model(model_path, num_classes=len(class_names))
    return model, class_names, project_root


user_id = st.text_input(
    "사용자 ID 또는 식물 이름을 입력하세요",
    placeholder="예: user_001, kgr's plant"
)

uploaded_file = st.file_uploader("식물 잎 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])


# 이미지 미리보기
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_column_width=True)


if st.button("🔍 진단하기"):
    if not user_id:
        st.warning("먼저 사용자 ID를 입력해 주세요.")
        st.stop()
    if uploaded_file is None:
        st.warning("이미지를 업로드해 주세요.")
        st.stop()

    with st.spinner("AI가 이미지를 분석 중입니다..."):
        pil_img = Image.open(uploaded_file).convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 🔹 1) 진행률 / 심각도 계산
        disease_area, progress, level = severity.calc_severity(cv_img)

        # 🔹 2) 분류 모델 로드 & 예측
        model, class_names, project_root = load_classification_model()
        temp_image_path = os.path.join(project_root, "temp_upload_for_classification.jpg")
        pil_img.save(temp_image_path)

        class_result = classifier.predict_disease(model, temp_image_path)
        class_name = class_result["class_name"]
        confidence = class_result["confidence"]

        # 🔹 식물 타입 / healthy 여부
        plant_type = class_name.split("_")[0] if "_" in class_name else class_name
        is_healthy = "healthy" in class_name.lower()

        # ✅ healthy 클래스는 무조건 진행도 0%, 심각도 0으로 처리
        if is_healthy:
            level = 0
            progress = 0.0

        # 🔹 3) 직전 기록 조회 (현재 class_name 기준)
        prev = db_solution.fetch_previous_progress(user_id, class_name)
        if prev:
            prev_percent = prev["severity_percent"]
            prev_grade = prev["severity_grade"]
            prev_time = prev["created_at"]
        else:
            prev_percent = None
            prev_grade = None
            prev_time = None

        # 🔹 4) 완치 여부 판정
        cured = False

        # (1) 같은 클래스 기준: 이전 심각도 >= 1 이고, 이번 심각도 == 0 이면 → 완치
        if (prev_grade is not None) and (prev_grade >= 1) and (level == 0):
            cured = True

        # (2) 식물 단위 기준:
        #     이번 클래스가 healthy 이고, 과거에 같은 식물의 '질병' 기록이 1번이라도 있으면 → 완치
        if is_healthy and db_solution.has_past_plant_disease(user_id, plant_type):
            cured = True

        st.write("🔍 DEBUG QUERY VALUES →", class_name, int(round(progress)), level)

        # 🔹 5) 솔루션 조회 (심각도 0이면 '처방 필요 없음' 문구 반환)
        solution = db_solution.fetch_solution(class_name, progress, level)

        # 🔹 6) 진단 결과 저장
        db_solution.save_diagnosis(
            user_id=user_id,
            disease_class=class_name,
            severity_percent=progress,
            severity_grade=level,
            solution_ko=solution
        )

        # 🔹 7) 히스토리 조회 (👉 식물 단위)
        history_rows = db_solution.fetch_history_series(user_id, plant_type)

    st.success("진단이 완료되었습니다. ✅")

    # =========================
    # 1. 분류 결과
    # =========================
    st.subheader("1. 식물 및 질병 분류 결과")
    st.write(f"**식물 종류:** {plant_type}")
    st.write(f"**질병 클래스:** {class_name}")
    st.write(f"**분류 신뢰도:** {confidence:.2f}%")

    # =========================
    # 2. 진행도 및 심각도 분석
    # =========================
    st.subheader("2. 진행도 및 심각도 분석")

    # 🔹 완치 축하 메시지
    if cured:
        st.success(f"🎉 축하합니다! {user_id}님의 식물이 완치되었습니다!")

    # 🔹 직전 진행도 출력 로직
    if prev_percent is not None:
        # 진짜로 직전 기록이 있을 때만 이 문구 사용
        st.write(f"**직전 진행도:** {prev_percent}% (진단 시각: {prev_time})")
    else:
        # prev 기록은 없지만, 이번에 완치로 판단된 경우 → '직전 없음' 문구 숨기기
        if not cured:
            st.write("이전에 저장된 진단 기록이 없습니다. (📌 이번이 첫 진단입니다.)")

    st.write(f"**현재 진행도:** {progress:.2f}%")
    st.write(f"**심각도(Level 0~5):** {level}")

    if prev_percent is not None:
        diff = progress - prev_percent
        if diff > 0:
            st.write(f"📉 **직전보다 {diff:.2f}% 악화되었습니다.**")
        elif diff < 0:
            st.write(f"📈 **직전보다 {abs(diff):.2f}% 호전되었습니다.**")
        else:
            st.write("➖ **직전과 진행도가 동일합니다.**")

    # =========================
    # 3. 추천 솔루션
    # =========================
    st.subheader("3. 추천 솔루션")
    st.write(solution)

    # =========================
    # 4. 병 진행도 추이 그래프
    # =========================
    st.subheader("4. 병 진행도 추이 그래프")
    if history_rows:
        severities = [row[0] for row in history_rows]  # 진행도 (%)
        times = [row[1] for row in history_rows]       # created_at (DATETIME)

        fig, ax = plt.subplots()
        # ✅ 초록색 선 그래프
        ax.plot(times, severities, marker="o", color="green")

        ax.set_title(f"disease_rate_graph ({user_id}, {plant_type})")
        ax.set_xlabel("diagnose_time")
        ax.set_ylabel("disease rate (%)")

        # 🔹 x축 포맷을 '년-월-일 시:분:초'로 표시
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

        # 🔹 라벨 겹치지 않게 회전
        fig.autofmt_xdate()
        st.pyplot(fig)
    else:
        st.write("아직 저장된 히스토리가 없어 그래프를 그릴 수 없습니다.")
