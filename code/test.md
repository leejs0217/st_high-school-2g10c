import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="KBO 구단 투수력 분석",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ KBO 구단 투수력 분석")
st.header("팀 평균자책점(ERA)과 승수(W)의 상관관계 분석")

st.write(
    "2024년부터 2026년까지 KBO 10개 구단의 정규시즌 기록을 활용하여 "
    "팀 평균자책점과 승수 사이의 관계를 분석합니다."
)

st.write(
    "산점도, 상관계수, 선형 회귀분석을 통해 "
    "팀 평균자책점으로 구단의 승수를 예측할 수 있는지 알아봅니다."
)


# ==========================================
# 2. KBO 데이터 직접 입력
# ==========================================

# 열 순서: 승수, 무승부, 패수, 팀ERA, 팀명, 연도

records = [

    # 2024년
    [87, 2, 55, 4.40, "KIA", 2024],
    [78, 2, 64, 4.68, "삼성", 2024],
    [76, 2, 66, 4.63, "LG", 2024],
    [74, 2, 68, 4.82, "두산", 2024],
    [72, 2, 70, 5.11, "KT", 2024],
    [72, 2, 70, 5.25, "SSG", 2024],
    [66, 4, 74, 5.05, "롯데", 2024],
    [66, 2, 76, 4.98, "한화", 2024],
    [61, 2, 81, 5.00, "NC", 2024],
    [58, 0, 86, 5.16, "키움", 2024],

    # 2025년
    [85, 3, 56, 3.79, "LG", 2025],
    [83, 4, 57, 3.55, "한화", 2025],
    [75, 4, 65, 3.63, "SSG", 2025],
    [74, 2, 68, 4.12, "삼성", 2025],
    [71, 6, 67, 4.82, "NC", 2025],
    [71, 5, 68, 4.09, "KT", 2025],
    [66, 6, 72, 4.75, "롯데", 2025],
    [65, 4, 75, 4.66, "KIA", 2025],
    [61, 6, 77, 4.30, "두산", 2025],
    [47, 4, 93, 5.39, "키움", 2025],

    # 2026년
    [83, 3, 58, 4.31, "삼성", 2026],
    [90, 5, 49, 4.14, "KT", 2026],
    [79, 2, 63, 4.27, "KIA", 2026],
    [78, 1, 65, 4.81, "LG", 2026],
    [73, 5, 66, 3.86, "두산", 2026],
    [65, 5, 74, 5.16, "SSG", 2026],
    [64, 3, 77, 4.72, "롯데", 2026],
    [64, 2, 78, 4.79, "NC", 2026],
    [57, 4, 83, 5.26, "한화", 2026],
    [50, 4, 90, 5.30, "키움", 2026]

]

columns = [
    "승수",
    "무승부",
    "패수",
    "팀ERA",
    "팀명",
    "연도"
]

df = pd.DataFrame(records, columns=columns)

# 데이터 형식 확인
df["승수"] = pd.to_numeric(df["승수"], errors="coerce")
df["무승부"] = pd.to_numeric(df["무승부"], errors="coerce")
df["패수"] = pd.to_numeric(df["패수"], errors="coerce")
df["팀ERA"] = pd.to_numeric(df["팀ERA"], errors="coerce")

df = df.dropna(
    subset=["승수", "무승부", "패수", "팀ERA", "팀명", "연도"]
)

df["연도"] = df["연도"].astype(int)

# 데이터 개수 확인
if len(df) != 30:
    st.error("데이터 행 수가 올바르지 않습니다.")
    st.stop()

if df.groupby("연도").size().to_dict() != {
    2024: 10,
    2025: 10,
    2026: 10
}:
    st.error("연도별 구단 데이터가 올바르지 않습니다.")
    st.stop()


# ==========================================
# 3. 데이터 확인
# ==========================================

st.divider()
st.header("1. 데이터 불러오기")

st.success("KBO 데이터 30행 × 6열을 불러왔습니다.")

st.dataframe(
    df.sort_values(["연도", "팀명"]),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "자료: KBO 공식 기록실 / 2024~2026년 정규시즌 구단 기록"
)


# ==========================================
# 4. 시즌 선택
# ==========================================

st.divider()
st.header("2. 분석할 시즌 선택")

selected_year = st.selectbox(
    "분석 기간을 선택하세요.",
    ["전체 시즌", 2024, 2025, 2026]
)

if selected_year == "전체 시즌":
    data = df.copy()
else:
    data = df[df["연도"] == selected_year].copy()

st.write(f"분석 대상: **{len(data)}개 구단 기록**")


# ==========================================
# 5. 기초 통계
# ==========================================

st.divider()
st.header("3. 기초 통계 분석")

col1, col2, col3 = st.columns(3)

col1.metric(
    "평균 승수",
    f"{data['승수'].mean():.1f}승"
)

col2.metric(
    "평균 팀 ERA",
    f"{data['팀ERA'].mean():.2f}"
)

col3.metric(
    "분석 데이터",
    f"{len(data)}개"
)


# ==========================================
# 6. 상관관계 분석
# ==========================================

st.divider()
st.header("4. 상관관계 분석")

correlation = data["팀ERA"].corr(data["승수"])

st.metric(
    "ERA와 승수의 상관계수",
    f"{correlation:.3f}"
)

if abs(correlation) < 0.2:
    strength = "거의 상관관계가 없습니다."
elif abs(correlation) < 0.4:
    strength = "약한 상관관계가 있습니다."
elif abs(correlation) < 0.6:
    strength = "중간 정도의 상관관계가 있습니다."
elif abs(correlation) < 0.8:
    strength = "비교적 강한 상관관계가 있습니다."
else:
    strength = "강한 상관관계가 있습니다."

if correlation < 0:
    direction = "음의 상관관계"
elif correlation > 0:
    direction = "양의 상관관계"
else:
    direction = "상관관계가 거의 없는 상태"

st.info(
    f"분석 결과 {direction}가 나타났으며, "
    f"{strength}"
)

st.write(
    "상관계수가 음수라면 ERA가 낮을수록 승수가 많아지는 "
    "경향을 의미합니다."
)


# ==========================================
# 7. 산점도 및 회귀선
# ==========================================

st.divider()
st.header("5. 산점도 분석")

st.write(
    "X축은 팀 평균자책점(ERA), "
    "Y축은 구단의 승수(W)입니다."
)

fig = px.scatter(
    data,
    x="팀ERA",
    y="승수",
    color="연도",
    hover_name="팀명",
    hover_data=["무승부", "패수"],
    labels={
        "팀ERA": "팀 평균자책점(ERA)",
        "승수": "승수(W)",
        "연도": "시즌"
    },
    title="KBO 구단의 ERA와 승수 관계"
)

# 선형 회귀선
if len(data) >= 2 and data["팀ERA"].nunique() >= 2:

    X_all = data[["팀ERA"]]
    y_all = data["승수"]

    visual_model = LinearRegression()
    visual_model.fit(X_all, y_all)

    line_x = np.linspace(
        data["팀ERA"].min(),
        data["팀ERA"].max(),
        100
    )

    line_y = visual_model.predict(
        pd.DataFrame({"팀ERA": line_x})
    )

    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="선형 회귀선",
            line=dict(width=3, dash="dash")
        )
    )

fig.update_layout(
    template="plotly_white",
    height=550
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# 8. 회귀식
# ==========================================

st.divider()
st.header("6. 선형 회귀식")

if len(data) >= 2 and data["팀ERA"].nunique() >= 2:

    slope = visual_model.coef_[0]
    intercept = visual_model.intercept_

    st.subheader("회귀 분석 결과")

    st.write(
        f"**승수 = {slope:.3f} × ERA + ({intercept:.3f})**"
    )

    st.write(
        f"회귀선의 기울기: {slope:.3f}"
    )

    if slope < 0:
        st.write(
            "ERA가 증가할수록 예측 승수가 감소하는 방향입니다."
        )
    elif slope > 0:
        st.write(
            "ERA가 증가할수록 예측 승수가 증가하는 방향입니다."
        )
    else:
        st.write(
            "ERA와 승수 사이의 선형적인 변화가 거의 없습니다."
        )


# ==========================================
# 9. 구단별 비교 그래프
# ==========================================

st.divider()
st.header("7. 구단별 기록 비교")

fig_era = px.bar(
    data.sort_values("팀ERA"),
    x="팀명",
    y="팀ERA",
    color="연도",
    barmode="group",
    text_auto=".2f",
    title="구단별 팀 평균자책점 비교",
    labels={
        "팀명": "구단",
        "팀ERA": "팀 평균자책점",
        "연도": "시즌"
    }
)

fig_era.update_layout(
    template="plotly_white",
    height=430
)

st.plotly_chart(
    fig_era,
    use_container_width=True
)

fig_win = px.bar(
    data.sort_values("승수", ascending=False),
    x="팀명",
    y="승수",
    color="연도",
    barmode="group",
    text_auto=True,
    title="구단별 승수 비교",
    labels={
        "팀명": "구단",
        "승수": "승수",
        "연도": "시즌"
    }
)

fig_win.update_layout(
    template="plotly_white",
    height=430
)

st.plotly_chart(
    fig_win,
    use_container_width=True
)


# ==========================================
# 10. 모델 평가
# ==========================================

st.divider()
st.header("8. 선형 회귀모델 성능 평가")

st.write(
    "전체 30개 데이터를 학습용 80%, "
    "테스트용 20%로 분리하여 평가합니다."
)

if selected_year == "전체 시즌":

    X = data[["팀ERA"]]
    y = data["승수"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # 기준선 모델
    baseline = np.full(
        len(y_test),
        y_train.mean()
    )

    # 회귀모델 평가
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # 기준선 평가
    baseline_mae = mean_absolute_error(
        y_test,
        baseline
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline
        )
    )

    result_df = pd.DataFrame({
        "평가 지표": ["MAE", "RMSE"],
        "기준선 모델": [
            baseline_mae,
            baseline_rmse
        ],
        "선형 회귀모델": [
            mae,
            rmse
        ]
    })

    st.subheader("기준선 모델과 회귀모델 비교")

    st.dataframe(
        result_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:.2f}승")
    c2.metric("RMSE", f"{rmse:.2f}승")
    c3.metric("R²", f"{r2:.3f}")

    st.subheader("테스트 데이터 예측 결과")

    prediction_df = pd.DataFrame({
        "실제 승수": y_test.to_numpy(),
        "예측 승수": np.round(predictions, 2),
        "예측 오차": np.round(
            y_test.to_numpy() - predictions,
            2
        )
    })

    st.dataframe(
        prediction_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "시즌별 데이터는 10개로 표본이 적기 때문에 "
        "모델 성능 평가는 전체 시즌을 선택했을 때만 진행합니다."
    )


# ==========================================
# 11. 분석의 한계
# ==========================================

st.divider()
st.header("9. 분석의 한계")

st.markdown("""
1. 분석에 사용한 데이터는 총 30개로 표본 수가 많지 않습니다.

2. 승수는 팀 평균자책점뿐만 아니라 타선의 득점 지원,
   수비력, 불펜 운영 등 다양한 요인의 영향을 받습니다.

3. 상관관계가 나타난다고 해서 평균자책점이 승수 변화의
   직접적인 원인이라고 단정할 수는 없습니다.

4. 학습 데이터와 테스트 데이터의 분할에 따라
   모델의 평가 결과가 달라질 수 있습니다.
""")

st.divider()

st.caption(
    "자료 출처: KBO 공식 기록실 | "
    "분석 대상: 2024~2026년 정규시즌"
)

st.caption(
    "본 분석은 교육용 데이터 분석 프로젝트입니다."
)
