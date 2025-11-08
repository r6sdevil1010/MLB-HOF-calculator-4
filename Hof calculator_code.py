import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pybaseball import playerid_lookup, batting_stats

# -----------------------------
# ⚾ 기본 데이터/모델 학습 파트
# -----------------------------
data = pd.DataFrame({
    "WAR": [50, 65, 75, 80, 100],
    "HOFm": [90, 110, 130, 140, 180],
    "JAWS": [45, 55, 60, 70, 85],
    "Elected": [0, 0, 1, 1, 1]
})

model_vote = LogisticRegression()
model_prob = LogisticRegression()
X = data[["WAR", "HOFm", "JAWS"]]
y = data["Elected"]
model_vote.fit(X, y)
model_prob.fit(X, y)

# -----------------------------
# ⚙️ 자동 선수 기록 추출 함수
# -----------------------------
@st.cache_data(show_spinner=False)
def get_player_stats(name):
    try:
        first, last = name.strip().split(" ", 1)
    except ValueError:
        return None
    lookup = playerid_lookup(last, first)
    if lookup.empty:
        return None
    try:
        debut = int(lookup.iloc[0]['mlb_played_first'])
        final = int(lookup.iloc[0]['mlb_played_last'])
    except:
        debut, final = 2000, 2020  # fallback

    # ⚾ 전체 시즌 기록 수집
    stats_all = batting_stats(debut, final)
    # pybaseball의 특정 버전에 따라 컬럼명(대소문자)이 다를 수 있습니다
    stats = stats_all[stats_all['Name'].str.strip() == name.strip()]
    if stats.empty:
        return None
    war = stats['WAR'].sum() if "WAR" in stats else None
    return {
        "WAR": war,
        "HOFm": None,
        "JAWS": None
    }

# -----------------------------
# ⚙️ 유틸 함수
# -----------------------------
def simulate_vote_growth(start_vote):
    votes = [start_vote]
    for i in range(1, 10):
        inc = 0.05 + 0.08 * (1 - votes[-1] / 100)
        votes.append(min(100, votes[-1] * (1 + inc)))
    return votes

def predict_HOF(name, WAR, HOFm, JAWS, doping=False, leadership=0.5, influence=0.5, era_adjust=0.0):
    basic_vote = model_vote.predict_proba([[WAR, HOFm, JAWS]])[0, 1] * 100
    basic_prob = model_prob.predict_proba([[WAR, HOFm, JAWS]])[0, 1]
    ext_factor = (-0.35 if doping else 0) + leadership * 0.15 + influence * 0.2 + era_adjust * 0.1
    final_vote = max(0, min(100, basic_vote * (1 + ext_factor)))
    final_prob = max(0, min(1, basic_prob * (1 + ext_factor)))
    vote_trend = simulate_vote_growth(final_vote)
    return {
        "name": name,
        "basic_vote": basic_vote,
        "final_vote": final_vote,
        "basic_prob": basic_prob,
        "final_prob": final_prob,
        "vote_trend": vote_trend
    }

def summarize_result(res):
    text = f"⚾ {res['name']} — Hall of Fame 예측 결과\n\n"
    text += f"📊 기본모델 득표율: {res['basic_vote']:.1f}%\n"
    text += f"🏅 외부요인 반영 득표율: {res['final_vote']:.1f}%\n"
    text += f"🎯 헌액 확률(성적기반): {res['basic_prob'] * 100:.1f}%\n"
    text += f"💬 최종 헌액 확률(외부요인 반영): {res['final_prob'] * 100:.1f}%\n\n"
    text += f"📈 연차별 득표율 추정: {[round(v, 1) for v in res['vote_trend']]}"
    return text

# -----------------------------
# 🌐 Streamlit UI
# -----------------------------
st.title("⚾ MLB Hall of Fame 예측 시스템")
st.caption("WAR, HOFm, 리더십, 도핑 여부 등을 고려한 명전 확률 추정기")

name = st.text_input("선수 이름", "Joe Mauer")

# 기본값 세팅용 state
if "WAR" not in st.session_state:
    st.session_state["WAR"] = 65.0
    st.session_state["HOFm"] = 120.0
    st.session_state["JAWS"] = 55.0

if st.button("♻️ 기록 자동 채우기"):
    player_stats = get_player_stats(name)
    if player_stats:
        st.session_state["WAR"] = player_stats["WAR"] if player_stats["WAR"] is not None else 65.0
        st.session_state["HOFm"] = player_stats["HOFm"] if player_stats["HOFm"] is not None else 120.0
        st.session_state["JAWS"] = player_stats["JAWS"] if player_stats["JAWS"] is not None else 55.0
        st.success("자동 입력 완료!")
    else:
        st.warning("선수 기록을 찾을 수 없습니다.")

WAR = st.number_input("WAR", 0.0, 150.0, float(st.session_state["WAR"]), key="WAR_box")
HOFm = st.number_input("HOF Monitor 점수", 0.0, 300.0, float(st.session_state["HOFm"]), key="HOFm_box")
JAWS = st.number_input("JAWS 점수", 0.0, 100.0, float(st.session_state["JAWS"]), key="JAWS_box")

doping = st.checkbox("도핑 이력 있음", value=False)
leadership = st.slider("리더십/영향력 점수", 0.0, 1.0, 0.5)
influence = st.slider("커리어/문화적 영향력", 0.0, 1.0, 0.5)
era_adjust = st.slider("시대 보정 (타고투저/투고타저)", -0.3, 0.3, 0.0)

if st.button("예측 실행"):
    res = predict_HOF(name, WAR, HOFm, JAWS, doping, leadership, influence, era_adjust)
    st.text(summarize_result(res))
