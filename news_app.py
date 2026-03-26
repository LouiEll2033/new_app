import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os

# --- 페이지 설정 ---
st.set_page_config(page_title="뉴스 카테고리 분류기", page_icon="📰", layout="centered")

# --- 모델 학습 및 로드 함수 (캐싱 처리) ---
# 앱이 실행될 때마다 모델을 새로 학습하지 않도록 캐싱합니다.
@st.cache_resource
def load_or_train_model():
    model_filename = 'news_model.joblib'
    
    # 한국어 카테고리 매핑 (영문 데이터셋을 한국어 카테고리처럼 시뮬레이션)
    category_map = {
        'talk.politics.misc': '정치',
        'sci.space': 'IT/과학 (우주)',
        'rec.sport.hockey': '스포츠',
        'sci.electronics': 'IT/과학 (전자)',
        'comp.graphics': 'IT/과학 (그래픽)',
        'rec.motorcycles': '생활/문화 (오토바이)'
    }
    
    # 데모용으로 6개 카테고리만 선택
    categories = list(category_map.keys())

    # 데이터셋 다운로드 (최초 1회만 다운로드됨)
    with st.spinner('데모용 모델을 준비 중입니다... (최초 실행 시 약간의 시간이 소요됩니다)'):
        newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
        
        # 파이프라인 생성: TF-IDF 벡터화 + 나이브 베이즈 분류기
        model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
        
        # 모델 학습
        model.fit(newsgroups_train.data, newsgroups_train.target)
        
        # 영문 타겟 이름을 한국어 이름 리스트로 변환하여 저장
        target_names_ko = [category_map[name] for name in newsgroups_train.target_names]
        
    return model, target_names_ko

# --- 메인 앱 UI ---
def main():
    st.title("📰 AI 뉴스 기사 카테고리 분류기")
    st.markdown("""
    뉴스 기사의 본문을 입력하면 AI가 내용을 분석하여 카테고리를 자동으로 분류합니다.
    
    * **참고:** 이 데모 앱은 학습 데이터의 한계로 인해 **영문 텍스트** 입력에 최적화되어 있습니다. 
        (내부적으로 Scikit-learn의 20 Newsgroups 영어 데이터셋을 활용해 학습되었습니다.)
    """)

    # 모델 로드
    model, target_names_ko = load_or_train_model()

    st.divider()

    # --- 입력 영역 ---
    st.subheader("심사할 뉴스 기사 입력")
    news_text = st.text_area("기사 본문을 입력하세요 (영문 권장)", height=300, placeholder="여기에 기사 내용을 붙여넣으세요...")

    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_btn = st.button("카테고리 분류하기", type="primary", use_container_width=True)

    # --- 결과 출력 영역 ---
    if predict_btn:
        if not news_text.strip():
            st.warning("⚠️ 기사 내용을 입력해 주세요.")
        else:
            with st.spinner('AI가 분석 중입니다...'):
                # 예측 수행
                prediction = model.predict([news_text])
                predicted_category = target_names_ko[prediction[0]]
                
                # 예측 확률 계산
                proba = model.predict_proba([news_text])[0]
                max_proba = proba.max()

            # 결과 화면 출력
            st.success(f"### 🎉 분석 결과: [ {predicted_category} ] 뉴스일 확률이 높습니다.")
            st.metric(label="분류 신뢰도", value=f"{max_proba*100:.1f}%")

            # 확률 분포 시각화
            st.write("---")
            st.subheader("📊 카테고리별 분석 상세 (신뢰도)")
            
            # 데이터프레임 생성
            proba_df = pd.DataFrame({
                '카테고리': target_names_ko,
                '확률 (%)': proba * 100
            }).sort_values(by='확률 (%)', ascending=False)
            
            # 바 차트 출력
            st.bar_chart(proba_df.set_index('카테고리'))
            
            # 테이블 출력
            st.dataframe(proba_df.style.format({'확률 (%)': '{:.1f}'}), use_container_width=True)

    # --- 사이드바 및 설명 ---
    with st.sidebar:
        st.header("앱 정보")
        st.markdown("""
        **작동 원리:**
        1.  **TF-IDF:** 입력된 텍스트에서 중요한 단어들을 추출하여 숫자로 바꿉니다.
        2.  **Multinomial Naive Bayes:** 학습된 확률 모델을 바탕으로 어떤 카테고리에 속할지 계산합니다.

        **학습 데이터:**
        Scikit-learn에서 제공하는 '20 Newsgroups' 영문 이메일/뉴스 데이터셋을 활용했습니다.

        **지원 카테고리 (데모):**
        * 정치
        * 스포츠
        * IT/과학 (우주, 전자, 그래픽)
        * 생활/문화 (오토바이)
        """)
        
        st.info("실제 서비스용 앱을 만들려면 한국어 뉴스 데이터셋(예: KLUE)으로 학습된 BERT 등의 딥러닝 모델을 탑재해야 합니다.")

if __name__ == '__main__':
    main()