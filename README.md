# 📰 Neutral News Rewriting Pipeline  
**Integrated Truthfulness, Sentiment, and Political Bias Analysis for Neutral Headline Generation**

---

## 📌 Overview  
이 프로젝트는 영어 뉴스 헤드라인을 대상으로  

✔ 진위 판별  
✔ 감성 분석  
✔ 정치적 편향 검출  
✔ 중립적 재작성(Neutral Rewriting)  

을 **통합적으로 수행하는 NLP 파이프라인**입니다.

각 분석 결과(라벨 + 확률)는 T5 재작성 모델의 입력으로 전달되어 핵심 의미는 유지하면서 **감정·편향·과장 표현을 완화한 중립적 문장**을 생성합니다.

또한 재작성 전·후 결과를 **레이더 차트로 시각화**해  중립화 효과를 직관적으로 확인할 수 있습니다.

---

## 🚀 Pipeline Architecture  

아래의 4단계로 구성된 End-to-End 파이프라인입니다.

### **1. Truthfulness Classification (Fake News Detection)**

- **Dataset**: FakeNewsNet (Train 18,556 / Test 4,640)  
- **Models compared**: DistilBERT, BERT-base  
- **Result**  
  - DistilBERT: 85–86% accuracy  
  - 학습 속도 & 경량성 우수 → **최종 선택**

---

### **2. Sentiment Analysis**

- **Dataset**: TweetEval Sentiment  
- **Models compared**: DeBERTa-v3-base, DistilBERT  
- **Result**:  
  - **DeBERTa-v3-base** 72% accuracy → **최종 선택**

---

### **3. Political Bias Detection**

- **Input**: 제목 + 헤딩 + 본문  
- **Model**: DistilBERT  
- **Performance**: Accuracy **0.63**

설계 포인트:  
- 편향 문제는 본질적으로 **변동성·불확실성이 큰 영역**  
- 따라서 정확도에 과적합하지 않고 **label + confidence만 T5에 전달**하는 구조 채택  
- 불확실성에도 불구하고 재작성 출력은 자연스럽게 중립으로 수렴

---

### **4. Neutral Rewriting (T5-base)**

- **Model**: T5-base, 단계적(stepwise) fine-tuning  
- **Training size**: 1K → 5K → 10K  
- 안정적인 Loss·ROUGE 유지  
- 단순 paraphrasing이 아닌 **meaning-preserving neutralization** 수행

---

## 🔗 Full Integrated Pipeline

모든 분석 결과를 아래 프롬프트 구조로 통합해 T5에 전달합니다:



### 이 방식의 장점
- 핵심 의미 유지  
- 감정·편향·과장 표현 자동 제거  
- 복합 분석 기반의 **의미적 중립화** 달성

---

## 📊 Experimental Results

### **Neutralization Example**

#### **Original**
> The greedy politicians passed another useless law to affect hardworking citizens.

#### **Analysis Results**
- Truth: False (97.1%)  
- Sentiment: Negative (99.7%)  
- Bias: Left (87.4%)

<img width="383" height="396" alt="Image" src="https://github.com/user-attachments/assets/db0dd9b7-3510-45f1-97cd-d2c9288645bb" />


#### **Neutralized Output**
> The politicians passed another law to affect hardworking citizens.

#### **After Rewriting**
- Truth: **Truthful (68.0%)**  
- Sentiment: **Neutral (55.3%)**  
- Bias: **Near-neutral (53.6%)**

<img width="383" height="396" alt="Image" src="https://github.com/user-attachments/assets/edf47239-5462-448d-94e6-8c53175b9464" />

→ 감정적 단어 제거  
→ 공격성 완화  
→ 의미 유지  
가 수치적으로 검증됨.

---

## 📈 Visualization

중립 전·후 변화를 **레이더 차트**로 시각화해 다음 요소의 변화를 파악합니다:

- 감정 강도  
- 편향 정도  
- 진위 확률  



---

## 🧾 Conclusion  

본 프로젝트는  
**진위·감정·편향을 통합 분석하고, 이를 기반으로 T5가 의미적 중립화를 수행하는 파이프라인**을 제안했습니다.

단일 라벨 분류가 아닌  
**다요인 기반 중립화 모델**이라는 점에서 기존 연구와 차별화되며, 실제 뉴스 소비에서 발생하는 **인지 편향 완화**에 기여할 수 있습니다.

---

## 🔮 Future Work
- 다국어 확장  
- 실시간 뉴스 스트림 중립화  
- 사용자 편향 프로파일 기반 맞춤 중립화  

---

## 📚 References  

1. Fake News Detection Using Document Bias and Sentiment Analysis, HCI 2023  
2. FakeNewsNet Dataset  
3. LIAR Dataset (Fake News Benchmark)  
