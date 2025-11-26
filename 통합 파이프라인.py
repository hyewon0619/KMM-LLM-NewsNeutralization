# 라이브러리

import torch
import joblib
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import numpy as np

# ================= 공통 설정 =================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 1. 진위 판별 모델 =================
truth_model_path = "/content/drive/MyDrive/데사 캡스톤/모델/혜원_distilbert_fake_model.pt"
truth_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
truth_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
truth_model.load_state_dict(torch.load(truth_model_path, map_location=device))
truth_model.to(device).eval()

# ================= 2️. 감정 분석 모델 =================
sentiment_model_path = "/content/drive/MyDrive/데사 캡스톤/모델/혜원_distilbert_sentiment_model2.pt"
sentiment_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
sentiment_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
sentiment_model.load_state_dict(torch.load(sentiment_model_path, map_location=device))
sentiment_model.to(device).eval()

# ================= 3️. 편향 분류 모델 =================
bias_model_path = "/content/drive/MyDrive/데사 캡스톤/모델/혜원_distilbert_bias_model2_fromQbias.pt"
bias_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bias_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
bias_model.load_state_dict(torch.load(bias_model_path, map_location=device))
bias_model.to(device).eval()

# ================= 4️. 뉴스 재작성 (중립화) 모델 =================
rewrite_model_path = "/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-final2"
rewrite_weights_path = "/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-final2_weights.pt"

rewrite_tokenizer = T5Tokenizer.from_pretrained(rewrite_model_path)
rewrite_model = T5ForConditionalGeneration.from_pretrained("t5-base")
rewrite_model.load_state_dict(torch.load(rewrite_weights_path, map_location=device))
rewrite_model.to(device).eval()


# ================= 라벨 매핑 =================
truth_map = {0: "truthful", 1: "false"}
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
bias_map = {0: "left", 1: "center", 2: "right"}


# ================= 예측 함수 (확률 포함) =================
def predict_label_with_confidence(model, tokenizer, text, num_labels, label_map):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)  # 확률 계산
        conf, pred_idx = torch.max(probs, dim=1)
        conf = conf.item()
        pred_idx = pred_idx.item()
        pred_label = label_map[pred_idx]
    return pred_label, conf

def plot_radar(truth_conf, sentiment_conf, bias_conf, truth_label, sentiment_label, bias_label):
    labels = [truth_label, sentiment_label, bias_label]
    values = [truth_conf, sentiment_conf, bias_conf]

    # 레이더 차트용 각도 계산
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,1)
    plt.show()


# ================= 뉴스 재작성 함수 =================
def rewrite_news(text):
    # ========== 원문 분석 ==========
    truth_str, truth_conf = predict_label_with_confidence(truth_model, truth_tokenizer, text, 2, truth_map)
    sentiment_str, sentiment_conf = predict_label_with_confidence(sentiment_model, sentiment_tokenizer, text, 3, sentiment_map)
    bias_str, bias_conf = predict_label_with_confidence(bias_model, bias_tokenizer, text, 3, bias_map)

    print(f"[원문 분석 결과]")
    print(f"- 진위: {truth_str} ({truth_conf*100:.1f}%)")
    print(f"- 감정: {sentiment_str} ({sentiment_conf*100:.1f}%)")
    print(f"- 편향: {bias_str} ({bias_conf*100:.1f}%)")

    # 레이더 차트 시각화
    plot_radar(truth_conf, sentiment_conf, bias_conf, truth_str, sentiment_str, bias_str)

    # ========== 뉴스 중립화 ==========
     prompt = (
        f"Neutralize the following sentence: {text}"
        f"neutralize: Rewrite the following news text in a neutral and factual tone.\n"
        f"Analysis summary:\n"
        f"- Truthfulness: {truth_str}\n"
        f"- Sentiment: {sentiment_str}\n"
        f"- Bias: {bias_str}\n\n"
        f"Original: {text}"
    )

    inputs = rewrite_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = rewrite_model.generate(**inputs, max_length=256)
    rewritten = rewrite_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ========== 중립화 후 분석 ==========
    truth_str_n, truth_conf_n = predict_label_with_confidence(truth_model, truth_tokenizer, rewritten, 2, truth_map)
    sentiment_str_n, sentiment_conf_n = predict_label_with_confidence(sentiment_model, sentiment_tokenizer, rewritten, 3, sentiment_map)
    bias_str_n, bias_conf_n = predict_label_with_confidence(bias_model, bias_tokenizer, rewritten, 3, bias_map)

    print(f"\n[중립화 후 분석 결과]")
    print(f"- 진위: {truth_str_n} ({truth_conf_n*100:.1f}%)  |  변경: {truth_conf_n - truth_conf:+.2f}")
    print(f"- 감정: {sentiment_str_n} ({sentiment_conf_n*100:.1f}%)  |  변경: {sentiment_conf_n - sentiment_conf:+.2f}")
    print(f"- 편향: {bias_str_n} ({bias_conf_n*100:.1f}%)  |  변경: {bias_conf_n - bias_conf:+.2f}")

    # 레이더 차트 시각화 (중립화 후)
    plot_radar(truth_conf_n, sentiment_conf_n, bias_conf_n, truth_str_n, sentiment_str_n, bias_str_n)

    return rewritten


# ================= 테스트 =================
if __name__ == "__main__":
    sample_text = "The greedy politicians passed another useless law to affect hardworking citizens."
    result = rewrite_news(sample_text)
    print("\n[원문]")
    print(sample_text)
    print("\n[중립화 결과]")
    print(result)
