# 데이터

## 데이터셋 생성

import pandas as pd

truth_df = pd.read_csv("/content/drive/MyDrive/데사 캡스톤/데이터/진위 데이터.csv")

truth_sample = truth_df[truth_df['fake_label'] == 1].sample(1000, random_state=42)
truth_sample = truth_sample[['title']].rename(columns={'title': 'text'})

from datasets import load_dataset

sentiment_ds = load_dataset("tweet_eval", "sentiment")
# label 0(negative) 또는 2(positive)인 샘플 1000개
sentiment_df = pd.DataFrame(sentiment_ds['train'])
sentiment_sample = sentiment_df[sentiment_df['label'].isin([0,2])].sample(1000, random_state=42)
sentiment_sample = sentiment_sample[['text']]

# ===== 3. 편향 데이터 =====
bias_df = pd.read_csv("/content/drive/MyDrive/데사 캡스톤/데이터/Qbias/allsides_balanced_news_headlines-texts.csv")
# bias_rating이 left 또는 right인 샘플 1000개, title+heading+text 합치기
bias_df_filtered = bias_df[bias_df['bias_rating'].isin(['left','right'])].sample(1000, random_state=42)
bias_sample = bias_df_filtered['title'] + " " + bias_df_filtered['heading'] + " " + bias_df_filtered['text']
bias_sample = pd.DataFrame({'text': bias_sample})

# ===== 4. 3개 통합 =====
combined_df = pd.concat([truth_sample, sentiment_sample, bias_sample], ignore_index=True)
print(combined_df.shape)
print(combined_df.head())

# ===== 5. 필요하면 CSV로 저장 =====
combined_df.to_csv("/content/drive/MyDrive/데사 캡스톤/데이터/rewrite_training_data.csv", index=False)

## 중립적 문장 데이터셋

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

texts = combined_df["text"].tolist()

def make_prompt(text):
    return f"Rewrite this news headline to be neutral: {text}"

from torch.utils.data import DataLoader

batch_size = 16  # GPU 메모리 상황에 맞게 조절

outputs = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    prompts = [make_prompt(t) for t in batch]
    encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    generated_ids = model.generate(**encodings, max_length=256)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    outputs.extend(decoded)

combined_df["neutral_text"] = outputs
combined_df.to_csv("/content/drive/MyDrive/데사 캡스톤/rewrite_neutral_texts.csv", index=False)

print(combined_df)
print(combined_df['neutral_text'])

# 1️⃣ 프롬프트 제거
combined_df["neutral_text"] = combined_df["neutral_text"].str.replace(
    r"(Rewrite this news headline to be neutral:|be neutral:)\s*", "", regex=True
).str.lstrip(": ").str.strip()

# 2️⃣ 시작이 ':'로 시작하면 제거
combined_df["neutral_text"] = combined_df["neutral_text"].str.lstrip(": ").str.strip()

# 3️⃣ 널값 확인
null_count = combined_df["neutral_text"].isnull().sum()
print(f"널값 개수: {null_count}")

# 4️⃣ 결과 확인
print(combined_df.head())

empty_count = (combined_df["neutral_text"].isnull() | (combined_df["neutral_text"].str.strip() == "")).sum()
print(f"널 혹은 빈 문자열 개수: {empty_count}")

combined_df = combined_df[combined_df["neutral_text"].str.strip() != ""]
combined_df = combined_df.dropna(subset=["neutral_text"]).reset_index(drop=True)

print(f"삭제 후 데이터 개수: {combined_df.shape[0]}")

combined_df.to_csv("/content/drive/MyDrive/데사 캡스톤/rewrite_neutral_texts.csv", index=False)

import pandas as pd

# 전체 열 내용 다 보이게 설정
pd.set_option('display.max_colwidth', None)

# 0~4행만 보기
print(combined_df.loc[0:4])

import pandas as pd
import re

# 1️⃣ 프롬프트 문구 제거
combined_df["neutral_text"] = combined_df["neutral_text"].str.replace(
    r"^(Rewrite this news headline to be neutral:|be neutral:)\s*", "", regex=True
)

# 2️⃣ 시작이 ':'나 공백으로 시작하면 제거
combined_df["neutral_text"] = combined_df["neutral_text"].str.lstrip(": ").str.strip()

# 3️⃣ 중복 반복 제거 (같은 문장이 반복되면 한 번만)
def remove_repeats(text):
    if pd.isnull(text) or text.strip() == "":
        return ""
    # 문장 단위로 분리
    sentences = re.split(r'(?<=[.!?]) +', text)
    seen = set()
    cleaned = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            cleaned.append(s_clean)
            seen.add(s_clean)
    return " ".join(cleaned)

combined_df["neutral_text"] = combined_df["neutral_text"].apply(remove_repeats)

# 4️⃣ 널 또는 빈 문자열 행 제거
combined_df = combined_df[combined_df["neutral_text"].str.strip() != ""]

# 5️⃣ 결과 확인
print("정리 후 데이터 수:", combined_df.shape[0])
print(combined_df.head())

## Wiki Neutrality Corpus (WNC) 데이터

import pandas as pd

wnc_df = pd.read_csv(
    '/content/drive/MyDrive/데사 캡스톤/데이터/biased.full',
    sep='\t',
    names=[
        "id", "src_tok", "tgt_tok", "src_raw", "tgt_raw", "src_POS_tags", "tgt_parse_tags"
    ],
    quoting=3,             # 따옴표 문제 방지
    on_bad_lines='skip',   # 문제 있는 줄 건너뛰기
    engine='python'        # 파서 안정성 높이기
)

# 편향 문장 ↔ 중립 문장 선택
train_df = wnc_df[["src_raw", "tgt_raw"]].rename(
    columns={"src_raw": "biased_text", "tgt_raw": "neutral_text"}
)

# 결측치 제거
train_df = train_df.dropna(subset=["biased_text", "neutral_text"]).reset_index(drop=True)

# 미리보기
print(train_df.sample(5))

train_df

# 저장
train_df.to_csv('/content/drive/MyDrive/데사 캡스톤/데이터/재작성 데이터.csv', index=False)

# 재작성 모델 훈련

import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/데사 캡스톤/데이터/재작성 데이터.csv")
print(df)

## T5-base 모델

!pip install transformers datasets accelerate

!pip install transformers datasets accelerate evaluate

!pip install rouge_score

!pip install --upgrade transformers datasets evaluate rouge_score

from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import math

# 프롬프트 구성 (입력/출력)
df["input_text"] = "Neutralize the following sentence: " + df["biased_text"]
df["target_text"] = df["neutral_text"]

# Hugging Face Dataset 변환
dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

# 토크나이저 로드
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def preprocess(batch):
    inputs = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=256)
    inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels["input_ids"]
    ]
    return inputs

# 모델 로드
import evaluate
from transformers import T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
model = T5ForConditionalGeneration.from_pretrained("t5-base")
import numpy as np

# ====== 📊 평가 함수 (ROUGE) ======
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # preds가 tuple로 들어올 경우 flatten
    if isinstance(preds, tuple):
        preds = preds[0]

    # logits일 수도 있으므로 argmax 처리
    if hasattr(preds, "ndim") and preds.ndim > 1:
        preds = np.argmax(preds, axis=-1)

    # ✅ labels에 -100이 포함되어 있으면 tokenizer가 디코딩 못 함
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # preds와 labels가 리스트가 아닐 경우 대비 flatten
    preds = preds.tolist() if isinstance(preds, np.ndarray) else preds
    labels = labels.tolist() if isinstance(labels, np.ndarray) else labels

    # 디코딩
    decoded_preds = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
    decoded_labels = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v * 100, 2) for k, v in result.items()}

i_start = 0  # 예: 이미 2배치 학습 완료 → i_start = 2

batch_size = 1000
num_batches = (len(dataset) + batch_size - 1) // batch_size  # 전체 배치를 계산

for i in range(i_start, num_batches):
    print(f"===== 학습 배치 {i+1}/{num_batches} =====")

    start = i * batch_size
    end = start + batch_size
    subset = dataset.select(range(start, min(end, len(dataset))))
    subset = subset.train_test_split(test_size=0.05, seed=42)
    tokenized_subset = subset.map(preprocess, batched=True)

    # 이전 모델 불러오기
    if i == 0 and i_start == 0:
        model_path = "t5-base"  # 처음이면 t5-base 초기화
    else:
        model_path = f"/content/t5-neutralizer-batch{i}"  # 이전 배치 모델

    model = T5ForConditionalGeneration.from_pretrained(model_path)

    training_args = Seq2SeqTrainingArguments(
        output_dir= f"/content/t5-neutralizer-batch{i+1}",
        per_device_train_batch_size=4,  # GPU 상황에 맞춰 조절
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        logging_steps=50,
        save_total_limit=1,
        eval_strategy="epoch", # no
        save_strategy="epoch",
        learning_rate=5e-5,
        fp16=True,
        load_best_model_at_end=True, #False
        predict_with_generate=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_subset["train"],
        eval_dataset=tokenized_subset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 학습 후 저장
    trainer.save_model(f"/content/t5-neutralizer-batch{i+1}")
    tokenizer.save_pretrained(f"/content/t5-neutralizer-batch{i+1}")

    # if i > 2:  # 저장 시킨 모델의 전전 모델 삭제
    #   shutil.rmtree(f"/content/t5-neutralizer-batch{i-1}")

    print(f"===== 배치 {i+1} 학습 완료 및 저장 =====\n")

# 학습 버전 2

import evaluate
# ====== 📊 평가 함수 (ROUGE) ======
rouge = evaluate.load("rouge")
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    if hasattr(preds, "ndim") and preds.ndim > 1:
        preds = np.argmax(preds, axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = preds.tolist() if isinstance(preds, np.ndarray) else preds
    labels = labels.tolist() if isinstance(labels, np.ndarray) else labels
    decoded_preds = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
    decoded_labels = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v * 100, 2) for k, v in result.items()}

## 2000개씩 누적 학습

import torch  # 🔹 추가
import evaluate
from transformers import T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
import os

# 모델 초기화
model = T5ForConditionalGeneration.from_pretrained("t5-base")


i_start = 0   # 이거 갱신하기 !!!!!!!!!!!!!!

batch_size = 2000
num_batches = (len(dataset) + batch_size - 1) // batch_size

for i in range(i_start, 8):
    print(f"===== 학습 배치 {i+1}/{num_batches} =====")

    start = i * batch_size
    end = start + batch_size
    subset = dataset.select(range(start, min(end, len(dataset))))
    subset = subset.train_test_split(test_size=0.05, seed=42)
    tokenized_subset = subset.map(preprocess, batched=True)

    # 🔹 이전 배치 학습 weight 불러오기 (state_dict 방식)
    if i > 0:
        state_path = f"/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-batch{i}_weights.pt"
        model.load_state_dict(torch.load(state_path))
        print(f"🔹 이전 배치 weight({state_path}) 불러옴")

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-batch{i+1}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        logging_steps=50,
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        fp16=True,
        load_best_model_at_end=True,
        predict_with_generate=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_subset["train"],
        eval_dataset=tokenized_subset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 🔹 학습 후 state_dict 저장
    state_path = f"/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-batch{i+1}_weights.pt"
    torch.save(model.state_dict(), state_path)
    print(f"🔹 배치 {i+1} weight 저장: {state_path}")

    # 🔹 tokenizer 저장 (변화 없음)
    tokenizer.save_pretrained(f"/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-batch{i+1}")

    print(f"===== 배치 {i+1} 학습 완료 및 저장 =====\n")

# 삭제

!rm -rf /content/t5-neutralizer-batch2*

## 최적 선택 -> 5000개 1차

import torch
from transformers import T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np

# 1️⃣ 모델 초기화
model = T5ForConditionalGeneration.from_pretrained("t5-base")
model.load_state_dict(torch.load("/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-batch4_weights.pt"))

# 2️⃣ 5000개 데이터 선택 및 tokenization
subset = dataset.select(range(5000))
subset = subset.train_test_split(test_size=0.05, seed=42)
tokenized_subset = subset.map(preprocess, batched=True)

# 3️⃣ Trainer 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-final1",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=5e-5,
    fp16=True,
    load_best_model_at_end=True,
    predict_with_generate=True,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_subset["train"],
    eval_dataset=tokenized_subset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 4️⃣ 학습 시작
#trainer.train()
# 이어서
trainer.train(resume_from_checkpoint=False)

# 5️⃣ 학습 후 weight 저장
torch.save(model.state_dict(), "/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-final1_weights.pt")
tokenizer.save_pretrained("/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-final1")

## 10000개 2차 (파라미터 조정)

import torch
from transformers import T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np

# 모델 weight 불러오기
model = T5ForConditionalGeneration.from_pretrained("t5-base")
model.load_state_dict(torch.load("/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-final1_weights.pt"))

# 데이터 준비 (10000개)
subset = dataset.select(range(10000))
subset = subset.train_test_split(test_size=0.05, seed=None)  # 시드 없애서 다양화
tokenized_subset = subset.map(preprocess, batched=True)

# Trainer 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-final2",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=5e-5,
    fp16=True,
    load_best_model_at_end=True,  # 체크포인트
    predict_with_generate=True,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_subset["train"],
    eval_dataset=tokenized_subset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 학습 시작
trainer.train()

# 최종 weight 저장
torch.save(model.state_dict(), "/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-final2_weights.pt")
tokenizer.save_pretrained("/content/drive/MyDrive/데사 캡스톤/모델/t5-neutralizer-final2")
