# Qbias 데이터

# ===== 0. 필요한 라이브러리 설치 / 임포트 =====
!git clone https://github.com/irgroup/Qbias.git

from google.colab import drive
drive.mount('/content/drive')

import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

data = pd.read_csv("/content/drive/MyDrive/데사 캡스톤/Qbias/allsides_balanced_news_headlines-texts.csv")
print(data)

# 1️⃣ 텍스트 합치기: title + heading
data['input_text'] = data['title'] + " " + data['heading']

# 2️⃣ bias_label 인코딩: left=0, center=1, right=2
label_mapping = {"left": 0, "center": 1, "right": 2}
data['bias_label'] = data['bias_rating'].map(label_mapping)

# 3️⃣ 확인
print(data[['input_text', 'bias_label']].head())
print(data['bias_label'].value_counts())

## DistilBERT 모델

from sklearn.model_selection import train_test_split
from datasets import Dataset

# 1️⃣ train/validation/test 분할
train_df, temp_df = train_test_split(data[['input_text', 'bias_label']], test_size=0.2, stratify=data['bias_label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['bias_label'], random_state=42)

# 2️⃣ HuggingFace Dataset 변환
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# 3️⃣ 확인
print(train_dataset)
print(val_dataset)
print(test_dataset)

from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 토크나이징 함수 정의
def tokenize_batch(batch):
    return tokenizer(batch['input_text'], padding='max_length', truncation=True, max_length=128)

# Dataset에 적용
train_dataset = train_dataset.map(tokenize_batch, batched=True)
val_dataset = val_dataset.map(tokenize_batch, batched=True)
test_dataset = test_dataset.map(tokenize_batch, batched=True)

# 모델 입력용 포맷 변환
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'bias_label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'bias_label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'bias_label'])

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(train_dataset.column_names)

# ===== 4. 모델 설정 =====
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# ===== 5. 학습 =====
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['bias_label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

# ===== 6. 평가 =====
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['bias_label'].to(device)

        outputs = model(input_ids, attention_mask=attn_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print("Bias Test Accuracy:", correct / total)

# ===== 7. 모델 저장 =====
torch.save(model.state_dict(), "/content/drive/MyDrive/데사 캡스톤/혜원_distilbert_bias_model_fromQbias.pt")
print("✅ Qbias 기반 편향 모델 저장 완료!")

## 2. 타이틀+헤딩+본문 병합 버전

import pandas as pd
data = pd.read_csv("/content/drive/MyDrive/데사 캡스톤/데이터/Qbias/allsides_balanced_news_headlines-texts.csv")
print(data)

# ===== 1️⃣ 텍스트 합치기: title + heading + text =====
data['input_text'] = data['title'] + " " + data['heading'] + " " + data['text']

# ===== 2️⃣ bias_label 인코딩 =====
label_mapping = {"left": 0, "center": 1, "right": 2}
data['bias_label'] = data['bias_rating'].map(label_mapping)

print(data[['input_text', 'bias_label']])

편향 = data[['input_text', 'bias_label']].to_csv("/content/drive/MyDrive/데사 캡스톤/데이터/편향 데이터.csv")

# ===== 3️⃣ train/val/test 분할 =====
from sklearn.model_selection import train_test_split
from datasets import Dataset

train_df, temp_df = train_test_split(data[['input_text', 'bias_label']], test_size=0.2, stratify=data['bias_label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['bias_label'], random_state=42)

train_dataset = Dataset.from_pandas(train_df[['input_text', 'bias_label']], preserve_index=False)
val_dataset = Dataset.from_pandas(val_df[['input_text', 'bias_label']], preserve_index=False)
test_dataset = Dataset.from_pandas(test_df[['input_text', 'bias_label']], preserve_index=False)


train_dataset
print(test_dataset)
val_dataset

# ===== 4️⃣ 토크나이징 =====
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_batch(batch):
    # batch['input_text']가 list라 가정하고, str로 변환 + NaN은 빈 문자열로
    texts = [str(x) if x is not None else "" for x in batch["input_text"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize_batch, batched=True)
val_dataset = val_dataset.map(tokenize_batch, batched=True)
test_dataset = test_dataset.map(tokenize_batch, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'bias_label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'bias_label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'bias_label'])

# ===== 5️⃣ DataLoader 생성 =====
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# DistilBERT 모델 정의, 클래스 수는 3 (left/center/right)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# ===== 7️⃣ 학습 =====
num_epochs = 3  # 필요에 따라 조정 가능

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['bias_label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

# ===== 8️⃣ 평가 =====
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['bias_label'].to(device)

        outputs = model(input_ids, attention_mask=attn_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print("Bias Test Accuracy:", correct / total)

# ===== 9️⃣ 모델 저장 =====
torch.save(model.state_dict(), "/content/drive/MyDrive/데사 캡스톤/혜원_distilbert_bias_model2_fromQbias.pt")
print("✅ Qbias 기반 편향 모델 저장 완료!")
