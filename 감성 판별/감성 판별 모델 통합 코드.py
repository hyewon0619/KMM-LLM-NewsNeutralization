from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
import torch

# ===== 1. 데이터셋 로드 =====
dataset = load_dataset("tweet_eval", "sentiment")

# train/validation/test 데이터 확인
print(dataset)

# ===== 2. 토크나이저 =====
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def encode_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

dataset_encoded = dataset.map(encode_batch, batched=True)

# ===== 3. 텐서로 변환 =====
def to_tensordataset(split):
    inputs = torch.tensor(dataset_encoded[split]['input_ids'])
    masks = torch.tensor(dataset_encoded[split]['attention_mask'])
    labels = torch.tensor(dataset_encoded[split]['label'])
    return TensorDataset(inputs, masks, labels)

train_dataset = to_tensordataset('train')
test_dataset = to_tensordataset('test')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

dataset = load_dataset("tweet_eval", "sentiment")


# ===== 4. 모델 =====
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# ===== 5. 학습 =====
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        input_ids, attn_mask, labels = [b.to(device) for b in batch]
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
        input_ids, attn_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attn_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total:.4f}")

# ===== 7. 저장 =====
torch.save(model.state_dict(), '/content/drive/MyDrive/데사 캡스톤/혜원_distilbert_sentiment_model1.pt')
print("✅ 감정 판별 모델 저장 완료!")

# 2번째 시도 - 전처리 적용

from datasets import load_dataset

# ===== 1. 데이터셋 로드 =====
dataset = load_dataset("tweet_eval", "sentiment")

# train/validation/test 데이터 확인
print(dataset)
print(dataset['train'][0]['text'])

import re

# ===== 1. 이모지/특수문자 제거 (띄어쓰기 유지) =====
def clean_text(example):
    text = example['text']
    # 기본 문자, 숫자, 공백만 남기고 제거
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    example['text'] = text.strip()
    return example

dataset = dataset.map(clean_text)

print(type(dataset['train'][0]['text']))
print(dataset['train'][0]['text'])
print(type(dataset['train'][:5]['text']))
print(dataset['train'][:5]['text'])

# ===== 2. 토크나이징 =====
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def encode_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

dataset_encoded = dataset.map(encode_batch, batched=True)

# ===== 3. TensorDataset 변환 =====
def to_tensordataset(split):
    inputs = torch.tensor(dataset_encoded[split]['input_ids'])
    masks = torch.tensor(dataset_encoded[split]['attention_mask'])
    labels = torch.tensor(dataset_encoded[split]['label'])
    return TensorDataset(inputs, masks, labels)

train_dataset = to_tensordataset('train')
test_dataset = to_tensordataset('test')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ===== 4. 모델 & 옵티마이저 =====
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# ===== 5. 학습 루프 =====
num_epochs = 4  # 에포크 조금 늘림
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        input_ids, attn_mask, labels = [b.to(device) for b in batch]
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
        input_ids, attn_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attn_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total:.4f}")

torch.save(model.state_dict(), '/content/drive/MyDrive/데사 캡스톤/혜원_distilbert_sentiment_model2.pt')
print("✅ 감정 판별 모델2 저장 완료!")
