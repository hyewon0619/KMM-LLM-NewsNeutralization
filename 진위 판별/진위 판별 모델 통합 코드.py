# 전처리

import pandas as pd

politifact_fake = pd.read_csv('/content/drive/MyDrive/데사 캡스톤/FakeNewsNet 데이터셋/politifact_fake.csv')
politifact_real = pd.read_csv('/content/drive/MyDrive/데사 캡스톤/FakeNewsNet 데이터셋/politifact_real.csv')
gossipcop_fake = pd.read_csv('/content/drive/MyDrive/데사 캡스톤/FakeNewsNet 데이터셋/gossipcop_fake.csv')
gossipcop_real = pd.read_csv('/content/drive/MyDrive/데사 캡스톤/FakeNewsNet 데이터셋/gossipcop_real.csv')

print(politifact_fake)
print(politifact_real)
print(gossipcop_fake)
print(gossipcop_real)

politifact_fake['fake_label'] = 1
politifact_real['fake_label'] = 0
gossipcop_fake['fake_label'] = 1
gossipcop_real['fake_label'] = 0

df = pd.concat([
    politifact_fake,
    politifact_real,
    gossipcop_fake,
    gossipcop_real
])

df = df[['title', 'fake_label']]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("/content/drive/MyDrive/데사 캡스톤/진위 데이터.csv")

print(df)
print(df.isnull().sum())

from sklearn.model_selection import train_test_split

# 학습/검증 분리 (8:2 비율)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['fake_label'])

print(train_df.shape, test_df.shape)

!pip install transformers datasets evaluate -q

!pip install --upgrade transformers

# DistilBERT 모델

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ===== 1. 토크나이저 불러오기 =====
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# ===== 2. 텍스트 토큰화 =====
train_encodings = tokenizer(list(train_df['title']), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_df['title']), truncation=True, padding=True, max_length=128)

# ===== 3. 토큰화 결과를 Tensor로 변환 =====
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(train_df['fake_label'].values)

test_input_ids = torch.tensor(test_encodings['input_ids'])
test_attention_mask = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(test_df['fake_label'].values)

# ===== 4. TensorDataset & DataLoader =====
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

from torch.optim import AdamW
from tqdm import tqdm

# -------------------------------
# 4. 모델 + 옵티마이저
# -------------------------------
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# -------------------------------
# 5. 학습 루프 (epoch 2)
# -------------------------------
num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # 진행률 + loss 표시
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

# -------------------------------
# 6. 간단 평가
# -------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total:.4f}")

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm

# ===== 1. 디바이스 설정 =====
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ===== 2. 토크나이저 불러오기 =====
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# ===== 3. 텍스트 토큰화 =====
train_encodings = tokenizer(list(train_df['title']), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_df['title']), truncation=True, padding=True, max_length=128)

# ===== 4. Tensor 변환 =====
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(train_df['fake_label'].values)

test_input_ids = torch.tensor(test_encodings['input_ids'])
test_attention_mask = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(test_df['fake_label'].values)

# ===== 5. Dataset & DataLoader =====
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ===== 6. 모델 & 옵티마이저 =====
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# ===== 7. 학습 루프 (epoch 3) =====
num_epochs = 3  # 기존 2 -> 3으로 증가

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # 진행률 + loss 표시
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

# 테스트 정확도 평가
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total:.4f}")

# 9. 모델 저장
torch.save(model.state_dict(), '혜원_distilbert_fake_model.pt')
