import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# 1. Define Intents and Synthetic Data
intents = ["book_search", "stock_check", "debt_check", "policy_query", "general_chat"]
label2id = {intent: i for i, intent in enumerate(intents)}
id2label = {i: intent for i, intent in enumerate(intents)}

data = [
    # book_search
    ("Tìm sách về lập trình Python", "book_search"),
    ("Có sách nào về AI không?", "book_search"),
    ("Gợi ý sách kinh tế vi mô", "book_search"),
    ("Tôi muốn tìm sách của tác giả Ngô Bảo Châu", "book_search"),
    ("Sách về Machine Learning", "book_search"),
    ("Làm sao để tìm sách Giải tích?", "book_search"),
    ("Tìm tài liệu về lịch sử Việt Nam", "book_search"),
    
    # stock_check
    ("Cuốn Clean Code còn sẵn không?", "stock_check"),
    ("Sách lính mới còn ở thư viện không?", "stock_check"),
    ("Ai đang mượn sách Harry Potter?", "stock_check"),
    ("Kiểm tra xem sách Java còn bao nhiêu bản", "stock_check"),
    ("Sách này có cho mượn không?", "stock_check"),
    
    # debt_check
    ("Tôi nợ bao nhiêu tiền phạt?", "debt_check"),
    ("Kiểm tra nợ của sinh viên SE123456", "debt_check"),
    ("Tôi có bị phạt quá hạn không?", "debt_check"),
    ("Số tiền phạt hiện tại của mình là bao nhiêu?", "debt_check"),
    ("Xem danh sách sách đang mượn và tiền nợ", "debt_check"),
    
    # policy_query
    ("Mượn sách được bao nhiêu ngày?", "policy_query"),
    ("Phí phạt quá hạn một ngày là bao nhiêu?", "policy_query"),
    ("Thư viện mở cửa lúc mấy giờ?", "policy_query"),
    ("Quy định mượn trả sách như thế nào?", "policy_query"),
    ("Có được mang đồ ăn vào thư viện không?", "policy_query"),
    ("Làm thẻ thư viện như thế nào?", "policy_query"),
    ("Mượn tối đa bao nhiêu cuốn sách?", "policy_query"),
    
    # general_chat
    ("Xin chào!", "general_chat"),
    ("Bạn là ai?", "general_chat"),
    ("Cảm ơn bạn rất nhiều", "general_chat"),
    ("Chào buổi sáng", "general_chat"),
    ("Bạn có thể giúp gì cho tôi?", "general_chat"),
    ("Tạm biệt", "general_chat"),
]

# 2. Prepare Dataset
texts, labels = zip(*data)
labels = [label2id[l] for l in labels]

dataset = Dataset.from_dict({"text": list(texts), "label": labels})
dataset = dataset.train_test_split(test_size=0.1)

# 3. Load Base Model
model_name = "vinai/phobert-base-v2"  # or specific one recommended by user
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(intents),
    id2label=id2label,
    label2id=label2id
)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./models/intent_classifier",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

# 5. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

print("\n🚀 Starting Fine-tuning PhoBERT for Library Intents...")
trainer.train()

# 6. Save Model
model_path = "./models/lib_intent_phobert"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"✅ Model saved to {model_path}")
