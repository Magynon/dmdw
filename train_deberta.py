from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd

# Load dataset
file_path = 'disneyland_reviews.csv'
df = pd.read_csv(file_path, encoding="cp1252")

# Create Sentiment column based on Rating (>=3 is POSITIVE, <3 is NEGATIVE)
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 3 else 0)  # 1 = POSITIVE, 0 = NEGATIVE

# Split the dataset into training and testing (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')

def preprocess_function(examples):
    return tokenizer(examples['Review_Text'], padding="max_length", truncation=True, max_length=512)

# Convert to Hugging Face Datasets format
train_dataset = Dataset.from_pandas(train_df[['Review_Text', 'Sentiment']])
test_dataset = Dataset.from_pandas(test_df[['Review_Text', 'Sentiment']])

# Tokenize the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Ensure the datasets have the correct format (including 'labels')
train_dataset = train_dataset.rename_column("Sentiment", "labels")
test_dataset = test_dataset.rename_column("Sentiment", "labels")

# Set the format for PyTorch (input_ids, attention_mask, and labels)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load the DeBERTa model for PyTorch
model = DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # evaluate after each epoch
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs

)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./deberta_finetuned_model')
tokenizer.save_pretrained('./deberta_finetuned_model')
