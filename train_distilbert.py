from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd

def main():
    # Load the dataset
    file_path = 'filtered_disneyland_reviews.csv'
    df = pd.read_csv(file_path, encoding="cp1252")

    # Create Sentiment column based on Rating (>=3 is POSITIVE, <3 is NEGATIVE)
    df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 3 else 0)  # 1 = POSITIVE, 0 = NEGATIVE

    # Split the dataset into training and testing (80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Preprocess the dataset (tokenization)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

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

    # Load the DistilBERT model for PyTorch
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./distilbert-results',  # Output directory
        evaluation_strategy="steps",  # Evaluate periodically based on steps, faster feedback
        eval_steps=500,  # Evaluate every 500 steps (adjust as needed)
        save_strategy="steps",  # Save checkpoints periodically
        save_steps=1000,  # Save model every 1000 steps
        save_total_limit=2,  # Keep only the last 2 checkpoints
        learning_rate=3e-5,  # Slightly higher learning rate for faster convergence
        per_device_train_batch_size=16,  # Larger batch size for faster training
        per_device_eval_batch_size=16,  # Match eval batch size with train batch size
        num_train_epochs=1,  # Reduce to 1 epoch to save time
        weight_decay=0.01,  # Strength of weight decay
        logging_dir='./logs',  # Directory for logs
        logging_steps=100,  # Log every 100 steps
        dataloader_num_workers=4,  # Utilize more CPU workers for data loading
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
    model.save_pretrained('./distilbert_finetuned_model')
    tokenizer.save_pretrained('./distilbert_finetuned_model')


if __name__ == "__main__":
    main()