import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset
from evaluate import load
import numpy as np

label2id = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}
id2label = {v: k for k, v in label2id.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./ner_model')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset('json', data_files={
        'train': f'{args.dataset_path}/train.json',
        'validation': f'{args.dataset_path}/val.json'
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    def tokenize_and_align_labels(examples):
        label2id = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}
        
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            padding='max_length',
            max_length=128,
            is_split_into_words=True,
            return_tensors="pt"
        )
        
        labels = []
        for i, label_seq in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label = label_seq[word_idx]
                    label_id = label2id[label]
                    label_ids.append(label_id)
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Tokenize dataset
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=args.per_device_train_batch_size
    )

    # Use proper data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Update training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        fp16=True,  # Enable mixed precision training
        report_to="none"
    )

    metric = load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        id2label = {0: "O", 1: "B-ANIMAL", 2: "I-ANIMAL"}
        
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(
            predictions=true_predictions, 
            references=true_labels,
            zero_division=0
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()