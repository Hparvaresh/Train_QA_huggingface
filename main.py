from utils.read_ds import read_qa
from utils.preprocess import preprocess_function
import pandas as pd
from transformers import AutoTokenizer, DefaultDataCollator, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

# Set parameters
params = {
    "train_data_path": "./data/pqa_train.json",
    "test_data_path": "./data/pqa_test.json",
    "model_dir_path": "./albert-fa-base-v2_pquad_and_persian_qa",
    "output_dir_path": "trained_qa_model",
    "evaluation_strategy": "epoch",
    "learning_rate": 2e-5,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "num_train_epochs": 1,
    "weight_decay": 0.01,
    "push_to_hub": False,
    "num_train_samples": 100,
    "num_test_samples": 20
}

# Read the data
train_data = read_qa(params["train_data_path"])
test_data = read_qa(params["test_data_path"])

# Create a dataset dictionary and convert the data to datasets
dataset = DatasetDict()
dataset["train"] = Dataset.from_pandas(pd.DataFrame(data=train_data).dropna()[:params["num_train_samples"]])
dataset["test"]  = Dataset.from_pandas(pd.DataFrame(data=test_data).dropna()[:params["num_test_samples"]])

# Preprocess the data and tokenize
tokenizer = AutoTokenizer.from_pretrained(params["model_dir_path"])
tokenized_map_data = dataset.map(preprocess_function, batched=True)
data_collator = DefaultDataCollator()

# Load the model weights
model = AutoModelForQuestionAnswering.from_pretrained(params["model_dir_path"])

# Set the training arguments
training_args = TrainingArguments(
    output_dir=params["output_dir_path"],
    evaluation_strategy=params["evaluation_strategy"],
    learning_rate=params["learning_rate"],
    per_device_train_batch_size=params["train_batch_size"],
    per_device_eval_batch_size=params["eval_batch_size"],
    num_train_epochs=params["num_train_epochs"],
    weight_decay=params["weight_decay"],
    push_to_hub=params["push_to_hub"]
)

# Train the model
trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=tokenized_map_data["train"],
     eval_dataset=tokenized_map_data["test"],
     tokenizer=tokenizer,
     data_collator=data_collator,
)
trainer.train()

# Evaluate the model
result = trainer.evaluate()  
print(result)
