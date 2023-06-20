from datasets import DatasetDict
from utils.read_ds import read_qa
from transformers import pipeline
from utils.preprocess import preprocess_function
from transformers import DefaultDataCollator, AutoTokenizer
from transformers import AutoModelForQuestionAnswering,TrainingArguments,Trainer
from datasets import load_dataset
from datasets import Dataset
import pandas as pd


a=read_qa('./data/pqa_train.json')
dataset = DatasetDict()
dataset["train"] = Dataset.from_pandas(pd.DataFrame(data=read_qa('./data/pqa_train.json')).dropna()[:100])
dataset["test"]  = Dataset.from_pandas(pd.DataFrame(data=read_qa('./data/pqa_test.json')).dropna()[:20])
# dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("./albert-fa-base-v2_pquad_and_persian_qa")

tokenized_map_data = dataset.map(preprocess_function, batched=True)

data_collator = DefaultDataCollator()

model = AutoModelForQuestionAnswering.from_pretrained("./albert-fa-base-v2_pquad_and_persian_qa")

training_args = TrainingArguments(
     output_dir="traned_qa_model",
     evaluation_strategy="epoch",
     learning_rate=2e-5,
     per_device_train_batch_size=8,
     per_device_eval_batch_size=8,
     num_train_epochs=1,
     weight_decay=0.01,
     push_to_hub=False,
 
)

trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=tokenized_map_data["train"],
     eval_dataset=tokenized_map_data["test"],
     tokenizer=tokenizer,
     data_collator=data_collator,
 )

trainer.train()
result = trainer.evaluate()  
print(result)