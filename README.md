
# Train_QA_huggingface
 
This code trains and evaluates a Question Answering (QA) model using the transformers library on a Persian dataset.


 
## Requirements 
 
- pandas
- transformers['torch]
- datasets

## Installation 
 
Make sure to install all the required libraries mentioned in the code. 
 
## How to use 
 
To use the code, just set the parameters as required.
after that code will doing this steps:

- Read the train and test data using the read_qa() function provided in utils.read_ds module.
- Create a dataset dictionary and convert the data to datasets.
- Preprocess the data and tokenize.
- Load the model weights using AutoModelForQuestionAnswering from transformers library.
- Set the training arguments using TrainingArguments from transformers library.
- Train the model using Trainer from transformers library.
- Evaluate the model.
The code has been written in a way that the parameters and the functions used for data preprocessing and tokenization can be customized.
