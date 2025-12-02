from transformers import (T5ForConditionalGeneration, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Trainer)
from datasets import load_from_disk
import numpy as np
import evaluate
import torch

# load the trained model
model_path = "./t5_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# load the tokenized dataset 
dataset = load_from_disk("./t5_tokenized")
test_data = dataset["test"]

rouge = evaluate.load("rouge")

def postprocess_text(preds, labels):
    # decode the token ids to strings
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # strip whitespace
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    return preds, labels

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_preds = []
all_labels = []

batch_size = 8  # adjust based on your GPU/CPU memory
for i in range(0, len(test_data), batch_size):
    batch = test_data[i : i + batch_size]
    print(f"Processing batch {i // batch_size + 1} / {len(test_data) // batch_size + 1}")
    
    # Convert batch to tensors
    input_ids = torch.tensor(batch["input_ids"]).to(device)
    attention_mask = torch.tensor(batch["attention_mask"]).to(device)
    labels = torch.tensor(batch["labels"]).to(device)
    
    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,  # same as your target length
            num_beams=2,     
            early_stopping=True
        )
    
    # Store predictions and labels
    all_preds.extend(outputs.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Decode predictions and references
decoded_preds, decoded_labels = postprocess_text(all_preds, all_labels)

# Compute ROUGE scores
result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
result = {key: value * 100 for key, value in result.items()}

# Print ROUGE scores
print("ROUGE scores on the test set:")
for k, v in result.items():
    print(f"{k}: {v:.2f}")

    print(f"Processed batch {i // batch_size + 1} / {len(test_data) // batch_size + 1}")

"""# set up the data collator, training arguments, and trainer for evaluation on the test set
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
    label_pad_token_id=tokenizer.pad_token_id
)

eval_args = Seq2SeqTrainingArguments(
    output_dir="./t5_evaluation",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=1,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=eval_args,
    data_collator=data_collator
)

print("\n===== Running prediction on the test set =====\n")
prediction_output = trainer.predict(test_data)

prediction_ids = prediction_output.predictions
label_ids = prediction_output.label_ids

decoded_predictions = tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

rouge = evaluate.load("rouge")

# Strip whitespace
decoded_preds = [pred.strip() for pred in decoded_preds]
decoded_labels = [label.strip() for label in decoded_labels]

rouge_results = rouge.compute(
    predictions=decoded_preds,
    references=decoded_labels,
    use_stemmer=True
)

print("\n===== ROUGE Scores =====\n")
for key in rouge_results:
    score = rouge_results[key].mid.fmeasure
    print(f"{key}: {score:.4f}")

print("\n===== SAMPLE MODEL OUTPUTS =====\n")

for i in range(5):
    print(f"--- Example {i+1} ---")
    print("Prediction:\n", decoded_preds[i])
    print("Reference:\n", decoded_labels[i])
    print("----------------------\n")"""