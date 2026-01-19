from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

token = os.getenv("HF_TOKEN")

model_name = "doubleyyh/exit-gemma-2b"

quantization_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_compute_dtype=torch.float16,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  quantization_config=quantization_config,
  device_map="auto",
  token=token
)

# Test classification format
query = "How do SSDs work?"
sentence = "SSDs use flash memory."
document = "SSDs use flash memory. They are faster than HDDs. The weather is nice."

prompt = f"Query: {query}\nDocument: {document}\nSentence: {sentence}\nRelevant?"

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
  outputs = model(**inputs)
  logits = outputs.logits[0, -1]

# Get Yes/No probabilities
yes_token = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_token = tokenizer.encode("No", add_special_tokens=False)[0]

probs = torch.softmax(logits[[yes_token, no_token]], dim=0)
yes_prob = probs[0].item()

print(f"Query: {query}")
print(f"Sentence: {sentence}")
print(f"Yes probability: {yes_prob:.3f}")
print(f"Decision: {'KEEP' if yes_prob > 0.5 else 'DISCARD'}")
