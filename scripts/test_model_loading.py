from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

token = os.getenv("HF_TOKEN")

def test_exit_model_loading():
  # Check CUDA availability first
  print("CUDA Diagnostics")
  print(f"PyTorch version: {torch.__version__}")
  print(f"CUDA available: {torch.cuda.is_available()}")
  
  if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
  else:
    print("❌ CUDA not available!")
    print("Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    return
  
  model_name = "doubleyyh/exit-gemma-2b"
  
  print("\n" + "="*60)
  print("Testing EXIT Model Loading on GPU")
  print("="*60)
  
  print("\n[1/4] Loading tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2b-it",
    token=token
  )
  print("✓ Tokenizer loaded successfully")
  
  print("\n[2/4] Loading model with 4-bit quantization...")
  
  # Proper quantization configuration for GPU
  quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
  )
  
  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",  # Automatically use GPU
    token=token
  )
  print("✓ Model loaded successfully")
  
  print("\n[3/4] Checking GPU memory...")
  print(f"Device: {model.device}")
  print(f"Model dtype: {model.dtype}")
  
  if torch.cuda.is_available():
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
  
  print("\n[4/4] Testing inference...")
  test_prompt = "Query: What is AI? Sentence: AI is artificial intelligence. Relevant?"
  
  # Move inputs to same device as model
  inputs = tokenizer(test_prompt, return_tensors="pt")
  inputs = {k: v.to(model.device) for k, v in inputs.items()}
  
  print(f"Input device: {inputs['input_ids'].device}")
  
  with torch.no_grad():
    outputs = model.generate(
      **inputs, 
      max_new_tokens=10,
      do_sample=False,
      pad_token_id=tokenizer.eos_token_id
    )
  
  response = tokenizer.decode(outputs[0], skip_special_tokens=True)
  
  print(f"\nTest prompt: {test_prompt}")
  print(f"Model response: {response}")
  
  # Extract generated text only
  generated = response[len(test_prompt):].strip()
  print(f"Generated text: {generated}")
  
  print("\n" + "="*60)
  print("All tests passed! ✓")
  print("="*60)

if __name__ == "__main__":
  test_exit_model_loading()
