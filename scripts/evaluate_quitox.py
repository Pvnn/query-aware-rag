import sys
import os
import torch
import warnings
from tqdm import tqdm
from datasets import load_dataset
import nltk

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Filter warnings
warnings.filterwarnings("ignore")

from src.compression.quitox_filter import QuitoxCoarseFilter

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')

def extract_sentences_from_schema(sample):
  """
  Parses JSON and splits text chunks into individual sentences.
  Returns:
    all_sentences: List[str] (The full document context)
    gold_sentences: Set[str] (The unique sentences that are 'answers')
  """
  all_sentences = []
  gold_sentences = set()
  
  def process_text(text, is_gold):
    # Split text into sentences to match the filter's operating granularity
    sents = nltk.sent_tokenize(text)
    for s in sents:
      clean_s = s.strip()
      # Ignore tiny artifacts like "." or " "
      if len(clean_s) > 5: 
        all_sentences.append(clean_s)
        if is_gold:
          gold_sentences.add(clean_s)

  if 'positive_contexts' not in sample:
    raise KeyError("Sample missing 'positive_contexts'")

  # Process Gold
  for ctx in sample['positive_contexts']:
    process_text(ctx['text'], is_gold=True)
      
  # Process Distractors
  for ctx in sample['negative_contexts']:
    process_text(ctx['text'], is_gold=False)
      
  return all_sentences, gold_sentences

def evaluate_quitox(data_path, num_samples=50, ratio=0.5):
  print(f"📂 Loading dataset from {data_path}...")
  
  try:
      dataset_dict = load_dataset("json", data_files=data_path)
      dataset = dataset_dict['train']
  except Exception as e:
      print(f"❌ Error loading JSON: {e}")
      return

  # Initialize Filter
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"🚀 Initializing QUITO-X on {device.upper()}...")
  filtr = QuitoxCoarseFilter(device=device)
  
  total_gold_unique = 0
  kept_gold_unique = 0
  total_input = 0
  total_output = 0
  
  print(f"\n📊 Starting Evaluation on {num_samples} samples (Target Ratio={ratio})...")
  
  limit = min(num_samples, len(dataset))
  
  for i in tqdm(range(limit)):
    sample = dataset[i]
    query = sample['question']
    
    try:
      # 1. Prepare Data
      all_sents, gold_set = extract_sentences_from_schema(sample)
      
      if not all_sents: 
          continue
          
      # 2. Run QUITO-X Filter
      filtered_sents = filtr.compress(
        query=query, 
        sentences=all_sents, 
        compression_ratio=ratio
      )
      
      # 3. Calculate Stats (Using Sets to avoid >100% bug)
      filtered_set = set(filtered_sents)
      
      # Intersection: How many UNIQUE gold sentences are in the filtered output?
      found_gold = gold_set.intersection(filtered_set)
      
      total_gold_unique += len(gold_set)
      kept_gold_unique += len(found_gold)
      
      total_input += len(all_sents)
      total_output += len(filtered_sents)
        
    except KeyError as e:
      continue

  if total_gold_unique == 0:
    print("No gold sentences found in data samples!")
    return

  recall = (kept_gold_unique / total_gold_unique) * 100
  compression = (1 - (total_output / total_input)) * 100 if total_input > 0 else 0
  
  print("\n" + "="*50)
  print(f"QUITO-X PERFORMANCE REPORT (n={limit})")
  print("="*50)
  print(f"Recall (Gold Retention):    {recall:.2f}%")
  print(f"Actual Compression:         {compression:.2f}%")
  print(f"Avg Input Context Size:     {total_input/limit:.1f} sentences")
  print(f"Avg Output Context Size:    {total_output/limit:.1f} sentences")
  print("="*50)

if __name__ == "__main__":
  JSON_PATH = os.path.join('data', 'processed', 'dev.json') 
  evaluate_quitox(JSON_PATH, num_samples=20, ratio=0.6)