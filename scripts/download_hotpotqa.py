from datasets import load_dataset

# Download validation (dev) set
hotpotqa_dev = load_dataset("hotpot_qa", "fullwiki", split="validation")
hotpotqa_dev.save_to_disk("data/hotpotqa/dev")

# Download training set
hotpotqa_train = load_dataset("hotpot_qa", "fullwiki", split="train")
hotpotqa_train.save_to_disk("data/hotpotqa/train")

print("Dev samples:", len(hotpotqa_dev))
print("Train samples:", len(hotpotqa_train))
