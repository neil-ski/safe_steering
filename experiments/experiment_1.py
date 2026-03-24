from collections import defaultdict

from datasets import load_dataset
from shield_gemma_reward_model import GemmaRewardModel
from utils import set_seed

# We want to ensure that our reward model works for the dataset we are using for training.
# This datasets comes with "chosen" and "rejected" responses so we can test that our reward model
# correctly marks the "chosen" response as safe and the "rejected" response as unsafe.
# For any samples that it does not, we can filter out.

dataset = load_dataset("LLM-LAT/harmful-dataset", split="train")

reward_model = GemmaRewardModel()

set_seed(0)

anomalies = defaultdict(list)
for row in dataset:
    prompt = row["prompt"]
    rejected_response = row["rejected"]
    chosen_response = row["chosen"]
    
    rejected_score = reward_model.score_with_str(prompt, rejected_response)
    chosen_score = reward_model.score_with_str(prompt, chosen_response)
    
    if rejected_score >= 0 or chosen_score <= 0:
        anomalies["prompt"].append(prompt)
        anomalies["rejected"].append(rejected_response)
        anomalies["chosen"].append(chosen_response)
        anomalies["rejected"].append(rejected_response)
        anomalies["chosen"].append(chosen_response)


print(anomalies)
# TODO it would be good to write this to disk 
# I put this in filtered_indices.py
print(set(range(len(dataset))) - set(anomalies[""]))
