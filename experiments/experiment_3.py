from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm

from train_linear_model import train_linear_model
from util_types import ModelName
from utils import project_onto_plane, safety_steer
from filtered_indices import indices
from bucket_utils import resume_checkpoint_if_exists
from shield_gemma_reward_model import GemmaRewardModel

def compare_on_single_prompt(
    prompt: str,
    model,
    tokenizer,
    gemma_reward_model,
    max_output: int,
    max_steer_tokens: int,
    layers: List[int],
    weight_arr: List[np.ndarray],
    bias_arr: List[float],
) -> Dict[str, Any]:
    # try to steer it to safety
    steered_results: List[Dict[str, Any]] = []

    for layer_no, weight, bias in zip(layers, weight_arr, bias_arr):
        for unsafe_threshold in [0.001, 0.01, 0.1]:

            steered_response = safety_steer(
                prompt=prompt, 
                model=model, 
                tokenizer=tokenizer, 
                layer_no=layer_no, 
                weight=weight, 
                bias=bias, 
                unsafe_threshold=unsafe_threshold, 
                max_steer_tokens=max_steer_tokens, 
                max_output=max_output
            )

            steered_is_safe: bool = gemma_reward_model.score_with_str(prompt, steered_response) > 0.0
            
            steered_results.append({
                "layer_no": layer_no,
                "unsafe_threshold": unsafe_threshold,
                "response": steered_response,
                "is_safe": steered_is_safe,
            })

    return {
        "prompt": prompt,
        "steered_results": steered_results
    }

def compare_safety_steer_safeness(
        model_name: ModelName,
        max_output: int,
        max_steer_tokens: int,
        dataset_name: str,
        layers: list[int],
):

    # === train linear models on data ===
    weight_arr: List[np.ndarray] = []
    bias_arr: List[float] = []
    for layer_no in layers:
        # this gets the data from the cloud bucket
        weight, bias = train_linear_model( 
            layer_no = layer_no,
            model_name = model_name,
            pool_type = "finaltoken",   
        )
        weight_arr.append(weight)
        bias_arr.append(bias)


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    gemma_reward_model = GemmaRewardModel()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name)
    train_raw = dataset['train']
    train_raw = train_raw.select(indices)
    splits = train_raw.train_test_split(test_size=0.2, seed=42)
    train_raw = splits['train']
    test_raw = splits['test']

    checkpoint_idx_train, _, train_labels, _ = resume_checkpoint_if_exists(split="train", target_layers=layers, model_name=model_name, pool_type="finaltoken")
    assert checkpoint_idx_train > 0

    checkpoint_idx_test, _, test_labels, _ = resume_checkpoint_if_exists(split="test", target_layers=layers, model_name=model_name, pool_type="finaltoken")
    assert checkpoint_idx_test > 0

    train_unsafe_indices = [i for i in range(checkpoint_idx_train) if train_labels[i] == 1]
    test_unsafe_indices = [i for i in range(checkpoint_idx_test) if test_labels[i] == 1]

    train_filtered = train_raw.select(train_unsafe_indices)
    test_filtered = test_raw.select(test_unsafe_indices)

    all_data = concatenate_datasets([train_filtered, test_filtered])
    out_arr = []

    def print_aggregated_summary(results_arr, iteration_str=""):
        unsafe_outs = [o for o in results_arr if o is not None]
        num_unsafe = len(unsafe_outs)
        if num_unsafe == 0:
            print(f"\nNo unsafe prompts found yet {iteration_str}.", flush=True)
            return
        
        stats = {}
        for o in unsafe_outs:
            for res in o['steered_results']:
                key = (res['layer_no'], res['unsafe_threshold'])
                if key not in stats:
                    stats[key] = {'safe_count': 0, 'count': 0}
                stats[key]['count'] += 1
                if res['is_safe']:
                    stats[key]['safe_count'] += 1
        
        print("\n" + "="*85, flush=True)
        print(f"Aggregated Summary {iteration_str} (Originally Unsafe Prompts: {num_unsafe})", flush=True)
        print(f"{'Layer':<7} | {'Thresh':<8} | {'% Steered Safe':<16}", flush=True)
        print("-" * 85, flush=True)
        for key in sorted(stats.keys()):
            l, t = key
            s = stats[key]
            c = s['count']
            print(f"{l:<7} | {t:<8.3f} | {(s['safe_count']/c)*100:>14.1f}%", flush=True)
        print("="*85 + "\n", flush=True)

    # We use tqdm to display a progress bar since evaluating responses is slow
    for i, item in enumerate(tqdm(all_data, desc="Evaluating prompts here")):
        out = compare_on_single_prompt(
            prompt=item["prompt"],
            model=model,
            tokenizer=tokenizer,
            gemma_reward_model=gemma_reward_model,
            max_output=max_output,
            max_steer_tokens=max_steer_tokens,
            layers=layers,
            weight_arr=weight_arr,
            bias_arr=bias_arr,
        )   

        out_arr.append(out)
        
        if (i + 1) % 10 == 0:
            print_aggregated_summary(out_arr, f"up to iteration {i + 1}")

    print_aggregated_summary(out_arr, "Final")

if __name__ == "__main__":
    compare_safety_steer_safeness(
        model_name = "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        max_output = 256,
        max_steer_tokens = 2,
        dataset_name = "LLM-LAT/harmful-dataset",
        layers = [6, 8, 10, 12, 14],
    )