from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from bucket_utils import resume_checkpoint_if_exists, upload_checkpoint
from util_types import DataSplit, ModelName, PoolingStrategy
from shield_gemma_reward_model import GemmaRewardModel
from utils import SAFE, UNSAFE, get_rng_state, login_to_hugging_face, set_rng_state, set_seed

import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True

login_to_hugging_face()

def handle_batch(
        batch: Dict[str, Any],
        idx: int, 
        batch_count: int, 
        start_checkpoint_idx: int, 
        target_layers: List[int],
        pool_type: PoolingStrategy, 
        model_name: ModelName, 
        split: DataSplit, 
        gemma_reward_model: GemmaRewardModel,
        model: PreTrainedModel,
        activations: Dict[int, torch.Tensor],
        tokenizer: PreTrainedTokenizerBase,
        max_len_input: int,
        max_len_output: int,
        x_arr_dict: dict[int, np.memmap], 
        y_arr: np.memmap,
        disable_hooks: List[bool],
        checkpoint_period: int,
        prefix: str,
    ) -> Tuple[int, int]:
    # This isn't a constant if the length of the dataset isn't a multiple of our chosen
    # batch size and we are on the last batch. Or if we changed how we selected batches or did streaming etc.
    current_batch_size = batch["input_ids"].size(0)
    
    # If this entire batch was already processed, skip it entirely.
    if idx + current_batch_size <= start_checkpoint_idx:
        idx += current_batch_size
        batch_count += 1
        return idx, batch_count
        
    # If the batch partially overlaps with the start_checkpoint_idx, slice the data
    if idx < start_checkpoint_idx:
        offset = start_checkpoint_idx - idx
        for key in batch:
            batch[key] = batch[key][offset:]
        
        # Update the variables after slicing
        current_batch_size = batch["input_ids"].size(0)
        idx = start_checkpoint_idx
        
    prompts = batch["prompt"]
    original_prompts = batch.get("original_prompt", prompts)
    with torch.inference_mode():
        input_ids = batch["input_ids"].to(model.device, non_blocking=True)
        attn      = batch["attention_mask"].to(model.device, non_blocking=True)
        
        disable_hooks[0] = False
        _ = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=False)

    # Safely find the exact index of the last prompt token regardless of left or right padding
    seq_lens = (attn.cumsum(dim=1) * attn).argmax(dim=1).cpu()

    # --- SANITY CHECK ---
    if tokenizer.pad_token_id is not None:
        extracted_token_ids = input_ids[torch.arange(current_batch_size), seq_lens]
        if (extracted_token_ids == tokenizer.pad_token_id).any():
            raise RuntimeError("Sanity check failed: Extracted a padding token instead of the final prompt token!")
            
    if batch_count == 0:
        sample_token_id = input_ids[0, seq_lens[0]].item()
        print(f"Sanity Check (Batch 0): Final prompt token is {tokenizer.decode([sample_token_id])!r} (ID: {sample_token_id})", flush=True)

    # === save the activations of the layers into x ===
    for layer_no in target_layers:
        h_gpu = activations[layer_no]
        out   = h_gpu.cpu().to(torch.float32)
        activations[layer_no] = None # trick to free up GPU memory

        # An LLM processes a sequence of tokens, producing a corresponding sequence of 
        # hidden state activations (one vector per token). Because input prompts vary in 
        # length, we pool these token-level activations (e.g., taking the mean, max, or 
        # just the final token) to aggregate them into a single, fixed-size vector.
        if pool_type == "finaltoken":
            vecs = out[torch.arange(current_batch_size), seq_lens]
        elif pool_type == "mean":
            mask = attn.unsqueeze(-1).cpu()
            vecs = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        elif pool_type == "max":
            lo  = out[:,1:,:]
            m   = attn[:,1:].unsqueeze(-1).cpu()
            vecs= (lo * m).max(1).values
        else:
            raise ValueError(f"Unknown POOL_TYPE {pool_type}")

        # --- SANITY CHECK ---
        # Ensure that different samples in the batch have different activations
        if batch_count == 0 and current_batch_size > 1:
            if torch.allclose(vecs[0], vecs[1], atol=1e-5):
                raise RuntimeError(f"Sanity check failed: Activations for sample 0 and 1 are identical at layer {layer_no}! Check padding or extraction logic.")

        # save in x
        x_arr_dict[layer_no][idx:idx+current_batch_size] = vecs.numpy()
    
    # === save the safety labels into y ===
    tokenizer.padding_side = 'left'
    gen_inputs = tokenizer(prompts, padding=True, truncation=True, max_length=max_len_input, return_tensors="pt").to(model.device)
    tokenizer.padding_side = 'right'
    
    input_len = gen_inputs.input_ids.shape[1]
    print(f"Batch {batch_count} (idx {idx}) max generation input length: {input_len} tokens", flush=True)
    
    # generate the response from the inputs
    disable_hooks[0] = True
    with torch.inference_mode():
        # NOTE: We use deterministic greedy sampling (do_sample=False) here to establish a baseline.
        # This data is being used to train a model to predict whether the response will be safe 
        # from the activations.
        # In a real production system or more rigorous evaluation, you might set do_sample=True,
        # sample M responses, and take the least safe response to better evaluate 
        # worst-case activation correlations.
        generated_outputs = model.generate(
            **gen_inputs,
            max_new_tokens=max_len_output,
            do_sample=False, # deterministic greedy sampling here
            pad_token_id=tokenizer.pad_token_id
        )
    
    responses_ids = generated_outputs[:, input_len:]
    responses = tokenizer.batch_decode(responses_ids, skip_special_tokens=True)
    
    # use ShieldGemma model to determine if response is safe
    safe_labels = []
    for prompt, response in zip(original_prompts, responses):
        is_safe = gemma_reward_model.is_safe(prompt, response)
        safe_labels.append(SAFE if is_safe else UNSAFE)

    # save labels in y
    y_arr[idx:idx+current_batch_size] = np.array(safe_labels, dtype=np.int64)
    
    # print out the cumulative % of safe vs unsafe responses
    total_processed = idx + current_batch_size
    total_unsafe = int(np.sum(y_arr[:total_processed]))
    total_safe = total_processed - total_unsafe
    pct_safe = (total_safe / total_processed) * 100
    print(f"Batch {batch_count} (idx {idx}) cumulative safety: {pct_safe:.1f}% safe, {100 - pct_safe:.1f}% unsafe", flush=True)

    # checkpoint to save progress in case we crash
    if batch_count % checkpoint_period == 0:

        # flush values to disk before checkpointing them 
        # because checkpoint function reads from files
        for arr in (*x_arr_dict.values(), y_arr):
            arr.flush()

        upload_checkpoint(
            split=split, 
            idx=idx, 
            target_layers=target_layers, 
            pool_type=pool_type, 
            model_name=model_name,
            rng_state=get_rng_state(),
            prefix=prefix,
        )
        
    return idx + current_batch_size, batch_count + 1

def save_all_layers_one_pass(
        loader: torch.utils.data.DataLoader,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        d: int,
        split: DataSplit, 
        gemma_reward_model: GemmaRewardModel, 
        model_name: ModelName, 
        target_layers: List[int], 
        pool_type: PoolingStrategy, 
        max_len_input: int,
        max_len_output: int,
        checkpoint_period: int,
        prefix: str,
    ):
    n = len(loader.dataset)
    
    start_checkpoint_idx, x_arr_dict, y_arr, rng_state = resume_checkpoint_if_exists(
        split=split, 
        target_layers=target_layers, 
        pool_type=pool_type, 
        model_name=model_name, 
        n=n, 
        d=d,
        prefix=prefix,
    ) 
    # Restore the generator states right before processing starts
    set_rng_state(rng_state)

    # this sets up the infrastructure to capture the activations of the model
    activations = {layer_no: None for layer_no in target_layers}
    handles: List[torch.utils.hooks.RemovableHandle] = []
    disable_hooks: List[bool] = [False] # list python trick to get pointer

    for layer_no in target_layers:
        def make_hook(idx):
            def hook(module, inp, out):
                if disable_hooks[0]:
                    return
                h = out[0] if isinstance(out, tuple) else out
                activations[idx] = h.detach()
            return hook
        handles.append(model.model.layers[layer_no].register_forward_hook(make_hook(layer_no)))

    try:
        idx: int = 0
        batch_count: int = 0
        
        for batch in tqdm(loader, desc=f"{split}-all-layers", total=len(loader)):
            idx, batch_count = handle_batch(
                batch=batch, 
                idx=idx, 
                batch_count=batch_count, 
                start_checkpoint_idx=start_checkpoint_idx,
                target_layers=target_layers,
                pool_type=pool_type,
                model_name=model_name,
                split=split,
                gemma_reward_model=gemma_reward_model,
                model=model,
                x_arr_dict=x_arr_dict,
                y_arr=y_arr,
                activations=activations,
                tokenizer=tokenizer,
                max_len_input=max_len_input,
                max_len_output=max_len_output,
                disable_hooks=disable_hooks, # this gets mutated across iterations
                checkpoint_period=checkpoint_period,
                prefix=prefix,
            )

    finally:
        for h in handles:
            h.remove()

        # flush values to disk before checkpointing them 
        # because checkpoint function reads from files
        for arr in (*x_arr_dict.values(), y_arr):
            arr.flush()
        upload_checkpoint(
            split=split, 
            idx=idx, 
            target_layers=target_layers, 
            pool_type=pool_type, 
            model_name=model_name,
            rng_state=get_rng_state(),
            prefix=prefix
        )


def extract_activations(
    model_name: ModelName,
    target_layers: List[int],
    pool_type: PoolingStrategy,
    batch_size: int,
    max_len_input: int,
    max_len_output: int,
    checkpoint_period: int,
    dataset: str,
    seed: int,
    selected_indices: Optional[List[int]],
    train_cutoff: int,
    prefix:str,
):
    set_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(batch):
        # Use the model's built-in chat template instead of hardcoded strings
        messages_list = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p}
            ] for p in batch["prompt"]
        ]
        formatted_prompts = tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
        
        enc = tokenizer(
            formatted_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_len_input,
            return_tensors="pt"
        )
        return {
            "original_prompt": batch["prompt"],
            "prompt":         formatted_prompts,
            "input_ids":      enc.input_ids,
            "attention_mask": enc.attention_mask,
        }

    def prepare_dataset(ds):
        ds = ds.map(preprocess, batched=True, batch_size=256)
        return ds.with_format(
            type="torch",
            columns=["input_ids", "attention_mask", "prompt", "original_prompt"]
        )

    ds_dict = load_dataset(dataset)

    if 'test' in ds_dict:
        train_raw = ds_dict['train']
        test_raw = ds_dict['test']
        if selected_indices:
            train_raw = train_raw.select(selected_indices)
    else:
        print(f"No 'test' split found for {dataset}. Creating an 80/20 train/test split automatically.")
        train_raw = ds_dict['train']
        if selected_indices:
            train_raw = train_raw.select(selected_indices)
        splits = train_raw.train_test_split(test_size=0.2, seed=seed)
        train_raw = splits['train']
        test_raw = splits['test']

    train_ds = prepare_dataset(train_raw)
    test_ds  = prepare_dataset(test_raw)

    train_ds = train_ds.select(range(min(train_cutoff, len(train_ds))))
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )

    # hidden dimension
    d = model.config.hidden_size
    
    model = torch.compile(model, mode="reduce-overhead")

    gemma_reward_model = GemmaRewardModel()

    save_all_layers_one_pass(
        loader=test_loader, 
        model=model, 
        tokenizer=tokenizer,  
        d=d, 
        split="test", 
        gemma_reward_model=gemma_reward_model, 
        model_name=model_name, 
        target_layers=target_layers,
        pool_type=pool_type,
        max_len_input=max_len_input,
        max_len_output=max_len_output,
        checkpoint_period=checkpoint_period,
        prefix=prefix,
    )

    save_all_layers_one_pass(
        loader=train_loader, 
        model=model, 
        tokenizer=tokenizer, 
        d=d, 
        split="train", 
        gemma_reward_model=gemma_reward_model, 
        model_name=model_name, 
        target_layers=target_layers,
        pool_type=pool_type,
        max_len_input=max_len_input,
        max_len_output=max_len_output,
        checkpoint_period=checkpoint_period,
        prefix=prefix,
    )
