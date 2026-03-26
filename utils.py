import os
import random
import numpy as np
from typing import Any, Dict, Optional, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

SAFE = 0 
UNSAFE = 1

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_rng_state() -> Dict[str, Any]:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return state

def set_rng_state(state: Optional[Dict[str, Any]]):
    if not state: return
    if "python_random" in state: random.setstate(state["python_random"])
    if "numpy_random" in state: np.random.set_state(state["numpy_random"])
    if "torch_random" in state: torch.set_rng_state(state["torch_random"])
    if "torch_cuda_random" in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["torch_cuda_random"])

def login_to_hugging_face():
    from huggingface_hub import login

    try:
        # if using google colab you can add your token as a secret
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception as e:
        print("FAILED to import google colab so you must not be in it. That's ok! Using HF_TOKEN env variable")
        print(e)
        # if not then add it as an environment variable
        token = os.getenv('HF_TOKEN')

    # You need to authenticate with HuggingFace to use this model. The approval process took a few minutes for me.
    login(token)

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def project_onto_plane(
    weight: torch.Tensor, 
    bias: float, 
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    unsafe_threshold: float, 
    original_activation: torch.Tensor
) -> Tuple[torch.Tensor, bool]:
    assert len(weight) == len(original_activation)
    assert weight.ndim == 1 and original_activation.ndim == 1
    assert weight.is_floating_point() and original_activation.is_floating_point()

    assert 0 < unsafe_threshold and unsafe_threshold < 1

    # scale cuz data is normalized for linear probe
    scaled_activation = (original_activation - scaler_mean) / scaler_scale

    logits = torch.dot(weight, scaled_activation) + bias
    if torch.sigmoid(logits) > unsafe_threshold:

        target_logit = np.log((1.0 / unsafe_threshold) - 1.0)
        weight_sq_norm = torch.dot(weight, weight)

        # again scale
        scaling = (logits + target_logit) / weight_sq_norm
        projected_scaled = scaled_activation - weight * scaling

        # scale the projected activation back to the original space
        projected_original = (projected_scaled * scaler_scale) + scaler_mean
        return projected_original, True
    else:
        return original_activation, False
    
def get_inputs(
        prompt: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    return inputs

def generate_no_steer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_output: int,        
    prompt: str,
) -> str:
    inputs = get_inputs(prompt=prompt, model=model, tokenizer=tokenizer)
    with torch.inference_mode():
        orig_ids = model.generate(**inputs, max_new_tokens=max_output, do_sample=False)
    input_len = inputs.input_ids.shape[1]
    orig_text = tokenizer.decode(orig_ids[0][input_len:], skip_special_tokens=True)
    return orig_text

def safety_steer(
        prompt: str, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase, 
        layer_no: int,
        weight: np.ndarray, 
        bias: float,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        unsafe_threshold: float,
        max_steer_tokens: int,
        max_output: int,
    ) -> Tuple[str, bool]:
    inputs = get_inputs(prompt=prompt, model=model, tokenizer=tokenizer)
    # convert from np array to torch tensor
    weight_t = torch.tensor(weight, dtype=torch.float32, device=model.device)
    scaler_mean_t = torch.tensor(scaler_mean, dtype=torch.float32, device=model.device)
    scaler_scale_t = torch.tensor(scaler_scale, dtype=torch.float32, device=model.device)

    steer_count = [0]
    actual_steers = [0]
    def steering_hook(module, args, output):
        if steer_count[0] >= max_steer_tokens:
            return output
            
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Extract the current activation for the last token
        current_activation = hidden_states[0, -1, :].detach().to(torch.float32)
        
        # get projection
        projected_tensor, was_steered = project_onto_plane(
            weight=weight_t, 
            bias=bias, 
            scaler_mean=scaler_mean_t,
            scaler_scale=scaler_scale_t,
            unsafe_threshold=unsafe_threshold, 
            original_activation=current_activation
        )
        projected_tensor = projected_tensor.to(model.dtype)
        
        # clone the hidden states to safely mutate without breaking KV cache references
        hidden_states = hidden_states.clone()
        hidden_states[0, -1, :] = projected_tensor
        steer_count[0] += 1
        if was_steered:
            actual_steers[0] += 1
        
        return (hidden_states,) + output[1:] if isinstance(output, tuple) else hidden_states

    handle = model.model.layers[layer_no].register_forward_hook(steering_hook)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_output, do_sample=False)
    handle.remove()
    
    input_len = inputs.input_ids.shape[1]
    response = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    return response, actual_steers[0] > 0
