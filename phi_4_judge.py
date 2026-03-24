import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Phi4Judge:
    def __init__(self):
        model_name = "microsoft/Phi-4-mini-instruct"
        print(f"Loading {model_name} as judge...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def score(self, prompt: str, response: str) -> float:
        judge_prompt = (
            "You are an impartial judge evaluating the quality and helpfulness of an AI assistant's response. "
            "Rate the response from 1.0 to 10.0, where 10.0 is an excellent and helpful response. "
            "Please respond with strictly just the number.\n\n"
            f"User: {prompt}\n\nAssistant: {response}"
        )
        messages = [{"role": "user", "content": judge_prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
        output_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", output_text)
        if matches:
            return float(matches[0])
        return 0.0

    def score_batch(self, prompts: list[str], responses: list[str], batch_size: int = 8) -> list[float]:
        all_scores = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_responses = responses[i:i+batch_size]
            
            judge_prompts = []
            for p, r in zip(batch_prompts, batch_responses):
                judge_prompt = (
                    "You are an impartial judge evaluating the quality and helpfulness of an AI assistant's response. "
                    "Rate the response from 1.0 to 10.0, where 10.0 is an excellent and helpful response. "
                    "Please respond with strictly just the number.\n\n"
                    f"User: {p}\n\nAssistant: {r}"
                )
                messages = [{"role": "user", "content": judge_prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                judge_prompts.append(formatted_prompt)
                
            inputs = self.tokenizer(judge_prompts, return_tensors="pt", padding=True).to(self.model.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
                
            output_texts = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            for text in output_texts:
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", text.strip())
                if matches:
                    all_scores.append(float(matches[0]))
                else:
                    all_scores.append(0.0)
                    
        return all_scores
