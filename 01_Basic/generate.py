# greedy / sampling generation implementation
from typing import List, Optional, Union
import torch
from mini_vllm.model import Model
from mini_vllm.tokenizer import Tokenizer

class Generator:
    def __init__(self, model_id: str, bnb_config=None):
        self.model = Model(model_id, bnb_config)
        self.tokenizer = Tokenizer(model_id)

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        pad_token_id: Optional[int] = None
    ) -> List[str]:
        generated_texts = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt).to(self.model.model.device)
            generated_ids = input_ids

            for _ in range(max_new_tokens):
                outputs = self.model.forward(generated_ids)
                logits = outputs.logits[:, -1, :] / temperature

                # Apply top-p sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')

                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

                generated_ids = torch.cat((generated_ids, next_token_id), dim=-1)

                if pad_token_id is not None and next_token_id.item() == pad_token_id:
                    break

            generated_text = self.tokenizer.decode(generated_ids[0])
            generated_texts.append(generated_text)

        return generated_texts
