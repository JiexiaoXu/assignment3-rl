import torch
import torch.nn as nn

from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.optimizer import AdamW
from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.training_util import load_checkpoint
import json
from types import SimpleNamespace


def generate_text(
    prompt: str,
    model: TransformerLM,
    tokenizer: BPETokenizer,
    context_length: int,
    max_gen_length: int,
    temperature: float,
    top_p_threshold: float,
) -> str:
    def temperature_softmax(logits, temperature):
        logits = logits / max(temperature, 1e-8)
        return torch.softmax(logits, dim=-1)

    def top_p_filtering(softmax_logits, top_p=top_p_threshold):
        sorted_logits, sorted_indices = torch.sort(softmax_logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        softmax_logits[indices_to_remove] = 0.0

        # Renormalize the probabilities
        softmax_logits = softmax_logits / softmax_logits.sum(dim=-1, keepdim=True)
        return softmax_logits

    model.eval()
    with torch.no_grad():
        # Tokenize the prompt
        accum_output = tokenizer.encode(prompt)
        accum_output = torch.tensor(accum_output, dtype=torch.long, device="cuda")
        eos_token_id = tokenizer.encode("<|endoftext|>")[0]

        # Generate tokens
        while True:
            if len(accum_output) >= max_gen_length:
                break

            input_tensor = accum_output[-context_length:].unsqueeze(0)
            logits = model(input_tensor)

            # Only take last token's logits
            last_token_logits = logits[0, -1, :]
            softmax_logits = temperature_softmax(last_token_logits, temperature)
            softmax_logits = top_p_filtering(softmax_logits, top_p=top_p_threshold)

            # sample from last token's distribution, stop if EOS token is generated
            sampled_token = torch.multinomial(softmax_logits, num_samples=1)
            if sampled_token.item() == eos_token_id:
                break

            accum_output = torch.cat([accum_output, sampled_token], dim=0)

        # Decode the generated text
        generated_text = tokenizer.decode(accum_output.tolist())
    return generated_text


if __name__ == "__main__":

    cfg_path = "/homes/iws/jiexiao/cse599o/hw1/config.json"
    with open(cfg_path, "r") as f:
        config_dict = json.load(f)
    config = SimpleNamespace(**config_dict)

    import tiktoken

    encoding = tiktoken.get_encoding("gpt2")
    vocab = {i: encoding.decode_single_token_bytes(i) for i in range(encoding.n_vocab)}
    merges = list(encoding._mergeable_ranks.items())
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer(vocab, merges, special_tokens)

    checkpoint_path = (
        "/local1/jiexiao/checkpoint/checkpoint.pth"
    )
    model = TransformerLM(
        vocab_size=encoding.n_vocab,
        d_model=config.model_params["d_model"],
        num_heads=config.model_params["num_heads"],
        num_layers=config.model_params["num_layers"],
        dff=config.model_params["dff"],
        context_length=config.model_params["context_length"],
        theta=config.model_params["theta"],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer_params["learning_rate"],
        weight_decay=config.optimizer_params["weight_decay"],
    )

    load_checkpoint(checkpoint_path, model, optimizer)

    result_str = generate_text(
        prompt="Write a story includes bags and ",
        model=model,
        tokenizer=tokenizer,  # load tokenizer
        context_length=config.model_params["context_length"],
        max_gen_length=512,
        temperature=1.0,
        top_p_threshold=0.9,
    )
    print("Generated text:")
    print(result_str)
