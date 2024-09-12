import torch

from puli import load_model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

puli = load_model("puli2-gpt", device)

prompt = "Elmesélek egy történetet a nyelvtechnológiáról."

print(
    puli.text_completion(
        prompt,
        max_new_tokens=100,
        batch_size=1,
        strategy="top_k_sampling",
        temperature=1.3,
        top_k=3,
    )
)

inputs = puli.tokenizer.encode(prompt).to(device)
print(f"token/sec: {puli.calculate_token_per_second(inputs, 100)}")