import torch

from puli import load_model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

puli = load_model("puli2-gpt", device)

print(
    puli.text_completion(
        "Elmesélek egy történetet a nyelvtechnológiáról.",
        max_new_tokens=10,
        batch_size=1,
        strategy="top_k_sampling",
        temperature=1.3,
        top_k=3,
    )
)