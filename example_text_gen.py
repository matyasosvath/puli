import torch

import utils
from puli import load_model
import utils.helper


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# puli = load_model("puli2-gpt", device)
puli = load_model("puli2-gpt", device, mode="int8")

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
tps = puli.calculate_token_per_second(inputs, 100)
print(f"token/sec: {tps}")

mbu = utils.helper.get_model_bandwidth_utilization(puli.model, "float32", tps)
print(f"mbu: {mbu}%")