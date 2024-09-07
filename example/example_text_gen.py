from puli import load_model

puli = load_model("puli2-gpt")

print(
    puli.text_completion(
        "Szia, te ki vagy?",
        max_new_tokens=10,
        strategy="top_k_sampling",
        temperature=1.3,
        top_k=3
    )
)
