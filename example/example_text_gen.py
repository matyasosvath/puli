from puli import load_model

puli = load_model("puli2-gpt")

print(puli.text_completion("Szia, te ki vagy?", max_gen_len=10))