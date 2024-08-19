

EVAL_SPLIT = 0.01       # train-test split

EPOCHS = 32             # number of epochs to train for (if early stopping doesn't intervene)
DETERMINISTIC = True    # set random seed for reproducibility

MAX_NEW_TOKENS = 100    # number of tokens to generate
TOP_K = 1               # top k for sampling
TEMPERATURE = 1.0       # temperature scaling

TOKENISER_MODEL_PATH = ""

PAD_IDX = -100


LOGS_PATH = "logs.txt"
MODEL_PATH = "gpt-2.pt"