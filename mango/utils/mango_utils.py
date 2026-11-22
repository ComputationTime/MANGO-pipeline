import os

import transformers

# <, -, *, X, |
BAD_WORD_IDS = [[0], [21], [23], [24], [25],]

MODEL_TO_HDIM = {
    "One_hot":21,
    "ESM2_t48_15B":5120,
    "ESM2_t36_3B":2560, 
    "ESM2_t33_650M":1280, 
    "ESM2_t30_150M":640, 
    "ESM2_t12_35M":480, 
    "ESM2_t6_8M":320, 
    "ESM-IF":0, 
    "ProteinMPNN":1280, 
    "AF-M":0, 
    "ESM3":0, 
    "PyRosetta_PRE":1, # Per-residue property
    "Biophysics":11, # Global property
}


mapping_chains = {
    "Fv":"both",
    "scFv":"both",
    "VHH":"H",
    "VL-only":"L",
}

ablang_vocab = {
    "<": 0,    # Start token
    "-": 21,   # Padding token
    ">": 22,   # End token
    "*": 23,   # Mask token
    "X": 24,   # Unknown (residue) token
    "|": 25,   # Separation (of heavy and light chain) token
    "M": 1, 
    "R": 2, 
    "H": 3, 
    "K": 4, 
    "D": 5, 
    "E": 6, 
    "S": 7, 
    "T": 8, 
    "N": 9, 
    "Q": 10, 
    "C": 11,
    "G": 12, 
    "P": 13, 
    "A": 14, 
    "V": 15, 
    "I": 16, 
    "F": 17, 
    "Y": 18, 
    "W": 19, 
    "L": 20,
}

ablang_decode = {i:j for j,i in ablang_vocab.items()}

MAX_SEQ_GEN_LENGTH = 300 

MANGO_configs = transformers.GPT2Config(
    vocab_size=26, # Should be 26
    n_positions=2048,
    n_layer=4,
    n_head=8,
    hidden_size=480
    #hidden_dropout_prob=0.3, # Added dropout probability
    #attention_probs_dropout_prob=0.3 # Added attention dropout probability
) 

project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
trained_models_dir = os.path.join(project_path, 'trained_models')

# Checkpoints are expected at <package>/trained_models/<name>.pt
MANGO_TRAINED_MODELS = {
    name: os.path.join(trained_models_dir, name)
    for name in ("One_hot", "ProteinMPNN", "PyRosetta_PRE", "Biophysics", "MANGO")
}

def mask_span(seq, start, end, append_span = False):
    masked_seq = seq[:start] + ['[MASK]'] + seq[end:] + ['[SEP]']
    if append_span:
        masked_seq += seq[start:end]

    return masked_seq
