import numpy as np 

import os
import mango
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

project_path = os.path.dirname(os.path.realpath(mango.__file__))
trained_models_dir = os.path.join(project_path, 'trained_models')

MANGO_TRAINED_MODELS = {
    "One_hot": os.path.join(trained_models_dir, 'One_hot'),
    "ESM2_": os.path.join(trained_models_dir, 'ESM2_'),
    "ESM2_": os.path.join(trained_models_dir, 'ESM2_'),
    "ESM2_": os.path.join(trained_models_dir, 'ESM2_'),
    "ESM2_": os.path.join(trained_models_dir, 'ESM2_'),
    "ESM2_": os.path.join(trained_models_dir, 'ESM2_'),
    "ESM2_": os.path.join(trained_models_dir, 'ESM2_'),
    "ESM2_": os.path.join(trained_models_dir, 'MANGO'),
    "ESM2_": os.path.join(trained_models_dir, 'MANGO'),
    "ESM2_": os.path.join(trained_models_dir, 'MANGO'),
    "ProteinMPNN": os.path.join(trained_models_dir, 'ProteinMPNN'),
    "PyRosetta_PRE": os.path.join(trained_models_dir, 'PyRosetta_PRE'),
    "Biophysics": os.path.join(trained_models_dir, 'Biophysics'),
    "MANGO": os.path.join(trained_models_dir, 'MANGO'),
}

def mask_span(seq, start, end, append_span = False):
    masked_seq = seq[:start] + ['[MASK]'] + seq[end:] + ['[SEP]']
    if append_span:
        masked_seq += seq[start:end]

    return masked_seq


def validate_MANGO_seq(input_ids):
    """ Validate that generated input ids are well formed ( if H,L no [SEP] else only 1 [CLS]) """

    # scFv:   [CHAIN][AG-INFORMED EMBEDDINGS] [E V Q L V E S S] [CLS]
    # H or L: [CHAIN][AG-INFORMED EMBEDDINGS] [E V Q ... V E S S] [CLS]
    cls_idx = np.where(input_ids == 22)[0] # (positions, dtype)
    sep_idx = np.where(input_ids == 25)[0]

    #mask_idx = np.where(input_ids == tokenizer.mask_token_id)[0] # Because he brings it back around front... so I actually don't need this 
    #sep_idx = np.where(input_ids == tokenizer.sep_token_id)[0]
    #cls_idx = np.where(input_ids == tokenizer.cls_token_id)[0]

    if len(mask_idx) != 1 or len(sep_idx) != 3 or len(cls_idx) != 1:
        return False # Needs one of each


    # if chain_to_generate = 'H' or 'L' --> NO [SEP]
    # if chain_to_generate = 'scFv' --> ONE [SEP]

    #if len(cls_idx) != 0 or len(sep_idx) != 0:
    #    return False
    print('NOW ITS READING this FUCK ASS FUNCTION')
    
    return True


def validate_generated_seq(input_ids, tokenizer):
    """
    Validate that generated input ids are well formed ([MASK] before [SEP] and [CLS] and that there's one of each).
    """
    mask_idx = np.where(input_ids == tokenizer.mask_token_id)[0]
    sep_idx = np.where(input_ids == tokenizer.sep_token_id)[0]
    cls_idx = np.where(input_ids == tokenizer.cls_token_id)[0]

    if len(mask_idx) != 1 or len(sep_idx) != 1 or len(cls_idx) != 1:
        return False

    mask_idx = mask_idx.squeeze()
    sep_idx = sep_idx.squeeze()
    cls_idx = cls_idx.squeeze()

    return (mask_idx < sep_idx) and (sep_idx < cls_idx)