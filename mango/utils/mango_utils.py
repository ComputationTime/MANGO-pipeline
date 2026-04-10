import numpy as np 

# <, -, >, *, X, |
BAD_WORD_IDS = [[0], [21], [22], [23], [24], [25],]

MODEL_TO_HDIM = {
    "One_hot":20,
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

def mask_span(seq, start, end, append_span = False):
    masked_seq = seq[:start] + ['[MASK]'] + seq[end:] + ['[SEP]']
    if append_span:
        masked_seq += seq[start:end]

    return masked_seq


def validate_MANGO_seq(input_ids):
    """ Validate that generated input ids are well formed ( if H,L no [SEP] else only 1 [CLS]) """

    # scFv:   [CHAIN][AG-INFORMED EMBEDDINGS] [E V Q] [SEP] [L V E S S] [CLS]
    # H or L: [CHAIN][AG-INFORMED EMBEDDINGS] [E V Q ... V E S S] [CLS]
    cls_idx = np.where(input_ids == self.tokenizer.cls_token_id)[0] # (positions, dtype)
    sep_idx = np.where(input_ids == self.tokenizer.sep_token_id)[0]

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