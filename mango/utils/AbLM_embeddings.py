import numpy
import torch
import ablang2
import transformers


"""
From AbLang2/tokenizers.py: 

self.pad_token = '-'
self.start_token = '<'
self.end_token = '>'
self.sep_token = '|'
self.mask_token = '*'
self.unknown_token = 'X'
"""
class AbLM_embeddings():
    def __init__(self):
        """ Generate embeddings using AbLang2. If no chain given (e.g. VHH, Bence Jones proteins [BJP], scFvs, etc.), use a mask token""" 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ab_model = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=1, device=self.device)

    def _tokenize(self, seq):
        """ seq (list): ['HEAVY', 'LIGHT'] """
        seqs = [f"{seq[0]}|{seq[1]}"] # AbLang2 input needs to be a list, with | used to separated the VH and VL 
        return self.ab_model.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.device)
    
    def embed(self, input_sequences=None, get_tokens=False):
        '''Applies self-attention, but then zero-out positions where input was padded'''
        AbLang2_tokens = self._tokenize(input_sequences)

        with torch.no_grad():
            raw_embeddings = self.ab_model.AbRep(AbLang2_tokens).last_hidden_states

        if get_tokens:
            return raw_embeddings, AbLang2_tokens
        else:
            return raw_embeddings

    def cleaner_embeddings(self, input_sequences=None, return_attentions=False):
        AbLang2_tokens = self._tokenize(input_sequences)

        with torch.no_grad(): # attentions, embeddings, logits 
            outputs = self.ab_model(
                input_ids = AbLang2_tokens['input_ids'],
                attention_mask = AbLang2_tokens['attention_mask'], # (B,L)
            )

        # Remove padding, get a list where each tensor is [(Length+3) x 1024]
        cleaned_embeddings = []
        for emb, mask in zip(outputs.last_hidden_state, AbLang2_tokens['attention_mask']):
            cleaned_embeddings.append(emb[mask.bool()]) # Extra padding would ruining mean_pooled embeddings

        # A list where each tensor is [(Length+1) x (Length+1)]
        if return_attentions==True: 
            cleaned_attns = [] # N_seqs x L_max x L_max (always 1 head)
            for attn, mask in zip(outputs.attentions, AbLang2_tokens['attention_mask']):
                cleaned_attns.append(attn[mask.bool()][:, mask.bool()]) # Clean up rows and columns that were padded
                
            return cleaned_embeddings, cleaned_attns

        else:
            return cleaned_embeddings

#AbLang2 = AbLM_embeddings()
#x = AbLang2.embed(['*', 'DIK']) # Using [MASK] does not stop things from attending to it (GOOD)
#print(x.shape)
#print(x[0].shape)

"""
LIGHT: tensor([[23, 25,  5, 16,  4]])
HEAVY: tensor([[ 5, 16,  4, 25, 23]])
"""