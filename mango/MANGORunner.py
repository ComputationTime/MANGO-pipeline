# FIX WHICH CHAIN IS CALLED + FIX HOT / COLD SPOTS
import torch
import transformers
from tqdm import tqdm

import mango
from mango import MANGO
from mango.utils.mango_utils import *

torch.cuda.empty_cache() # Important while debugging

# NEED A WRAPPER TO LOAD IN ALL MODEL WEIGHTS CORRECTLY, CAN'T JUST USE SELF.TRANSFORMERS.LOAD_FROM_PRETRAINED
""" Main MANGO model that is used to format generated sequences (for inference) """

class MANGORunner():
    def __init__(self, ag_representation='One_hot'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("hemantn/ablang2", trust_remote_code=True)
        self.tokenizer.cls_token_id = 22 # from Ablang2 github

        CHECKPOINT_PATH = f"{MANGO_TRAINED_MODELS[ag_representation]}.pt"
        CHECKPOINT = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        
        self.model = MANGO(MANGO_configs, ag_representation=ag_representation)
        self.model.encoder.load_state_dict(CHECKPOINT['encoder'])
        self.model.lm_head.load_state_dict(CHECKPOINT['lm_head'])


    def generate(self, antigen_pdb_path, antigen_chains, prompt_ab_seq='*', chain_to_generate='H', 
                 num_to_generate=5, top_p=1, temperature=1, ag_cold_spots=None, ag_hot_spots=None,): 
        '''
        Function to generate sequences from MANGO, an antigen conditioned autoregressive model. 

        Args: 
            antigen_pdb_path (str): Path to the antigen structure to condition generation 
            antigen_chains (list): Chain(s) in the PDB to use as antigen context (e.g. ['A'] or ['A','B'])
            prompt_ab_seq (str): An optional single prompt sequence (either H or L)
            chain_to_generate (str): 'H', 'L', or 'scFv'.[NOTE: scFv does not take a prompt sequence]
            num_to_generate (int): How many sequences to generate from the model 
            top_p (int): Maximum cdf to consider (from sum of token probability). [Compared to top_k]
            temperature (float): sampling temperature, higher is more even distribution 
            ag_cold_spots (list of ints): Manually curated positions to ignore during cross attention
            ag_hot_spots (list of ints): Manually curated positinos to highlight during cross attention

        Returns:
            generated_seqs: A list of generated sequences
        '''

        #if len(prompt_ab_seq)<L_MAX:
        #    N_PAD = (L_MAX-len(prompt_ab_seq))
        #    prompt_ab_seq += '*'* N_PAD

        if chain_to_generate=='H':
            ab_seq_context = [prompt_ab_seq,'*'] # Prompt, MASK
        elif chain_to_generate=='L':
            ab_seq_context = ['*', prompt_ab_seq] # MASK, Prompt
        elif chain_to_generate=='scFv':
            ab_seq_context = ['*','*'] # MASK, MASK

        
        inputs_embeddings = self.model._get_custom_input_embeddings(
            antigen_pdb_path=antigen_pdb_path,
            antigen_chains=antigen_chains, 
            ab_seq_context=ab_seq_context,
            chain_to_generate=chain_to_generate)

        
        decoded_seqs = set()
        pbar= tqdm(total=num_to_generate)

        while len(decoded_seqs) < num_to_generate:
            seq = self.model.generate(
                inputs_embeds=inputs_embeddings,
                max_new_tokens=300, # max_length = total sequence length including prompt ; max_new_tokens = only newly generated tokens
                pad_token_id=21, # from looking at Ablang2 tokenizer
                eos_token_id=22,
                forced_eos_token_id=22,
                bad_words_ids=BAD_WORD_IDS,
                do_sample=True,
                top_p=top_p,
                temperature=temperature
            ).detach().cpu().numpy() # (B,L_gen)
            
            decoded_tokens = self.tokenizer.decode(seq[0, :-1]) # Squeeze out batch dimension, remove [CLS] at end
            #print(decoded_tokens)
            decoded_seq = ''.join(decoded_tokens).replace(' ', '')
            if decoded_seq not in decoded_seqs:
                decoded_seqs.add(decoded_seq)
                pbar.update(1)
        
        pbar.close()
        return list(decoded_seqs)

    def log_likelihood(self, antigen_pdb_path, sequences, chains_to_score='H', ag_cold_spots=None, ag_hot_spots=None,):

        lls = []
        with torch.no_grad():
            for seq in sequences:
                if chains_to_score=='H':
                    prompt_sequence = [seq, '*']
                elif chains_to_score=='L':
                    prompt_sequence = ['*', seq]
                elif chains_to_score=='scFv':
                    propmt_sequence = [seq, '']

                chain_aware_embeddings, token_seq = self.encoder(antigen_pdb_path=antigen_pdb_path, ab_seq_context=prompt_sequence, chain=chains_to_score, ag_cold_spots=ag_cold_spots, ag_hot_spots=ag_hot_spots, get_tokens=True)
                
                logits = self.model(inputs_embeds=chain_aware_embeddings).logits # B x Lmax x 30
                shift_logits = logits[..., 1:-1, :].contiguous() # Remove [Ag_embeddings] ... [CLS]
                shift_labels = token_seq[..., 1:].contiguous().long() # [Ag_embeddings]
                nll = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean',
                )

                lls.append(-nll.item())

        return lls



#model = MANGORunner()
#model._check_IgBERT_vocab()

#list_of_de_novo_seqs = model.generate(antigen_pdb_path='/weka/scratch/jgray21/dvincen9/TRAINING/MANGO/SAbDAb/structures/8hnm.pdb', antigen_chains=['A', 'B'])
#model.log_likelihood(antigen_pdb_path=['/scratch/jgray21/dvincen9/projects/MANGO/mango/utils/Penta_Alanine_Antigen.pdb'], sequences=['QVQ', "EIV"])
#print(list_of_de_novo_seqs)


# ENSURE OPTIMIZER DOESN'T INCLUDE FROZEN WEIGHTS 
#optimizer = torch.optim.Adam(
#    filter(lambda p: p.requires_grad, model.parameters()),
#    lr=1e-4
#)