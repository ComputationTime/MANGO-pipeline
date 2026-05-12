# FIX WHICH CHAIN IS CALLED + FIX HOT / COLD SPOTS
import os
import sys

import torch
import torch.nn as nn
import transformers
from transformers import GPT2LMHeadModel
from tqdm import tqdm

import mango
from mango.utils.mango_utils import *
from mango.utils.AbLM_embeddings import *
from mango.utils.cross_attention import * 
from mango.utils.Ag_structure_embeddings import * 

torch.cuda.empty_cache() # Important while debugging


# scFv:   [CHAIN][AG-INFORMED EMBEDDINGS] [E V Q] [SEP] [L V E S S] [CLS]
# H or L: [CHAIN][AG-INFORMED EMBEDDINGS] [E V Q ... V E S S] [CLS] *********

project_path = os.path.dirname(os.path.realpath(mango.__file__))
trained_models_dir = os.path.join(project_path, 'trained_models')

CHECKPOINT_DICT = {
    "MANGO": os.path.join(trained_models_dir, 'MANGO'),
    "MANGO-S": os.path.join(trained_models_dir, 'MANGO-S'),
}
#ABLANG2_VOCAB_FILE = os.path.join(project_path, 'utils/Ablang2_vocab.txt')

L_MAX = 300 #AMINO ACIDS

class EncodeInputs(nn.Module):
    """
    Creates structural embeddings for the Ag structure and supplements it with optional antibody context. The representations
    are combined via a learned cross-attention and then used by MANGO to generate de novo ab sequences. If antibody context
    is not given, the Ag structural embeddings will suffice as context to generate and the Ab will be completely masked.
    """
    def __init__(self, d_model, n_heads, n_layers, ag_representation, mpnn_type='vanilla_models', mpnn_noise=2, bb_perturbation=0.0):
        super().__init__()

        self.MPNN_TYPE = mpnn_type
        self.MPNN_NOISE = mpnn_noise 
        self.MPNN_BB_PERTURBATION = bb_perturbation

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.chain_embed = nn.Embedding(3, 480) # 3 chains: H, L, scFv --> AbLM dim: 480
        self.ag_struct_encoder = Ag_embeddings(method=ag_representation)
        self.ab_seq_encoder = AbLM_embeddings()

        self.cross_attn = TransformerStack(
            d_model=d_model, 
            n_heads=n_heads, 
            n_layers=n_layers, 
            d_Ag_rep=MODEL_TO_HDIM[ag_representation]
        ).to(self.device)

    def forward(self, antigen_pdb_path, antigen_chains, ab_seq_context, chain, ag_cold_spots, ag_hot_spots, get_tokens=False):

        #cold_spots, hot_spots (NEXT IMPLEMENTATION)
        h_V = self.ag_struct_encoder.embed(antigen_pdb_path, antigen_chains, 
                                           mpnn_type= self.MPNN_TYPE, noise=self.MPNN_NOISE, bb_perturbation=self.MPNN_BB_PERTURBATION)
        x = self.ab_seq_encoder.embed(ab_seq_context, get_tokens=get_tokens) # If no chain, this will be a [MASK] token that "absorbs" all Ag info

        if get_tokens: 
            x, tokens = x # unpack tuple

        input_embeddings = self.cross_attn(x, h_V.to(self.device), ag_cold_spots, ag_hot_spots)

        if chain=='H':
            chain_embeddings = self.chain_embed(torch.tensor([0]).to(self.device)) # (B,1024)
        elif chain=='L':
            chain_embeddings = self.chain_embed(torch.tensor([1]).to(self.device)) # (B,1024)
        elif chain=='scFv':
            chain_embeddings = self.chain_embed(torch.tensor([2]).to(self.device)) # (B,1024) - Should learn this token means it can generate longer

        chain_aware_embeddings = torch.cat([
            chain_embeddings.unsqueeze(1).to(self.device),
            input_embeddings
        ], dim=1)

        pad_len = L_MAX - chain_aware_embeddings.shape[1]
        chain_aware_embeddings = F.pad(
            chain_aware_embeddings,
            (0, 0, 0, pad_len),
            value=0
        )
            
        if get_tokens:
            return chain_aware_embeddings, tokens
        else: 
            return chain_aware_embeddings


class MANGO(nn.Module):

    def __init__(self, ag_representation='One_hot', model_name="MANGO", n_cross_attn_heads=1, n_cross_attn_layers=1):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("hemantn/ablang2", trust_remote_code=True)
        self.tokenizer.cls_token_id = 22 # from Ablang2 github

        #self.model = transformers.GPT2LMHeadModel.from_pretrained(CHECKPOINT_DICT[model_name]).to(self.device)
        #self.model.load_state_dict(torch.load(''))

        configs = transformers.GPT2Config(
            vocab_size=len(self.tokenizer), # Should be 26
            n_positions=2048,
            n_layer=4,
            n_head=8,
            hidden_size=480
        ) #12M parameters
        self.model = transformers.GPT2LMHeadModel(configs).eval().to(self.device)
        #self.model.load_state_dict(torch.load('/scratch/jgray21/dvincen9/projects/MANGO/mango/trained_models/One_Hot/model.pt'))
       
        #total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #print(f"Total parameters: {total_params:,}")


        self.encoder = EncodeInputs(
            configs.hidden_size, 
            n_heads=n_cross_attn_heads, 
            n_layers=n_cross_attn_layers, 
            ag_representation=ag_representation
        )
    
    def _generate(self, input_embeddings, chains_to_generate, num_to_generate, top_p, temperature):
        """Inference-only: samples sequences autoregressively."""
        decoded_seqs = set()  # Set to remove duplicates
        pbar = tqdm(total=num_to_generate)
        
        while len(decoded_seqs) < num_to_generate:
            seq = self.model.generate(
                inputs_embeds=input_embeddings,
                max_length=300, #scFV, else it should only generate chains
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.cls_token_id,
                forced_eos_token_id=self.tokenizer.cls_token_id,
                bad_words_ids=BAD_WORD_IDS,
                do_sample=True,
                top_p=top_p,
                temperature=temperature
            ).detach().cpu().numpy() # Always (B,L_gen)
            
            decoded_tokens = self.tokenizer.decode(seq[0, :-1]) # Squeeze out batch dimension, remove [CLS] at end
            #print(decoded_tokens)
            decoded_seq = ''.join(decoded_tokens).replace(' ', '')
            if decoded_seq not in decoded_seqs:
                decoded_seqs.add(decoded_seq)
                pbar.update(1)
        
        pbar.close()
        return list(decoded_seqs)

    def forward(self, antigen_pdb_path, antigen_chains, prompt_ab_seq='*', chains_to_generate='H',
                labels=None, ag_cold_spots=None, ag_hot_spots=None):
        """
        Forward pass used during training. Returns loss if labels are provided, else logits.

        Args:
            antigen_pdb_path (str): Path to the antigen PDB structure.
            antigen_chains (list): Chain IDs to use as antigen context.
            prompt_ab_seq (str): Optional prompt sequence (H or L chain); '*' for none.
            chains_to_generate (str): 'H', 'L', or 'scFv'.
            labels (torch.LongTensor): Token IDs of shape (1, L) for computing cross-entropy loss.
            ag_cold_spots (list of int): Antigen positions to down-weight in cross-attention.
            ag_hot_spots (list of int): Antigen positions to highlight in cross-attention.

        Returns:
            loss (torch.Tensor) if labels provided, else logits (torch.Tensor).
        """
        if chains_to_generate == 'H':
            ab_seq_context = ['', prompt_ab_seq]
        elif chains_to_generate == 'L':
            ab_seq_context = [prompt_ab_seq, '']
        elif chains_to_generate == 'scFv':
            # FIX: was referencing undefined `prompt_sequence`
            ab_seq_context = [prompt_ab_seq, '']

        chain_aware_embeddings = self.encoder(
            antigen_pdb_path=antigen_pdb_path, 
            antigen_chains=antigen_chains,
            ab_seq_context=ab_seq_context, 
            chain=chains_to_generate, 
            ag_cold_spots=ag_cold_spots, 
            ag_hot_spots=ag_hot_spots
        )  # (1, L_ctx, d_model)

        output = self.model(inputs_embeds=chain_aware_embeddings, labels=labels)
        
        if labels is not None:
            return output.loss
        else:
            return output.logits

    def generate(self, antigen_pdb_path, antigen_chains, prompt_ab_seq='*', chains_to_generate='H', 
                 num_to_generate=10, top_p=1, temperature=1, ag_cold_spots=None, ag_hot_spots=None,): 
        '''
        Function to generate sequences from MANGO, an antigen conditioned autoregressive model. 

        Args: 
            antigen_pdb_path (str): Path to the antigen structure to condition generation 
            antigen_chains (list): Chain(s) in the PDB to use as antigen context (e.g. ['A'] or ['A','B'])
            prompt_ab_seq (str): An optional single prompt sequence (either H or L)
            chains_to_generate (str): 'H', 'L', or 'scFv'.[NOTE: scFv does not take a prompt sequence]
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

        if chains_to_generate=='H':
            ab_seq_context = ['',prompt_ab_seq]
        elif chains_to_generate=='L':
            ab_seq_context = [prompt_ab_seq,'']
        elif prompt_sequence=='*' and chains_to_generate=='scFv':
            ab_seq_context = [prompt_ab_seq,'']

        chain_aware_embeddings = self.encoder(
            antigen_pdb_path=antigen_pdb_path, 
            antigen_chains=antigen_chains,
            ab_seq_context=ab_seq_context, 
            chain=chains_to_generate, 
            ag_cold_spots=ag_cold_spots, 
            ag_hot_spots=ag_hot_spots
        )

        return self._generate(
            input_embeddings=chain_aware_embeddings, 
            chains_to_generate=chains_to_generate, 
            num_to_generate=num_to_generate, 
            top_p=top_p, 
            temperature=temperature
        )       

    def log_likelihood(self, antigen_pdb_path, sequences, chains_to_score='H', ag_cold_spots=None, ag_hot_spots=None,):

        lls = []
        with torch.no_grad():
            for seq in sequences:
                if chains_to_score=='H':
                    prompt_sequence = [seq, '']
                elif chains_to_score=='L':
                    prompt_sequence = ['', seq]
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

#model = MANGO()
#model._check_IgBERT_vocab()

# De novo generation (Ideally want 1 Ag, 1 sequence?)
#list_of_de_novo_seqs = model.generate(antigen_pdb_path='/weka/scratch/jgray21/dvincen9/TRAINING/MANGO/SAbDAb/structures/8hnm.pdb', antigen_chains=['A', 'B'])
#list_of_de_novo_seqs = model.generate(antigen_pdb_path=['/scratch/jgray21/dvincen9/projects/MANGO/mango/utils/Penta_Alanine_Antigen.pdb'])
#model.log_likelihood(antigen_pdb_path=['/scratch/jgray21/dvincen9/projects/MANGO/mango/utils/Penta_Alanine_Antigen.pdb'], sequences=['QVQ', "EIV"])
#print(list_of_de_novo_seqs)


#seqs = model.generate('QVQ')
#print(seqs)


# ENSURE OPTIMIZER DOESN'T INCLUDE FROZEN WEIGHTS 
#optimizer = torch.optim.Adam(
#    filter(lambda p: p.requires_grad, model.parameters()),
#    lr=1e-4
#)