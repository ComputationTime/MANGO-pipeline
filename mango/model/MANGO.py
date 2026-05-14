# FIX WHICH CHAIN IS CALLED + FIX HOT / COLD SPOTS
import os
import sys

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

import mango
from mango.utils.mango_utils import *
from mango.utils.AbLM_embeddings import *
from mango.utils.cross_attention import * 
from mango.utils.Ag_structure_embeddings import * 

torch.cuda.empty_cache() # Important while debugging


"""
chain_aware_embeddings = self.encoder(
    antigen_pdb_path=antigen_pdb_path, 
    antigen_chains=antigen_chains,
    ab_seq_context=ab_seq_context, 
    chain=chains_to_generate, 
    ag_cold_spots=ag_cold_spots, -------------------------------
    ag_hot_spots=ag_hot_spots    -------------------------------
)
"""

class EncodeInputs(nn.Module):
    """ Creates Ag structural embeddings and supplements it with optional antibody context. The representations
    are combined via a learned cross-attention and then used by MANGO to generate de novo ab sequences. If antibody context
    is not given, the Ag structural embeddings will suffice as context to generate and the Ab will be completely masked.
    """
    def __init__(self, d_model, n_heads, n_layers, ag_representation, mpnn_type='vanilla_models', mpnn_noise=2, bb_perturbation=0.0):
        super().__init__()

        self.MPNN_TYPE = mpnn_type
        self.MPNN_NOISE = mpnn_noise 
        self.MPNN_BB_PERTURBATION = bb_perturbation

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ag_struct_encoder = Ag_embeddings(method=ag_representation)
        self.ab_seq_encoder = AbLM_embeddings()

        self.cross_attn = TransformerStack(
            d_model=d_model, 
            n_heads=n_heads, 
            n_layers=n_layers, 
            d_Ag_rep=MODEL_TO_HDIM[ag_representation]
        ).to(self.device)

    def forward(self, antigen_pdb_path, antigen_chains, ab_seq_context, chain, ag_cold_spots=None, ag_hot_spots=None,):

        #cold_spots, hot_spots (NEXT IMPLEMENTATION)
        h_V = self.ag_struct_encoder.embed(antigen_pdb_path, antigen_chains, 
                                           mpnn_type= self.MPNN_TYPE, noise=self.MPNN_NOISE, bb_perturbation=self.MPNN_BB_PERTURBATION)
        x = self.ab_seq_encoder.embed(ab_seq_context) # If no chain, this will be a [MASK] token that "absorbs" all Ag info

        input_embeddings = self.cross_attn(x, h_V.to(self.device), ag_cold_spots, ag_hot_spots)

        pad_len = MAX_SEQ_GEN_LENGTH - input_embeddings.shape[1]
        input_embeddings = F.pad(
            input_embeddings,
            (0, 0, 0, pad_len),
            value=0
        )

        return input_embeddings

class MANGO(GPT2LMHeadModel):
    """ Underlying custom model that creates Ag-informed embeddings and returns a loss (for training) """
    def __init__(self, configs, ag_representation='One_hot', n_cross_attn_heads=1, n_cross_attn_layers=1):
        super().__init__(configs)

        self.encoder = EncodeInputs(
            configs.hidden_size,
            n_heads=n_cross_attn_heads,
            n_layers=n_cross_attn_layers,
            ag_representation=ag_representation
        )
    
    def _get_custom_input_embeddings(self, antigen_pdb_path, antigen_chains, ab_seq_context, chain_to_generate): 
        return self.encoder(
            antigen_pdb_path=antigen_pdb_path,
            antigen_chains=antigen_chains,
            ab_seq_context=ab_seq_context,
            chain=chain_to_generate
        )
    
    def get_MANGO_Casual_loss(self, antigen_pdb_path, antigen_chains, ab_seq_context, chain, labels):

        embeddings = self.encoder(
            antigen_pdb_path=antigen_pdb_path,
            antigen_chains=antigen_chains,
            ab_seq_context=ab_seq_context,
            chain=chain
        )

        outputs = super().forward(
            inputs_embeds=embeddings,
            labels=labels
        )
        
        return outputs.loss