import os
import copy
import random
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from mango.utils.MPNN_utils import *
from mango.utils.MPNN_features import *


ca_models = '/weka/scratch/jgray21/dvincen9/learning/ProteinMPNN/ca_model_weights'
vanilla_models = '/weka/scratch/jgray21/dvincen9/learning/ProteinMPNN/vanilla_model_weights'
soluble_models = '/weka/scratch/jgray21/dvincen9/learning/ProteinMPNN/soluble_model_weights'

CHECKPOINTS = {
    "ca_models": {
        2:os.path.join(ca_models, 'v_48_002.pt'),
        10:os.path.join(ca_models, 'v_48_010.pt'),
        20:os.path.join(ca_models, 'v_48_020.pt'),
        30:os.path.join(ca_models, 'v_48_020.pt'),
    },

    "vanilla_models": {
        2:os.path.join(vanilla_models, 'v_48_002.pt'),
        10:os.path.join(vanilla_models, 'v_48_010.pt'),
        20:os.path.join(vanilla_models, 'v_48_020.pt'),
        30:os.path.join(vanilla_models, 'v_48_030.pt'),
    },

    "soluble_models": {
        2:os.path.join(soluble_models, 'v_48_002.pt'),
        10:os.path.join(soluble_models, 'v_48_010.pt'),
        20:os.path.join(soluble_models, 'v_48_020.pt'),
        30:os.path.join(soluble_models, 'v_48_030.pt'),
    },

}

class ProteinMPNN_Encoder(nn.Module):
    '''
    Goal: input should just be model = 'ca_models', 'vanilla_models', 'soluble_models' --> Then the class should handle the rest
    Initialize by picking your designed ProteinMPNN model, then selecting an amount of backbone noise to add: 

    Args: 
        1. model (str): Whether to load ca_models, vanilla_models, soluble_models 
        2. noise (int): Which training noise weights to load (2, 10, 20, 30)
        3. bb_perturbation (float): How much to randomly perturb the backbone coordinates
    '''

    def __init__(self, model='vanilla_models', noise=2, bb_perturbation=0.0):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        weights_to_load = CHECKPOINTS[model][noise]

        checkpoint = torch.load(weights_to_load, map_location = self.device) # map_location = device
        
        print(f'This ProteinMPNN model was trained with: {checkpoint["noise_level"]}Å noise')
        hidden_dim = checkpoint['model_state_dict']['features.edge_embedding.weight'].shape[0]

        self.ca_only = True if model=='ca_models' else False

        self._load_ProteinMPNN(
            num_letters=21, 
            node_features=hidden_dim, 
            edge_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_encoder_layers=3, # 3
            num_decoder_layers=3, # 3
            vocab=21, 
            k_neighbors=checkpoint['num_edges'], # 64?, 48
            augment_eps=bb_perturbation,
            dropout=0.1, 
            ca_only=self.ca_only
        )
        # The contributed implementation built the correct architecture but
        # never applied the pretrained checkpoint, leaving random Xavier
        # weights. Load strictly so an upstream checkpoint/API mismatch fails
        # instead of silently producing meaningless embeddings.
        self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _load_ProteinMPNN(self, num_letters, node_features, edge_features,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, ca_only=True):
        super(ProteinMPNN_Encoder, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim


        # Featurization layers
        if ca_only:
            self.features = CA_ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers (Never used, but keeping to avoid "Error(s) Unexpected key(s) in state_dict")
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode(self, X, S, mask, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers: # This is where ProteinMPNN updates via message passing
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        
        return h_V
    
    def encode(self, antigen_pdb_path, ag_chains, max_len=20000): # Max length from ProteinMPNN
        
        pdb_dict_list = parse_PDB(antigen_pdb_path, input_chain_list=ag_chains)
        print(f"Using ProteinMPNN to encode {pdb_dict_list[0]['num_of_chains']} chains: {ag_chains}")

        dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_len)

        with torch.no_grad():
            for ix, protein in enumerate(dataset_valid):
                batch_clones = [copy.deepcopy(protein) for i in range(1)] # Always uses 1 antigen?
                X, S, mask, chain_encoding_all, residue_idx,  = tied_featurize_minimal(batch_clones, self.device, ca_only=self.ca_only)

        # X: (B, L, 3) --> embeddings: (B,L, H_dim)
        embeddings = self._encode(X, S, mask, residue_idx, chain_encoding_all) # (1, L, 128)

        return embeddings


#device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

#model = ProteinMPNN_Encoder()
#model.encode('/scratch/jgray21/dvincen9/projects/0_EXP_DATA/AbYBank/LH_Protein_Chothia_3000/1A2Y_1.pdb', ag_chains= "H : L") # torch.Size([1, 223, 3])


'''
Notes: 

Tied_positions_homomer: 
    1. Encoder takes the bb str, and produces embeddings per residue
    2. Encoder does not need to know tied positions since it just embeds the 3D coordinates into features 
    3. Decoder uses tied positions to force amino acids to have the same AA prob distributions --> So it doesn't break symmetry

'''

'''
homomer = True
designed_chain = "H L"
fixed_chain = ""

if designed_chain == "":
  designed_chain_list = []
else:
  designed_chain_list = re.sub("[^A-Za-z]+",",", designed_chain).split(",")

if fixed_chain == "":
  fixed_chain_list = []
else:
  fixed_chain_list = re.sub("[^A-Za-z]+",",", fixed_chain).split(",")
#print(designed_chain_list)

chain_list = list(set(designed_chain_list + fixed_chain_list))


num_seqs = 1 #@param ["1", "2", "4", "8", "16", "32", "64"] {type:"raw"}
num_seq_per_target = num_seqs


sampling_temp = "0.1" #@param ["0.0001", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5"]



save_score=0                      # 0 for False, 1 for True; save score=-log_prob to npy files
save_probs=0                      # 0 for False, 1 for True; save MPNN predicted probabilites per position
score_only=0                      # 0 for False, 1 for True; score input backbone-sequence pairs
conditional_probs_only=0          # 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)
conditional_probs_only_backbone=0 # 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)

batch_size=1                      # Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory
max_length=20000                  # Max sequence length

out_folder='.'                    # Path to a folder to output sequences, e.g. /home/out/
jsonl_path=''                     # Path to a folder with parsed pdb into jsonl
omit_AAs='X'                      # Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.

pssm_multi=0.0                    # A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions
pssm_threshold=0.0                # A value between -inf + inf to restric per position AAs
pssm_log_odds_flag=0               # 0 for False, 1 for True
pssm_bias_flag=0                   # 0 for False, 1 for True



folder_for_outputs = out_folder

NUM_BATCHES = num_seq_per_target//batch_size
BATCH_COPIES = batch_size
temperatures = [float(item) for item in sampling_temp.split()]
omit_AAs_list = omit_AAs
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

chain_id_dict = None
fixed_positions_dict = None
pssm_dict = None
omit_AA_dict = None
bias_AA_dict = None
tied_positions_dict = None
bias_by_res_dict = None
bias_AAs_np = np.zeros(len(alphabet))


pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)

chain_id_dict = {}
chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

#for chain in chain_list:
  #l = len(pdb_dict_list[0][f"seq_chain_{chain}"])
  #print(f"Length of chain {chain} is {l}")

# Homomer is more important for decoding... not sure if I really need it for encoding
if homomer:
  tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
else:
  tied_positions_dict = None


with torch.no_grad():
  for ix, protein in enumerate(dataset_valid):
    batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
    print(batch_clones)
    print("\n\n\n", device)
    print("\n\n\n", chain_id_dict)
    print("\n\n\n", fixed_positions_dict)
    print("\n\n\n", omit_AA_dict)
    print("\n\n\n", tied_positions_dict)
    print("\n\n\n", pssm_dict)
    print("\n\n\n", bias_by_res_dict)


    # X, S, mask, _, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta
    X, S, mask, _, _, chain_encoding_all, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _,  = tied_featurize(
        batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=True)
    
    print(X.shape)

'''
