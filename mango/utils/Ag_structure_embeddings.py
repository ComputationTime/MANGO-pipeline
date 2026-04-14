import esm
import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F

from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from mango.utils.MPNN_embeddings import *
from pyrosetta import *

# Note, I still don't have the full set up for ESM-IF and AF-M, so please ignore those for now!
# Note, I broke PyRosetta.... I will fix this later
class Ag_embeddings(nn.Module):
    def __init__(self, method):
        """
        Main MANGO class that lets the user choose how to represent the Antigen input: 
            method (str): One_hot, ESM2_t48_15B, ESM2_t36_3B, ESM2_t33_650M, ESM2_t30_150M, ESM2_t12_35M, ESM2_t6_8M,  
            ESM-IF, ProteinMPNN, AF-M, ESM3, PyRosetta_PRE, Biophysics 

        Functions:
            embed(list of str paths to antigen pdb files) -> tensor(B, Lmax+2, H)
        """
        super().__init__()
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def embed(self, pdb_path, mpnn_type='vanilla_models', noise=2, bb_perturbation=0.0):
        parser = PDBParser(QUIET=True)
        ppb = PPBuilder()

        ag_chains = ['A', 'B'] # Might need to use set later in case of duplicates

        SEQUENCES = {}
        parsed = parser.get_structure('antigen', pdb_path) # FIX LATER, MAKE THIS A STRING INSTEAD OF A LIST
        for pp in ppb.build_peptides(parsed):
            chain_id = pp[0].get_parent().id 
            if chain_id in ag_chains: # No need to save all the useless chains
                SEQUENCES[chain_id] = pp.get_sequence()


        self.L_MAX = 100 #max([len(seq) for seq in SEQUENCES])

        # For all methods L* = sum(chain lens) + n_chain breaks except Biophysics which has GLOBAL properties

        if self.method == "One_hot": # Sequence based --> (1, L*, 20)
            return self.One_hot(SEQUENCES)

        elif self.method[:4] == "ESM2": # Sequence based --> (B, L*, Hdim) where Hdim depends on the ESM2 model used
            return self.ESM2(SEQUENCES)
        
        elif self.method == "ESMIF": # Structure based
            return self.ESMIF(pdb_path)
        
        elif self.method == "ProteinMPNN": # Structure based --> (B, Lmax+2, 1280)
            return self.ProteinMPNN(pdb_path, mpnn_model=mpnn_type, noise=noise, bb_perturbation=bb_perturbation)
        
        elif self.method == "AFM": # Structure based
            return self.AFM(pdb_path)
        
        elif self.method == "ESM3": # Sequence + Structure based --> (B, Lmax+2, )
            return self.ESM3(pdb_path)
        
        elif self.method == "PyRosetta_PRE": # Structure based --> (B, Lmax, 1) since PRE is a per-residue feature
            return self.PyRosetta_PRE(pdb_path, ag_chains)
        
        elif self.method == "Biophysics": # Sequence based --> (B, N_chains+breaks, 11) since biophysical features are global sequence properties (could technically repeat across seq len)
            return self.Biophysics(SEQUENCES)
        else:
            raise ValueError("Invalid embedding method")
    
    def One_hot(self, sequences):
        """
        Simple one-hot encoding representations of the Antigen sequence.
        Args: 
            sequences (dict): Given an Antigen pdb, it is formatted as {"chain": Seq, "chain": Seq, ...}
        Returns:
            torch.Tensor: A tensor of shape (1, max_sequence_length, 20) [batch size is always 1 antigen at a time]
        """

        VOCAB = "|ACDEFGHIKLMNPQRSTVWY"
        full_seq = "|".join(str(seq) for seq in sequences.values()) 

        ENCODE_MATRIX = torch.zeros((len(full_seq), len(VOCAB))) # initialize to all 0s (PAD later)
        for i, aa in enumerate(full_seq):
            if aa in VOCAB: # Just in case there are non-standard AAS 
                ENCODE_MATRIX[i, VOCAB.index(aa)] = 1.0
        
        return ENCODE_MATRIX.unsqueeze(0)

    def ESM2(self, sequences): # https://github.com/facebookresearch/esm?tab=readme-ov-file#quickstart
        """
        Generates encodings using the ESM2 model from Facebook Research. Embeddings are taken from the final layer of the model.
        The default model is esm2_t33_650M_UR50D, which has 33 layers and 650 million parameters. Hdims are: 
        15B: 5120, 3B: 2560, 650M: 1280, 150M: 640, 35M: 480, 8M: 320
        Args: 
            sequences (list of str): List of antigen sequences.
        Returns:
            torch.Tensor: A tensor of shape (num_structures, Lmax+2, 1280)
        """

        if self.method=='ESM2_t48_15B':
            model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
            last_layer = 48
        elif self.method=='ESM2_t36_3B':
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            last_layer = 36
        elif self.method=='ESM2_t33_650M':
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            last_layer = 33
        elif self.method=='ESM2_t30_150M':
            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            last_layer = 30
        elif self.method=='ESM2_t12_35M':
            model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            last_layer = 12
        elif self.method=='ESM2_t6_8M':
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            last_layer = 6
        else:
            print('No specified ESM2 model, defaulting to esm2_t33_650M_UR50D')
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            last_layer = 33


        batch_converter = alphabet.get_batch_converter()
        model.to(self.device).eval() # disables dropout for deterministic results

        data = [(chain, seq) for chain, seq in zip(sequences.keys(), sequences.values())] # ESM2 format: list of tuples (protein_id, sequence)
        lens = [len(s) for s in sequences.values()]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = model(
                batch_tokens.to(self.device),
                repr_layers=[last_layer],
                return_contacts=False
            ) # Padding shouldn't matter because ESM SHOULD have its attention mask ignoring the pad tokens

            # Keep all <cls> tokens from each sequence, but remove the ending one
            embeddings = results["representations"][last_layer][:,:-1,:] # Keep all <cls> tokens, Remove special tokens at end
            l_of_embs = [emb[:l+1, :] for emb, l in zip(embeddings, lens)]
            
        return torch.cat(l_of_embs, dim=0)[1:,:].unsqueeze(0) # 1 x Lsum x V (remove the first CLS, so it is treated as a sep chain)

    def ESMIF(self, structures): # https://colab.research.google.com/github/facebookresearch/esm/blob/master/examples/inverse_folding/notebook_multichain.ipynb#scrollTo=99d74757
        """
        Generates encodings using the ESMIF model from Facebook Research. Embeddings are taken from the final layer of the model.
        Args: 
            sequences (list of str): List of antigen sequences.
        Returns:
            torch.Tensor: A tensor of shape (num_structures, Lmax+2, Hdim)
        """
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        batch_converter = alphabet.get_batch_converter()
        model.eval() # disables dropout for deterministic results

        coords, native_seq = esm.inverse_folding.util.load_coords(structures[0]) # Just load the first structure for now, will need to modify to handle multiple structures
        print(coords)

        data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)] # ESM2 format: list of tuples (protein_id, sequence)
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        return token_representations

    def ProteinMPNN(self, structures, mpnn_model, noise, bb_perturbation):
        encoder = ProteinMPNN_Encoder(
            model=mpnn_model,
            noise=noise,
            bb_perturbation=bb_perturbation
        ).to(self.device)

        EMBEDDINGS = []
        for pdb_path in structures:
            h_V = encoder.encode(pdb_path, ag_chains='A')

            EMBEDDINGS.append(h_V[0]) # Squeeze out batch dimension (maybe do this first for memory efficiency w padding?)
            padding = torch.zeros(self.L_MAX+2 - EMBEDDINGS[-1].shape[0], EMBEDDINGS[-1].shape[1]).to(self.device) # B x L x H -> L x H
            EMBEDDINGS[-1] = torch.cat((EMBEDDINGS[-1], padding), dim=0) # Pad to L_MAX

        return torch.stack(EMBEDDINGS) # Shape: (num_structures, Lmax+2, 1280)

    def AFM(self, structures):
        pass

    def ESM3(self, structures):
        """
        NOTE: WILL NEED A SEPARATE ENVIRONMENT FOR THIS SINCE BOTH ESM2 AND ESM3 ARE INITIATED USING "IMPORT ESM"
        Args: 
            structures (list of str): List of paths to antigen PDB files. ESM3 will use sequence + structure from the pdb to generate embeddings.
        Returns:
            torch.Tensor: A tensor of shape (num_structures, Lmax+2, 1536)

        """
        import torch
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, LogitsConfig, ESM3InferenceClient
        from esm.utils.structure.protein_chain import ProteinChain

        import warnings
        warnings.filterwarnings("ignore", category=UserWarning) # Get rid of pesky ""

        # Load model
        model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cpu")
        model = model.float() # Fix dtype mismatch between model and input data, which is in float32. ESM3 model parameters are in float16 by default, so we need to convert them to float32 for compatibility with the input data.

        # Load PDB
        EMBEDDINGS = []
        for structure in structures:
            protein_chain = ProteinChain.from_pdb(structure, chain_id="A")
            sequence = protein_chain.sequence

            coords = torch.tensor(protein_chain.atom37_positions, dtype=torch.float32) # Each residue is represented by 37 possible atom positions

            protein = ESMProtein(sequence=sequence, coordinates=coords)
            protein_tensor = model.encode(protein)

            # Get embeddings
            output = model.logits(
                protein_tensor,
                LogitsConfig(
                    sequence=True,
                    structure=True,
                    return_embeddings=True
                )
            )

            EMBEDDINGS.append(output.embeddings[0]) # Squeeze out batch dimension 
            padding = torch.zeros(self.L_MAX+2 - output.embeddings.shape[1], output.embeddings.shape[2]) # B x L x H -> L x H
            EMBEDDINGS[-1] = torch.cat((EMBEDDINGS[-1], padding), dim=0) # Pad to L_MAX

        return torch.stack(EMBEDDINGS) # Shape: (num_structures, Lmax+2, 1536)

    def PyRosetta_PRE(self, structure, ag_chains):
        """
        NOTE: WILL NEED A SEPARATE ENVIRONMENT FOR THIS SINCE PYROSETTA MAY NOT BE COMPATIBLE WITH THE OTHER PACKAGES
        Uses PyRosetta to get a per residue energy (PRE) score for each residue in the antigen. 

        Args: 
            structures (list of str): List of paths to antigen PDB files.
        Returns:
            torch.Tensor: A tensor of shape (num_structures, Lmax, 1) since PRE is a per-residue feature.
        """
        #from pyrosetta.toolbox import cleanATOM

        #cleanATOM(structure)
        #clean_path = structure.replace(".pdb", ".clean.pdb")
        #pyrosetta.init("-mute all", silent=True)
        pyrosetta.init("-mute all -ignore_unrecognized_res 1 -load_PDB_components false", silent=True)

        score_fxn = pyrosetta.get_score_function(True)
        #per_res_sasa_metric = pyrosetta.rosetta.core.simple_metrics.per_residue_metrics.PerResidueSasaMetric()
        per_res_energy_metric = pyrosetta.rosetta.core.simple_metrics.per_residue_metrics.PerResidueEnergyMetric()

        pose = pyrosetta.pose_from_pdb(structure)
        pre = per_res_energy_metric.calculate(pose)
        chain_energies = [
            energy for res_idx, energy in pre.items()
            if pose.pdb_info().chain(res_idx) in ag_chains
        ]

        #pre = list(per_res_energy_metric.calculate(pose).values()) # List of PRE values for each residue in the antigen
        #pre += [0] * (self.L_MAX - len(pre)) # Pad to L_MAX with 0s, since PRE is a per-residue feature
        #EMBEDDINGS.append(torch.tensor(pre).unsqueeze(1)) # Shape: Lmax x 1

        #return torch.stack(EMBEDDINGS).to(self.device)

    def Biophysics(self, sequences):
        """
        Generates encodings using all sequence-based biophysical features from Biopython. The methods included are:
        molecular weight, aromaticity (% [Phe+Trp+Tyr]), instability index, gravy hydrophobicity, isoelectric point, 
        charge at pH 7.2, % helix, % turns, % sheet, and molar extinction coefficient for Cys and Cys-Cys bonds.
        Args: 
            sequences (dict): Given an Antigen pdb, it is formatted as {"chain": Seq, "chain": Seq, ...}
        Returns:
            torch.Tensor: A tensor of shape (num_structures, 1, 11) 
            NOTE: Biophysical features are global properties of the sequence, so we can just repeat them across the sequence length dimension.
        """
        
        PROPERTIES = ["aromaticity", "instability_index", "gravy", "isoelectric_point", "charge_at_pH", "secondary_structure_fraction", "flexibility"]
        CH_BRK = [0]*11 # Biophysical properties
        PRECISION = 4
        
        ALL_MATRICES = []
        for i, seq in enumerate(sequences.values()):
            X = ProteinAnalysis(str(seq))
            properties = [
                X.molecular_weight(), # Keep, Ags can be anywhere from 10s to 1000s of AAs so this may fluctuate and provides insight into size
                X.aromaticity(), # Keep, it tells how bulky,greasy the antigen is, which can affect binding
                X.instability_index(), # Keep, <40 is stable, can provide insight into how well the antigen will fold and maintain its structure
                # np.mean(X.flexibility()), # Only works with Ags that are at least 10 AAs [NEED TO REREAD THE PAPER FIRST]
                X.gravy(), # Keep, it tells how hydrophobic the antigen is, which can affect binding
                X.isoelectric_point(), # Keep, it can affect how the antigen interacts with the environment and the antibody
                X.charge_at_pH(7.2), # Close to physiological pH, which is relevant for antibody-antigen interactions
            ] # aliphatic index

            properties += list(X.secondary_structure_fraction()) # Helix, Turn, Sheet fractions (CRAZY TYPO HERE)
            properties += list(X.molar_extinction_coefficient()) # for Cys, and Cys-Cys bond

            properties = [round(prop, PRECISION) for prop in properties] # Round all properties to 4 decimal places for consistency
            ALL_MATRICES.append(properties)

            if len(sequences)>1 and (i+1)<len(sequences):
                ALL_MATRICES.append(CH_BRK)
        
        embeddings = torch.tensor(ALL_MATRICES)
        return embeddings.unsqueeze(0)
        
        
# ESM-IF, AF-M, ESM3 (DONE, BUT NEED SEP ENVIRONMENT)
embedder = Ag_embeddings(method="One_hot")
#structure = "Penta_Alanine_Antigen.pdb"
structure = '/weka/scratch/jgray21/dvincen9/TRAINING/MANGO/SAbDAb/structures/8hnm.pdb'
embeddings = embedder.embed(structure)
#print(embeddings.shape)