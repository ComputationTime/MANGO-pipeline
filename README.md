# MANGO
Official repository for MANGO - a multi-modal, autoregressive antibody-specific language model that generates and scores conditioned on antigen context, as described in [PAPER](https://www.google.com)

![Model Logo](mango/utils/images/MANGO_2.png)

## Setup 
To use MANGO, install via pip: 
```bash
pip install gray_mango
```

Alternatively, you can clone this repository and install locally
```bash
git clone git@github.com:Donnievin/MANGO.git
```

```bash
cd MANGO
bash install_mango.sh
```

## Usage

### Generation using only Antigen structure
To generate 100 unpaired sequences using MANGO conditioned on Antigen structure, use the following code:

```python
from mango import MANGO
from mango.utils import Penta_Alanine_Antigen.pdb

model = MANGO(ag_representation='One_hot')

antigen_pdb_path = 'test.pdb'
chains_to_generate = 'H'
num_to_generate = 100

# antigen_pdb_path, prompt_sequence=None, chains_to_generate='H', num_to_generate=10, top_p=1, temperature=1, ag_cold_spots=None, ag_hot_spots=None,
sequences = model.generate(
    antigen_pdb_path=antigen_pdb_path, NA_seqs,
    chains_to_generate=chains_to_generate,
    num_to_generate=num_to_generate,
    )

print(sequences)
```

`Note`: Currently available ag_representations are: `One_hot`, ``, ``,


### Generation using Antigen structure and a complement chain
BANANA provides rich antibody embeddings (via PyTorch tensors) of shape `[Nseqs x 300 x Hdim]` that consider the amino acid sequence, the DNA sequence, and the organism of origin. To gather these embeddings, use the following code:  

```python
from mango import MANGO

model = MANGO(ag_representation='One_hot')

antigen_pdb_path = 'test.pdb'
prompt_sequence = 'EVQLVESGGGLVQPGGSLRLSCAASGFNIKEYYMHWVRQAPGKGLEWVGLIDPEQGNTIYDPKFQDRATISADNSKNTAYLQMNSLRAEDTAVYYCARDTAAYFDYWGQGTLVTVS'
chains_to_generate = 'L'
num_to_generate = 100

# antigen_pdb_path, prompt_sequence=None, chains_to_generate='H', num_to_generate=10, top_p=1, temperature=1, ag_cold_spots=None, ag_hot_spots=None,
sequences = model.generate(
    antigen_pdb_path=antigen_pdb_path, NA_seqs,
    prompt_sequence=prompt_sequence
    chains_to_generate=chains_to_generate,
    num_to_generate=num_to_generate,
    )

print(sequences)
```


### Sequence scoring using Antigen structure
To forward translate Amino Acid sequences into the corresponding DNA sequence, use the following code: 

```python
from mango import MANGO

model = MANGO(ag_representation='One_hot')

antigen_pdb_path = 'test.pdb'
chains_to_score = 'L'

# antigen_pdb_path, sequences=None, chains_to_score='H', ag_cold_spots=None, ag_hot_spots=None,
sequences = model.log_likelihood(
    antigen_pdb_path=antigen_pdb_path, NA_seqs,
    sequences=sequences,
    chains_to_generate=chains_to_generate,
    num_to_generate=num_to_generate,
    )

print(sequences)
```


## Citing this work
```bibtex
@article{vincent2026MANGO,
    title = {Naive versus Learned versus biophysical embeddings: The effect of antigen representation on generating developabe antibodies},
    author = {Vincet Jr, Donovan and Fortes, Andre and Sanghvi, Tanay and Gray, Jeffrey J},
    journal = {???},
    year= {2026}
}
```