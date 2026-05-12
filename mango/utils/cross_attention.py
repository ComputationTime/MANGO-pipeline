import math
import einops
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


#######################################################################################
def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )
#######################################################################################



############ IGNORE FOR NOW ############
class Gated_MultiHead_Cross_Attention(nn.Module):
    '''
    Idea: Want the query to come from the AbLM, and attends to the Ag representation. (LM pulls in graph information where it needs it).
    However, we let the model learn how much the MPNN embeddings should contribute to the meaning
    Q comes from LM: (Batch x N x 256)
    K/V comes from MPNN: (Batch x M x 128)
    Cross-Attention: (Batch x N x M)
    '''
    def __init__(self, d_LM, n_heads, d_Ag_rep, bias=False, qk_layernorm=True):
        super().__init__()

        self.d_model = d_LM
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads

        self.W_q = nn.Sequential( 
            nn.LayerNorm(d_LM), nn.Linear(d_LM, d_LM, bias=bias)
        ) 
        self.W_kv =  nn.Sequential( 
            nn.LayerNorm(d_Ag_rep), nn.Linear(d_Ag_rep, d_LM*2, bias=bias) # Faster + Couples K and V 
        ) # nn.Linear(d_Ag_rep, d_LM * 2, bias=bias)

        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(self.d_model, bias=bias)
            self.k_ln = nn.LayerNorm(self.d_model, bias=bias)
        else: 
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()
    
    def forward(self, x, h_V, seq_id=None):
        query_BLD = self.W_q(x) # Query comes from the language model (Batch x Len x d_LM)

        kv_BLD2 = self.W_kv(h_V) # Key and Value come from h_V (ProteinMPNN encoder)
        key_BLD, value_BLD = torch.chunk(kv_BLD2, 2, dim=-1)
        
        # QK-norm (from the Dehghani et al. "ViT-22B" paper) is meant to stabilize the logit scale
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )

        # No rotary positions applied, these live in different domains with different positional encodings
        # Also, positional encodings are already applied from their counterparts

        # Reshape to multi-head format [H_dim (h d) -> Attention Heads (h)]
        reshaper = functools.partial(
            einops.rearrange, pattern = "b s (h d) -> b h s d", h=self.n_heads
        )

        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        if seq_id is not None:
            # Where True, enable participation in attention.
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)

            #mask_BNM = seq_id_q.unsqueeze(-1) == seq_id_k.unsqueeze(-2)  # (B, N, M)
            #mask_BHNM = mask_BNM.unsqueeze(1)  # (B, 1, N, M)

            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL
            )
        else:
            # Shortcut, if we don't use attention biases then torch
            # will autoselect flashattention as the implementation
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD
            )

        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")

        return self.out_proj(context_BLD)

    def init2(self, hidden_dim, n_heads):
        super().__init__() 
        # BASED ON FLAMINGO ARCHITECTURE 
        # Probably best for fine-tuning GPT2, gate let the pretrained weights stabilize before structural context kicks in 
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads)
        self.gate = nn.Parameter(torch.tensor(0.0))  # starts at 0, learned
        self.ln = nn.LayerNorm(hidden_dim)

    def forward2(self, x, context):
        attn_out = self.cross_attn(self.ln(x), context, context)[0]
        return x + self.gate.tanh() * attn_out  # gate controls contribution
############ IGNORE FOR NOW ############



#######################################################################################
class MultiHead_Cross_Attention(nn.Module):
    '''
    Idea: Want the query to come from the AbLM, and attends to the Ag representation. (LM pulls in graph information where it needs it)
    Q comes from LM: (Batch x L_ab x 1024)
    K/V comes from MPNN: (Batch x L_ag x H_dim)
    Cross-Attention: (Batch x L_ab x L_ag)
    '''
    def __init__(self, d_LM, n_heads, d_Ag_rep, bias=False, qk_layernorm=True):
        super().__init__()

        self.d_model = d_LM
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads

        self.W_q = nn.Sequential( 
            nn.LayerNorm(d_LM), nn.Linear(d_LM, d_LM, bias=bias)
        ) 
        self.W_kv =  nn.Sequential( 
            nn.LayerNorm(d_Ag_rep), nn.Linear(d_Ag_rep, d_LM*2, bias=bias) # Faster + Couples K and V 
        )

        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(self.d_model, bias=bias)
            self.k_ln = nn.LayerNorm(self.d_model, bias=bias) # normalize after projection
        else: 
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()
    
    def forward(self, x, h_V, seq_id=None):

        query_BLD = self.W_q(x) # Query comes from the language model --> (B, Lmax, 1024)

        kv_BLD2 = self.W_kv(h_V) # Key and Value come from Antigen encoder --> (B, Lmax, 2048)
        key_BLD, value_BLD = torch.chunk(kv_BLD2, 2, dim=-1)
        
        # QK-norm (from the Dehghani et al. "ViT-22B" paper) is meant to stabilize the logit scale
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )

        # Reshape to multi-head format [H_dim (h d) -> Attention Heads (h)]
        reshaper = functools.partial(
            einops.rearrange, pattern = "b s (h d) -> b h s d", h=self.n_heads
        )

        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )
        
        # Shortcut, if we don't use attention biases then torch  will autoselect flashattention as the implementation
        context_BHLD = F.scaled_dot_product_attention(
            query_BHLD, key_BHLD, value_BHLD
        )
        #print(context_BHLD.shape, "HERE IS THE ATTENTION")

        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")

        return self.out_proj(context_BLD)


class UnifiedTransformerBlock(nn.Module):
    """
    A unified transformer block that incorporates cross attention between different modalities. 
    For MANGO, this is LM and MPNN protein representations.

    Args:
        1. d_model (int): The dimensionality of the input and output features of the transformer block.
        2. n_heads (int): The number of attention heads in the multi-head attention mechanism. (width)
        3. d_Ag_rep (int): The dimensionality of the Hdim for the chosen antigen representation method.
        4. bias (bool): Whether to use an affine function (True) or linear transformation (False)
        5. expansion_ratio (float): 
        6. residue_scaling_factor (float): 
        7. qk_layernorm (bool): Apply layer norm after linear projection to q and k from MPNN.
        8. ffn_type (str): Feed forward network type (swiglu | gelu)
    """

    def __init__(self, d_model, n_heads, d_Ag_rep, bias=False, expansion_ratio = 4.0, 
                residue_scaling_factor = 1.0, qk_layernorm = True, ffn_type = "swiglu"):
        super().__init__()
    
        self.attn = MultiHead_Cross_Attention(
            d_LM=d_model, n_heads=n_heads, d_Ag_rep=d_Ag_rep, bias=bias, qk_layernorm=qk_layernorm
        )

        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        self.scaling_factor = residue_scaling_factor

    def forward(self, x, h_V, sequence_id=None) -> torch.Tensor:
        """
        Forward pass for the UnifiedTransformerBlock.

        Args:
            1. x (torch.Tensor[float]): Input transformer embeddings (from the LM)
            2. h_V (torch.Tensor[float]): Input structural embeddings (from the MPNN)
            3. sequence_id (torch.Tensor[int]): Tensor containing sequence IDs for each element in the batch, used for attention masking. ***

        Returns:
            x (torch.Tensor[float]): Output tensor after applying the transformer block operations.
        """

        r1 = self.attn(x, h_V, sequence_id)
        x = x + r1 / self.scaling_factor
        
        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3 # Keep this syntax to avoid inplace modification during autograd

        return x


class TransformerStack(nn.Module):
    """
    A stack of transformer blocks used in the MANGO model.

    Args:
        1. d_model (int): The dimensionality of the input and output feature vectors.
        2. n_heads (int): The number of attention heads.
        3. n_layers (int): The number of transformer blocks in the stack.
        4. scale_residue (bool, optional): Whether to scale the residue connections in each transformer block.
        5. 
    """

    def __init__(self, d_model, n_heads, n_layers, d_Ag_rep, bias=False, expansion_ratio = 8/3, 
                 scale_residue=True, qk_layernorm= True, ffn_type = "swiglu"):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    d_Ag_rep=d_Ag_rep,
                    bias=bias,
                    expansion_ratio=expansion_ratio,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    qk_layernorm=qk_layernorm,
                    ffn_type=ffn_type,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(self, x, h_V, cold_spots=None, hot_spots=None, get_hidden=False):
        """
        Forward pass of the TransformerStack.

        Args:
            1. x (torch.Tensor): The input LM tensor (batch_size, Ab_seq_len, d_model).
            2. h_V (torch.Tensor): The input MPNN tensor (batch_size, Ag_res_length, d_model)
            3. get_hidden (bool): Whether to return all hidden states or only the last hidden state
            4. cold_spots (list of ints): 
            5. hot_spots (list of ints): 

        Returns:
            1. torch.Tensor (batch_size, Ab_seq_len, d_model).
            2. If get_hidden: tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
        """

        #print('AbLM', x.shape) # 1,12,1024
        #print('Ag_rep', h_V.shape) # 1,5,20


        if cold_spots != None or hot_spots != None:
            pass

        hiddens = []
        for block in self.blocks: # Updates the query by some influence from the Key --> So output is same shape as AbLM
            x = block(x, h_V)
            hiddens.append(x)
            #print("Post cross atten", x.shape)

        if get_hidden:
            return self.norm(x), x, hiddens # post_norm, pre_norm, hidden_states
        else:
            return self.norm(x)
#######################################################################################