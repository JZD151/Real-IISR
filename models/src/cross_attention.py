import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_q, x_kv):
        """
        x_q: [n, seq_len_q, embed_dim]
        x_kv: [n, seq_len_kv, embed_dim]
        """
        N, L_q, _ = x_q.shape
        _, L_kv, _ = x_kv.shape

        Q = self.q_proj(x_q)
        K = self.k_proj(x_kv)
        V = self.v_proj(x_kv)

        Q = Q.view(N, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, L_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).reshape(N, L_q, self.embed_dim)

        output = self.out_proj(attn_output)
        return output, attn_weights
