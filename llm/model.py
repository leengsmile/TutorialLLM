import torch
import torch.nn as nn
import torch.nn.functional as F

"""
hot fix disap
"""
class LayerNorm(nn.Module):
    
    def __init__(self, n_embd: int, bias: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None
        self.eps = eps
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
    
class CasualSelfAttention(nn.Module):
    
    def  __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0., bias: bool = False) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        
        self.c_attn = nn.Linear(n_embd, n_embd*3, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size))
                                         .view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.n_embd//self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, -math.inf)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.c_proj(y))
        return y
        

class MLP(nn.Module):
    
    def __init__(self, n_embd: int, dropout: float = 0., bias: bool = False) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, n_embd*4, bias=bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(n_embd*4, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0., bias: bool = False) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CasualSelfAttention(n_embd, n_head, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)
    
    def __forward__(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, 
                 n_head: int, n_layer: int, dropout: float = 0., bias: bool = False) -> None:
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            blocks = nn.ModuleList([Block(n_embd, n_head, dropout, bias) for _ in range(n_layer)]),
            ln = LayerNorm(n_embd, bias=bias)
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.transformer.wpe.weight
        
        self.block_size = block_size
        
    def _init_weights(self, modlue):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None, reduce_loss: bool = True):
        
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), 
                                   ignore_index=-1, 
                                   reduction='mean' if reduce_loss else 'sum')
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else  idx[:, -self.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(probs, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -math.inf
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx
    