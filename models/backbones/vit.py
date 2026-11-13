import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BackboneOutput


# ─────────────────────────────────────────────────────────────────────────────
# 1. PATCH EMBEDDING
#    This is ViT's "stem" — the equivalent of your ResNet conv1 + maxpool.
#    But instead of learning spatial feature maps, it carves the image into a
#    sequence of flat patch vectors.
#
#    What you need to implement in __init__:
#      - A projection layer that maps each (C, P, P) patch → a D-dim vector.
#        Hint: a single Conv2d with kernel_size=patch_size, stride=patch_size
#        does the flattening AND the linear projection in one shot.
#      - A [CLS] token: a learnable parameter of shape (1, 1, D). During
#        forward it is prepended to the patch sequence, giving the network an
#        "aggregation slot" that doesn't correspond to any patch.
#      - Positional embeddings: a learnable parameter of shape
#        (1, num_patches + 1, D). Without this, the transformer has no idea
#        where patch 0 is vs patch 99 — it's permutation-invariant by design.
#
#    What you need to implement in forward(x):
#      1. Project the image into patch tokens                → (B, N, D)
#      2. Expand and prepend the CLS token                  → (B, N+1, D)
#      3. Add positional embeddings (just a +, they're the same shape)
#      4. Return the sequence                               → (B, N+1, D)
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
 
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 use_cls_token=True, pos_emb="learned"):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.use_cls_token = use_cls_token
        seq_len = self.num_patches + int(use_cls_token)  # N+1 with CLS, N without
 
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
 
        if use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
 
        if pos_emb == "learned":
            self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        else: 
            self.register_buffer("pos_emb", torch.zeros(1, seq_len, embed_dim))  # "none" — register as buffer so it moves with the model to GPU
 
    def forward(self, x):
        B = x.shape[0]
 
        x = self.projection(x).flatten(2).transpose(1, 2)  # (B, N, D)
 
        if self.use_cls_token:
            cls = self.cls.expand(B, -1, -1)               # (1,1,D) → (B,1,D)
            x = torch.cat([cls, x], dim=1)                 # (B, N+1, D)
 
        x = x + self.pos_emb                               # same shape, just add
        return x

# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTI-HEAD SELF-ATTENTION (MHSA)
#    The core operation. Every token attends to every other token (including
#    itself and the CLS token).
#
#    The mechanism:
#      - Project each token into three vectors: Query (Q), Key (K), Value (V).
#        These three come from three separate linear layers (or one big one
#        you split), all of size D → D.
#      - Attention scores: A = softmax(Q @ K.T / sqrt(head_dim))
#        The sqrt(head_dim) scaling prevents the dot products from growing so
#        large that softmax saturates into near-zero gradients.
#      - Weighted sum: out = A @ V
#      - "Multi-head" means you do this in parallel H times on D/H-dim
#        subspaces, then concatenate and project back to D.
#
#    Layers you need:
#      - nn.Linear(embed_dim, embed_dim * 3)  ← one fused QKV projection
#        (or three separate linears — same thing)
#      - nn.Linear(embed_dim, embed_dim)       ← output projection
#      - nn.Dropout                            ← applied to attention weights
#
#    Forward steps:
#      1. Compute Q, K, V and reshape to (B, H, N, head_dim)
#      2. Compute scaled dot-product scores
#      3. Softmax + dropout
#      4. Weighted sum of V
#      5. Reshape back to (B, N, D) and apply output projection
#
#    Tip: torch.nn.functional.scaled_dot_product_attention exists and handles
#    steps 2-4 for you — but implement it manually first to understand it.
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim=768, num_heads=12, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.embed_dim = embed_dim

        # TODO: qkv projection, output projection, dropouts
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        # TODO: implement attention
        B, N, D = x.shape

        # 1. Project and split Q, K, V
        qkv = self.qkv_proj(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)           # each: (B, H, N, head_dim)

        # 2. Scaled dot-product
        # attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        # attn = attn.softmax(dim=-1)
        # attn = self.dropout(attn)

        # 3. Weighted sum, reshape back
        # x = (attn @ v)                              # (B, H, N, head_dim)

        # FlashAttention-powered Scaled Dot-Product Attention
        # PyTorch natively handles scaling, softmax, dropout, and matrix multiplication
        # while leveraging FlashAttention (or memory-efficient attention) under the hood.
        x = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=None, 
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False, # Vision Transformers use bidirectional attention, not causal masking
            scale=self.scale
        ) # Output shape: (B, H, N, head_dim)

        x = x.transpose(1, 2).reshape(B, N, D)      # (B, N, D)
        x = self.out_proj(x)
        return self.proj_drop(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MLP BLOCK
#    A simple two-layer feedforward network applied token-wise after attention.
#    Nothing complicated here — the only new thing vs ResNet is:
#      - The hidden dimension is typically 4 × embed_dim (an "expansion" just
#        like your BottleneckBlock, but in the other direction).
#      - The activation is GELU, not ReLU. (Smoother, works better in
#        transformers — worth reading about why.)
#
#    Layers: Linear → GELU → Dropout → Linear → Dropout
# ─────────────────────────────────────────────────────────────────────────────

class MLPBlock(nn.Module):

    def __init__(self, embed_dim=768, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        # TODO: two linears, GELU, dropout
        self.layer1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        # TODO: linear → gelu → drop → linear → drop
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRANSFORMER ENCODER BLOCK
#    Wraps MHSA + MLP with:
#      - LayerNorm BEFORE each sub-layer (this is "Pre-LN", the modern default)
#      - A residual connection AROUND each sub-layer (same idea as your
#        identity shortcut, but simpler — no downsampling ever needed here
#        because the sequence length and embed_dim never change!)
#
#    Why LayerNorm and not BatchNorm?
#      BatchNorm normalises across the batch dimension — it needs a large batch
#      to estimate stable statistics, and doesn't work well on variable-length
#      sequences. LayerNorm normalises across the feature dimension for each
#      token independently, making it sequence-length and batch-size agnostic.
#
#    Structure (Pre-LN variant):
#      x = x + MHSA(LayerNorm(x))
#      x = x + MLP(LayerNorm(x))
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0,
                 attn_drop=0.0, drop=0.0):
        super().__init__()

        # TODO: two LayerNorms, one MHSA, one MLPBlock
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop, drop)
        self.mlp = MLPBlock(embed_dim, mlp_ratio, drop)

    def forward(self, x):
        # TODO: pre-LN residual for attention, then for MLP
        x += self.mhsa(self.layernorm1(x))
        x += self.mlp(self.layernorm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. VIT BACKBONE
#    Stacks everything together. Compare to your ResNet.__init__:
#      ResNet:  stem → layer1 → layer2 → layer3 → layer4 → avgpool
#      ViT:     patch_embed → (CLS + pos_embed inside embed) →
#               [TransformerEncoderBlock × depth] → LayerNorm → extract CLS
#
#    What you need in __init__:
#      - PatchEmbedding
#      - nn.Dropout for embedding output (stochastic depth / drop_rate)
#      - A list of `depth` TransformerEncoderBlocks — use nn.Sequential or
#        nn.ModuleList (think about which lets you iterate vs index)
#      - A final LayerNorm (applied after all blocks, before extracting CLS)
#      - self.out_features = embed_dim  ← same convention as your ResNet
#
#    What you need in forward(x):
#      1. Patch embed (returns (B, N+1, D))
#      2. Apply dropout
#      3. Pass through all encoder blocks
#      4. Apply final LayerNorm
#      5. Extract the CLS token: output[:, 0]   ← index 0, the one we prepended
#      6. Return a BackboneOutput (or just a dict/tuple for now)
#
#    The freeze / unfreeze helpers from your ResNet copy over unchanged.
# ─────────────────────────────────────────────────────────────────────────────

class ViT(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, attn_drop=0.0, drop=0.0,
                 use_cls_token=True, pos_emb="learned"):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.out_features = embed_dim

        # TODO: patch_embed, dropout, blocks (×depth), norm
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, use_cls_token, pos_emb)
        self.dropout = nn.Dropout(drop)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio,attn_drop, drop) for _ in range(depth)]) 
        self.layernorm = nn.LayerNorm(embed_dim)
        
        self.out_features = embed_dim

    def forward(self, x):
        # TODO: embed → dropout → blocks → norm → CLS → BackboneOutput
        x = self.patch_embed(x)
        x = self.dropout(x)
        hs = []
        for block in self.blocks:
            x = block(x)
            hs.append(x)

        x = self.layernorm(x)
        
        B, N, D = x.shape
        H = W = int(self.patch_embed.num_patches ** 0.5)

        if self.use_cls_token: 
            cls = x[:, 0]                                                   # (B, D)
            feature_map = x[:, 1:].transpose(1,2).unflatten(2, (H, W))      # (B, N, D) -> (B, D, H, W)
        else:
            cls = x.mean(dim=1)                                             # (B, N, D) -> (B, D) 
            feature_map = x.transpose(1,2).unflatten(2, (H, W)) 

        return BackboneOutput(
            feature_map=feature_map,
            pooled=cls,
            stages=hs)


    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
    
    def freeze_stages(self, stages):
        for stage in stages:
            module = getattr(self, stage)

            for p in module.parameters():
                p.requires_grad = False


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions
#
# Tiny and Small come from the DeiT paper (Touvron et al., 2021), which
# introduced lighter variants for training efficiently without the massive
# JFT-300M dataset the original ViT required. They are now standard.
#
#   ViT-Ti (Tiny):  embed=192,  heads=3,  depth=12  ~5.7M params
#   ViT-S  (Small): embed=384,  heads=6,  depth=12  ~22M  params
#   ViT-B  (Base):  embed=768,  heads=12, depth=12  ~86M  params
#   ViT-L  (Large): embed=1024, heads=16, depth=24  ~307M params
#
# Notice the pattern: embed_dim / num_heads = 64 in every config.
# That constant head_dim of 64 is a deliberate design choice — it keeps
# the attention scaling factor (sqrt(head_dim)) fixed across model sizes,
# so you only scale width and depth, not the attention geometry.
# ──────────────────────────────────────────────────────────────────────

def vit_tiny_16(img_size=224):
    return ViT(img_size=img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3)
 
def vit_small_16(img_size=224):
    return ViT(img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6)

def vit_base_16(img_size=224):
    return ViT(img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12)

def vit_large_16(img_size=224):
    return ViT(img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16)