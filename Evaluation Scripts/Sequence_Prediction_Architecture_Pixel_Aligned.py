import torch
import torch.nn as nn
import math
from unet_sdf_pixel import *
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,n_heads=8):
        super().__init__()

        # cross attention SDF
        self.cross_attn_global = nn.MultiheadAttention(d_model, n_heads,  batch_first = True)
        self.norm1_global = nn.LayerNorm(d_model)

        # cross attention SDF Local
        self.cross_attn_local = nn.MultiheadAttention(d_model, n_heads,  batch_first = True)
        self.norm1_local = nn.LayerNorm(d_model)

        # self attention
        self.self_attn_vertices = nn.MultiheadAttention(d_model, n_heads,  batch_first = True)
        self.norm2_vertices = nn.LayerNorm(d_model)

        self.gate_fc = nn.Linear(d_model * 2, d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.norm3 = nn.LayerNorm(d_model) 

    def forward(self, vertex_embedding, global_features, pixel_features):

        vertex_attn_out, vertex_attn_output_weights = self.self_attn_vertices(vertex_embedding, vertex_embedding, vertex_embedding)
        vertex_embedding_out = self.norm2_vertices(vertex_embedding + vertex_attn_out)

        global_attn_out, _ = self.cross_attn_global(vertex_embedding_out, global_features, global_features)
        global_attn_out = self.norm1_global(vertex_embedding_out + global_attn_out)

        local_attn_out, _ = self.cross_attn_local(vertex_embedding_out, pixel_features, pixel_features)
        local_attn_out = self.norm1_local(vertex_embedding_out + local_attn_out)

        fusion_input = torch.cat([local_attn_out, global_attn_out], dim=-1)
        gate = torch.sigmoid(self.gate_fc(fusion_input))  # shape: (B, N, D)
        vertex_embedding_out = gate * local_attn_out + (1 - gate) * global_attn_out

        vertex_embedding_out_inter =  nn.ReLU()(self.linear1(vertex_embedding_out))
        vertex_embedding_out = self.norm3(vertex_embedding_out + vertex_embedding_out_inter)
        
        return vertex_embedding_out



def positionalencoding_xy_batched(xy, d_model):
    """
    Sinusoidal positional encoding for batched (x, y) point sequences.

    Args:
        xy (Tensor): shape [B, N, 2], where each entry is (x, y)
        d_model (int): total embedding size, must be divisible by 4

    Returns:
        Tensor: [B, N, d_model] positional embeddings
    """
    if d_model % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4, got {d_model}")

    B, N, _ = xy.shape
    d_half = d_model // 2
    div_term = torch.exp(torch.arange(0., d_half, 2, device=xy.device) * -(math.log(10000.0) / d_half))  # [d_half/2]
    # xy = (xy / 2.2) * (2 * math.pi)
    pos_x = xy[:, :, 0].unsqueeze(-1)  # [B, N, 1]
    pos_y = xy[:, :, 1].unsqueeze(-1)  # [B, N, 1]

    pe_x = torch.zeros(B, N, d_half, device=xy.device)
    pe_y = torch.zeros(B, N, d_half, device=xy.device)

    pe_x[:, :, 0::2] = torch.sin(pos_x * div_term)
    pe_x[:, :, 1::2] = torch.cos(pos_x * div_term)

    pe_y[:, :, 0::2] = torch.sin(pos_y * div_term)
    pe_y[:, :, 1::2] = torch.cos(pos_y * div_term)

    return torch.cat([pe_x, pe_y], dim=-1)  # [B, N, d_model]

def positionalencoding_index_batched(N, d_model):
    """
    Sinusoidal positional encoding for batched index sequences.

    Args:
        N (int): Number of tokens or sequence length
        d_model (int): Total embedding size, must be divisible by 4

    Returns:
        Tensor: [N, d_model] positional embeddings for indices
    """
    if d_model % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4, got {d_model}")

    # Create indices as a tensor (0 to N-1)
    indices = torch.arange(0, N, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).unsqueeze(-1)  # [N, 1]

    # Half of the model dimension for sine/cosine
    d_half = d_model // 2

    div_term = torch.exp(torch.arange(0., d_half, device=indices.device) * -(math.log(10000.0) / d_half))  # [d_half/2]
    # Apply sine and cosine transformations
    pos = indices * div_term  # [N, d_half]]
    pe = torch.zeros(N, d_model, device=indices.device)  # [N, d_model]
    pe[:, 0::2] = torch.sin(pos)  # Even indices (sine)
    pe[:, 1::2] = torch.cos(pos)  # Odd indices (cosine)

    return pe  # [N, d_model]

def bilinear_sample_sdf_features(sdf_features, vertices):
    """
    sdf_features: (B, C, H, W)
    vertices: (B, N, 2)  # [x, y] in range [0, 2.2]
    Returns: (B, N, C)
    """
    B, C, H, W = sdf_features.shape
    device = sdf_features.device

    # Normalize to [-1, 1]
    # x: width direction → divide by (W-1)*resolution
    # y: height direction → divide by (H-1)*resolution
    x = ((vertices[..., 0] / 2.2) * 2) - 1  # (B, N)
    y = ((vertices[..., 1] / 2.2) * 2) - 1  # (B, N)

    # Create grid in (y, x) order
    grid = torch.stack((x,y), dim=-1)  # (B, N, 2)
    grid = grid.unsqueeze(2)  # (B, N, 1, 2)

    # grid_sample expects (B, H_out, W_out, 2), so this is (B, N, 1, 2)
    sampled = F.grid_sample(sdf_features, grid, align_corners=False, mode='bilinear', padding_mode="border")  # (B, C, N, 1)
    # Reshape to (B, N, C)
    sampled = sampled.squeeze(-1).permute(0, 2, 1)

    return sampled

class PolygonVisibilityTransformerDecoderIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.vis_embedding = nn.Sequential(
        nn.Linear(325, 128, bias=True),
        nn.ReLU(),
         )
        self.vertex_embedding_net = nn.Linear(256,256)
        self.sdf_embedding_net = nn.Linear(512,128)
        self.sdf_embedder = UNet_SDF_Intermediate(n_channels = 1, n_classes = 1)
        self.transformer_decoder_1 = TransformerDecoderLayer(d_model=256, d_ffn=256,n_heads=1)
        self.transformer_decoder_2 = TransformerDecoderLayer(d_model=256, d_ffn=256,n_heads=1)
        self.transformer_decoder_3 = TransformerDecoderLayer(d_model=256, d_ffn=256,n_heads=1)
        self.mlp_out = nn.Linear(256, 2)
        
    def forward(self, polygon_init, visibility, sdf):
        query_embed = positionalencoding_xy_batched(polygon_init, 256)
        index_embed = positionalencoding_index_batched(25,256)
        index_embed = index_embed.unsqueeze(0).expand(visibility.shape[0], -1, -1)
        vertex_embedding = nn.ReLU()(self.vertex_embedding_net(query_embed + index_embed))
        sdf_embedding_pixel, sdf_embedding_global = self.sdf_embedder(sdf)
        sdf_embedding_global = self.sdf_embedding_net(sdf_embedding_global)
        intrepolated_features = bilinear_sample_sdf_features(sdf_embedding_pixel, polygon_init)
        vis_emb = self.vis_embedding(visibility)
        vis_emb_local = vis_emb.unsqueeze(1).expand(-1, intrepolated_features.shape[1], -1)
        intrepolated_features = torch.concat([intrepolated_features, vis_emb_local], axis = -1)

        vis_emb_global = vis_emb.unsqueeze(1).expand(-1, sdf_embedding_global.shape[1], -1)
        sdf_embedding_global = torch.concat([sdf_embedding_global, vis_emb_global], axis = -1)
        vertex_embedding = self.transformer_decoder_1(vertex_embedding, sdf_embedding_global, intrepolated_features)
        vertex_embedding = self.transformer_decoder_2(vertex_embedding, sdf_embedding_global, intrepolated_features)
        vertex_embedding = self.transformer_decoder_3(vertex_embedding, sdf_embedding_global, intrepolated_features)
        vertex_embedding_out = self.mlp_out(vertex_embedding)
        return vertex_embedding_out