from transformers import DetrForObjectDetection
from timm import create_model
from torch import nn
import torch
from timm import create_model
from typing import List, Tuple, Optional
import math
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.nn import SiLU



def replace_batch_norm(m):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if isinstance(target_attr, nn.BatchNorm2d):
            frozen = DetrFrozenBatchNorm2d(target_attr.num_features)
            bn = getattr(m, attr_str)
            frozen.weight.data.copy_(bn.weight)
            frozen.bias.data.copy_(bn.bias)
            frozen.running_mean.data.copy_(bn.running_mean)
            frozen.running_var.data.copy_(bn.running_var)
            setattr(m, attr_str, frozen)
    for n, ch in m.named_children():
        replace_batch_norm(ch)


class DetrFrozenBatchNorm2d(nn.Module):
    # TODO explain what this does
    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, 
                              state_dict, 
                              prefix, 
                              local_metadata, 
                              strict, 
                              missing_keys, 
                              unexpected_keys, 
                              error_msgs):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    def forward(self, x):
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)   
        eps = 1e-5
        scale = weight * (running_var + eps).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


class DetrConvEncoder(nn.Module):
    def __init__(self) -> None: 
        super().__init__()
        backbone = resnet_fpn_backbone("resnet50", weights=ResNet50_Weights.IMAGENET1K_V2)
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
    
    def forward(self, 
                pixel_values: torch.Tensor, 
                pixel_mask: torch.Tensor
                ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # Example: 
        # pixel_values shape [1, 3, 800, 1024], pixel_mask shape [800, 1024]
        # send pixel values through the model to get list of feature maps
        features = self.model(pixel_values)
        out = []
        for feature_map in features.values():
            # downsample pixel mask to match shape of corresponding feature map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out 


class DetrSinePositionEmbedding(nn.Module):
    def __init__(self, 
                 embedding_dim: int,
                 temperature: int = 10_000,
                 normalize: bool = False,
                 scale: float = None,
                 ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("Normalize should be true if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, 
                pixel_values: torch.Tensor,
                pixel_mask: torch.Tensor,
                ) -> torch.Tensor:
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        # Intuition behind these cumulative sums is following 
        # let's take y_embed, as you go down the image the value will represent 
        # total number of valid pixels from the top of the image to that position 
        # As you go further down the image the value increases indicating accumulation of 
        # valid pixels. This gives a sense of vertical position within the image, 
        # while x_embed having same intuition but for horizontal position within the image.
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            # To normalize we divide on the last dimension
            # (on the dimension we cumulatively summed so all numbers will be 
            #  from 0 to 1 + small number to avoid dividing by zero)
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale
        # Here we generate dim_t to divide positional embeddings of x 
        # and y. Why is this done this way? First we generate numbers from 0 to embedding_dim
        # after this we transform this so if numbers were [0, 1, 2, ... 31] they now will be 
        # [0, 0, 2, 2 ..., 30, 30], this is done this way because later in code we calculate sine 
        # and cosine on odd and even numbers of the last dimension. 
        # dim_t dimension is [embedding_dim]
        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        # x_embed dimension example here can be [C, H, W, 1] after applying None indexing
        # and after dividing by dim it gets broadcasted to [C, H, W, embedding_dim]
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # Apply sin and cos for even and odd numbers of dim, 
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DetrLearnedPositionEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim: int = 256
                 ) -> None:
        super().__init__()
        self.row_embeddings = nn.Embedding(200, embedding_dim)
        self.column_embeddings = nn.Embedding(200, embedding_dim)

    def forward(self, 
                pixel_values, 
                pixel_mask=None,
                ) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos


class DetrPositionalEmbedder(nn.Module):
    def __init__(self, 
                 position_embedding_type: str,
                 embedding_dim: int,
                 ) -> None:
        super().__init__()
        if position_embedding_type == "sine":
            self.position_embedder = DetrSinePositionEmbedding(embedding_dim=embedding_dim)
        elif position_embedding_type == "learned":
            self.position_embedder = DetrLearnedPositionEmbedding(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Not supported {self.position_embedding_type}")

    def forward(self, 
                pixel_values: torch.Tensor, 
                pixel_mask: torch.Tensor = None,
                ) -> torch.Tensor:
        return self.position_embedder(pixel_values, pixel_mask)


class DetrConvModel(nn.Module):
    """
    Adds 2d Position embeddings to all intermediate feature maps of the convolutional encoder
    """
    def __init__(self, 
                 conv_encoder, 
                 position_embedding
                 ) -> None:
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def forward(self, 
                pixel_values:torch.Tensor,
                pixel_mask: torch.Tensor,
                ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # send pixel values and pixel mask through backbone to get list of (feature_map and pixel_mask tuples)
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))
        
        return out, pos


class DetrAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num heads (got `embed_dim`: {self.embed_dim} and `num_heads`:)"
                f" {num_heads}."
            )
        self.scaling = self.head_dim ** -0.5
        
        self.key = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.query = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out = nn.Linear(embed_dim, embed_dim, bias=bias)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[torch.Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_embeddings: Optional[torch.Tensor] = None,
            key_value_states: Optional[torch.Tensor] = None,
            key_value_position_embeddings: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape i.e. hidden_states shape: Batch x Time x Channel"""

        # if key value states are already provided this layer is used as a cross attention layer for the decoder
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()

        # add position embeddings to the hidden states before projecting to queries and keys 
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)
        
        # add key-value position embeddings to the key value states
        if key_value_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, key_value_position_embeddings)
        
        # get query projection
        query_states = self.query(hidden_states) * self.scaling
        if is_cross_attention:
            # cross attentions
            key_states = self._shape(self.key(key_value_states), -1, batch_size)
            value_states = self._shape(self.value(key_value_states_original), -1, batch_size)
        else:
            key_states = self._shape(self.key(hidden_states), -1, batch_size)
            value_states = self._shape(self.value(hidden_states_original), -1, batch_size)
            # self attention, returns tensor of shape [B, seq_len(-1), num_heads, head_dim]
            # So if before it was [1, 7500, 64] it becomes [1, 8, 7500, 8]

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        if attention_mask is not None:
            # NOTE why do we sum the attention mask here?
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # This operation is a bit awkward but its required to make sure that 
            # attention weights keeps its gradient.
            # In order to do so, attention weights have to be reshaped twice and have to be reused in the 
            # following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out(attn_output)

        return attn_output, attn_weights_reshaped
        

class DetrEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int = 64,
                 num_heads: int = 8,
                 encoder_ffn_dim: int = 32,
                 attention_dropout: float = 0.0,
                 dropout: float = 0.0,
                 activation_dropout: float = 0.0,
                 ) -> None:
        super().__init__()
        self.embed_dim = d_model
        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = dropout
        self.activation_fn = SiLU()
        self.activation_dropout = activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            position_embeddings: torch.Tensor = None,
            output_attentions: bool = False,
            ):
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states = hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        outputs = (hidden_states, )

        if output_attentions:
            outputs += (attn_weights, )

        return outputs
    
#TODO encoder done, now remains the loss decoder and all other stuff, omg this model is big.

encoder_layer = DetrEncoderLayer(d_model=64, num_heads=8, encoder_ffn_dim=8, attention_dropout=0.0, dropout=0.0, activation_dropout=0.0)
pixel_values = torch.rand((1, 3, 300, 400))
pixel_mask = torch.randint(0, 2, (1, ) + pixel_values.size()[-2:])
model = DetrConvEncoder()
last_intermediate_channel_size = 256
d_model = 64
pos_embedder = DetrPositionalEmbedder("learned", d_model // 2)
output = model(pixel_values, pixel_mask)

feature = output[0][0]
mask = output[0][1]

position_embedding = pos_embedder(feature, mask)
input_projection = nn.Conv2d(last_intermediate_channel_size, d_model, kernel_size=1)
projected_feature_map = input_projection(feature)

flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
position_embedding = position_embedding.flatten(2).permute(0, 2, 1)
mask = mask.flatten(1)


attention_layer = DetrAttention(embed_dim=64, num_heads=8, dropout=0)
a = attention_layer(hidden_states=flattened_features,
                attention_mask=mask,
                position_embeddings=position_embedding)

b = encoder_layer(flattened_features, mask, position_embedding)
