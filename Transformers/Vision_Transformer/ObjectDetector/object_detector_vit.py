from transformers import DetrForObjectDetection
from torch import nn
import torch
from typing import List, Tuple, Optional
import math
import timm
from torch.nn import SiLU
from scipy.optimize import linear_sum_assignment
from transformers.image_transforms import center_to_corners_format
from torch import Tensor


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

def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area

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
        # backbone = resnet_fpn_backbone("resnet50", weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone = timm.create_model("resnet50", 
                                     pretrained=True,
                                     features_only=True,
                                     out_indices=(1, 2, 3, 4),
                                     in_chans=3
                                     )

        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_size = self.model.feature_info.channels()
    
    def forward(self, 
                pixel_values: torch.Tensor, 
                pixel_mask: torch.Tensor
                ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # Example: 
        # pixel_values shape [1, 3, 800, 1024], pixel_mask shape [800, 1024]
        # send pixel values through the model to get list of feature maps
        features = self.model(pixel_values)
        out = []
        for feature_map in features:
            # downsample pixel mask to match shape of corresponding feature map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out 


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, target_len: Optional[int] = None):
    """
    Expands attention_mask from `[batch_size, seq_len]` to `[batch_size, 1, target_seq_len, source_seq_len]`.
    """
    batch_size, source_len = mask.size()
    target_len = target_len if target_len is not None else source_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_len, source_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)



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
            object_queries: Optional[torch.Tensor] = None,
            key_value_states: Optional[torch.Tensor] = None,
            spatial_position_embeddings: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        position_embeddings = None
        key_value_position_embeddings = None

        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        if key_value_position_embeddings is not None and spatial_position_embeddings is not None:
            raise ValueError(
                "Cannot specify both key_value_position_embeddings and spatial_position_embeddings. Please use just spatial_position_embeddings"
            )

        if position_embeddings is not None:
            object_queries = position_embeddings

        if key_value_position_embeddings is not None:
            spatial_position_embeddings = key_value_position_embeddings

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()

        # add position embeddings to the hidden states before projecting to queries and keys
        if object_queries is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, object_queries)

        # add key-value position embeddings to the key value states
        if spatial_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, spatial_position_embeddings)

        # get query proj
        query_states = self.query(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.key(key_value_states), -1, batch_size)
            value_states = self._shape(self.value(key_value_states_original), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.key(hidden_states), -1, batch_size)
            value_states = self._shape(self.value(hidden_states_original), -1, batch_size)

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
            if attention_mask.size() != (batch_size, 1, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, target_len, source_len)}, but is"
                    f" {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
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
            object_queries: torch.Tensor = None,
            output_attentions: bool = False,
            ):
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states = hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
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

    
class DetrEncoder(nn.Module):
    def __init__(self, dropout: float = 0.0, encoder_layerdrop: float = 0.0, encoder_layers: int = 4, ):
        super().__init__()
        self.dropout = dropout
        self.layerdrop = encoder_layerdrop
        
        self.layers = nn.ModuleList([DetrEncoderLayer() for _ in range(encoder_layers)])
    
    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        object_queries=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        position_embeddings = kwargs.pop("position_embeddings", None)
        
        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )
        if position_embeddings is not None:
            object_queries = position_embeddings
        
        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        #expand attention_mask
        if attention_mask is not None:
            # [B, seq_len] -> [B, 1, target_seq_len, source_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
        
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop: # skip the layer
                    to_drop = True
            
            if to_drop:
                layer_outputs = (None, None)
            else:
                # we add object_queries as extra input to the encoder layer
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    object_queries=object_queries,
                    output_attentions=output_attentions,
                )
                
                hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )
        
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states, )
        
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)


class DetrDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int = 64, 
                 decoder_attention_heads: int = 8, 
                 attention_dropout: float = 0.0, 
                 dropout: float = 0.0, 
                 activation_dropout: float = 0.0, 
                 decoder_ffn_dim: int = 32):
        super().__init__()
        self.embed_dim = d_model
        
        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=attention_dropout,
        )
        self.dropout = dropout
        self.activation_fn = SiLU()
        self.activation_dropout = activation_dropout
        
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = DetrAttention(
            self.embed_dim,
            decoder_attention_heads,
            dropout=attention_dropout,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, decoder_ffn_dim)
        self.fc2 = nn.Linear(decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            hidden_states (torch.FloatTensor): input to the layer of shape [B, seq_len, embed_dim]
            attention_mask (torch.FloatTensor): attention mask of size (B, 1, target_len, source_len) where padding 
            elements are indiicated by very large negative values
            object_queries: object_queries that are added to the hidden_states in the cross attn layer
            query_position_embeddings:
                position_embeddings that are added to the queries and keys in the self-attention layer
            encoder_hidden_states: 
                cross attention input to the layer of shape (B, seq_len, embed_dim)
            encoder_attention_mask: encoder attention mask of size
                [B, 1, target_len, source_len] where padding elements are indicated by very large negative values
            output_attentions: bool
                Whether or not to return the attentions tensors of all attention layers. 
        """
        position_embeddings = kwargs.pop("position_embeddings", None)
        
        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )
        if position_embeddings is not None:
            object_queries = position_embeddings
        
        residual = hidden_states
        
        # self attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            object_queries=query_position_embeddings,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Cross attention block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            
            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                object_queries=query_position_embeddings,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                spatial_position_embeddings=object_queries,
                output_attentions=output_attentions,
            )
            
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        
        return outputs


# d_model: int = 64,
#                  num_heads: int = 8,
#                  encoder_ffn_dim: int = 32,
#                  attention_dropout: float = 0.0,
#                  dropout: float = 0.0,
#                  activation_dropout: float = 0.0,

class DetrDecoder(nn.Module):
    def __init__(self, 
                 dropout: float = 0.0,
                 decoder_layerdrop: float = 0.0,
                 decoder_layers: int = 4,
                 d_model: int = 64, 
                 decoder_attention_heads: int = 8,
                 attention_dropout: float = 0.0,
                 activation_dropout:float = 0.0,
                 decoder_ffn_dim: int = 32):
        super().__init__()
        self.dropout = dropout
        self.layerdrop = decoder_layerdrop
        
        self.layers = nn.ModuleList([DetrDecoderLayer(d_model, decoder_attention_heads, attention_dropout, 
                                                      dropout, activation_dropout, decoder_ffn_dim) for _ in range(decoder_layers)])
        # in DETR, the decoder uses layernorm after the last decoder layer output
        self.layernorm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = False
        self.auxiliary_loss = None
    
    def forward(self,
                inputs_embeds=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                object_queries=None,
                query_position_embeddings=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=None,
                **kwargs,
                ):
        position_embeddings = kwargs.pop("position_embeddings", None)
        
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            input_shape = inputs_embeds.size()[:-1]
        
        combined_attention_mask = None
        
        if attention_mask is not None and combined_attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            combined_attention_mask = combined_attention_mask + _expand_mask(
                attention_mask, inputs_embeds.dtype, target_len=input_shape[-1]
            )
        
        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            encoder_attention_mask = _expand_mask(
                encoder_attention_mask, inputs_embeds.dtype, target_len=input_shape[-1]
            )
        
        # optional intermediate hidden states
        intermediate = () if self.auxiliary_loss else None
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
                
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    object_queries=object_queries,
                    query_position_embeddings=query_position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )
            
                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                    if encoder_hidden_states is not None:
                        all_cross_attentions += (layer_outputs[2],)
                # finally apply layernorm
                hidden_states = self.layernorm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_self_attns, all_cross_attentions, intermediate]
            if v is not None
        ) 


class DetrModel(nn.Module):
    def __init__(self, 
                 d_model: int = 64,
                 num_queries: int = 400,
                 ):
        super().__init__()
        # Create backbone + positional encoding
        backbone = DetrConvEncoder()
        object_queries = DetrPositionalEmbedder(position_embedding_type="learned", embedding_dim=d_model // 2)
        self.backbone = DetrConvModel(backbone, object_queries)
        
        # Create a projection layer
        self.input_projection = nn.Conv2d(backbone.intermediate_channel_size[-1], d_model, kernel_size=1)
        
        self.query_position_embeddings = nn.Embedding(num_queries, d_model)
        
        self.encoder = DetrEncoder()
        self.decoder = DetrDecoder()
        
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def freeze_backbone(self):
        for _, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for _, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device
        
        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)
        
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # pixel_values should be of shape (B, num_ch, H, W)
        # pixel_mask should be of shape (B, H, W)
        features, object_queries_list = self.backbone(pixel_values, pixel_mask)
        
        # get final feature map and downsampled mask
        feature_map, mask = features[-1]
        
        if mask is None:
            raise ValueError("Backbone does not return downsampled pixel mask")

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        projected_feature_map = self.input_projection(feature_map)
        
        # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW and permute it to NxHWxC
        # i.e. turn the shape to [B, seq_len, hidden_size] (if you understand transformer inputs more)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)
        
        # Mask from [B, H, W] -> [B, HW]
        flattened_mask = mask.flatten(1)
        
        # Fourth, send flattened_features + flattened_mask + position_embeddings through encoder
        # flattened features is a Tensor of shape (B, HW, hidden_size)
        # flattened_mask is a Tensor of shape (B, HW)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                object_queries=object_queries,
                output_attentions=output_attentions, # bool
                output_hidden_states=output_hidden_states, # bool
                return_dict=return_dict, # bool
            )

        # Fifth, send query embeddings + object_queries through the decoder(which is conditioned on the encoder output)
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)
        
        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if not return_dict:
            return decoder_outputs + encoder_outputs


class DetrMLPPredictionHead(nn.Module):
    "Simple MLP used to predict the normalized center coordinates, height and width of the bbox w.r.t an image"
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of teh network
    
    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """
    def __init__(self, class_cost: float = 1.0, bbox_cost: float = 1.0, giou_cost: float = 1.0):
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
        outputs (dict):
            A dictionary that containts at least these entries:
                logits - Tensor of dim [B, num_queries, num_classes] with cls logits
                pred_boxes - Tensor of dim [B, num_queries, 4] with the predicted bbox coordinates(relative to img size i.e. <= 1)
        targets:
            A list of targets (len(targets) == batch_size), where each target is a dict containing:
                class_labels: Tensor of dim [num_target_boxes] where num_target_boxes is the number of gt objects in the target containing the class label
                boxes - Tensor of dim [num_target_boxes, 4] containing the target box coordinates.
        
        Returns:
            List[Tuple]: A lsit of size batch_size, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions(in order)
            - index_j is the indices of the correspondong selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]
        
        # Flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flaten(0, 1).softmax(-1) # [ B * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1) # [B * num_queries, 4]
        
        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Compute the classification cost. Contrary to the loss, we dont use the NLL(negative log likelihood)
        # but approximate it in 1 - proba[target class]
        # The 1 is a constant doesnt change the matching, it can be ommited.
        class_cost = -out_prob[:, target_ids]
        
        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
        
        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))
        
        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class DetrLoss(nn.Module):
    def __init__(self, matcher, num_classes, eos_coef, losses):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
    
    def loss_labels(self, outputs, targets, indices):
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]
        
        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o
        
        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets):
        """
        Compute the carindality error, i.e. the absolute error in the number of predicted non empty boxes.
        This is not a loss, just intended for logging purposes only
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and GIoU loss.
        
        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes 
        are expected in format (center_x, center_y, w, h) normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")
        
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        loss_giou = 1- torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses
    
    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)
    
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}
        # Retrieve the matching between the outptus of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        return losses
        

# d_model: int = 64,
#                  num_heads: int = 8,
#                  encoder_ffn_dim: int = 32,
#                  attention_dropout: float = 0.0,
#                  dropout: float = 0.0,
#                  activation_dropout: float = 0.0,

class DetrForObjectDetection(nn.Module):
    def __init__(self,
                 d_model,
                 num_labels,
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels
        self.eos_coefficient = 0.1
        # DETR encoder-decoder model
        self.model = DetrModel()
        
        # Object detection heads
        self.class_labels_classifier = nn.Linear(
            d_model, num_labels + 1,
        ) # Add one for the no object class
        self.bbox_predictor = DetrMLPPredictionHead(
            input_dim=d_model, hidden_dim=d_model, output_dim=4, num_layers=3,
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        # First, send images through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            # decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            # inputs_embeds=inputs_embeds,
            # decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        
        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()
        
        loss, loss_dict, aux_outputs = None, None, None
        if labels is not None:
            # First create the matcher
            matcher = DetrHungarianMatcher()
            # Second create the criterion
            losses = ["labels", "boxes", "cardinality"]
            criterion = DetrLoss(
                matcher=matcher,
                num_classes=self.num_labels,
                eos_coef=self.eos_coefficient,
                losses=losses,
            )
            criterion.to(self.device)
            # Third compute the losses based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            
            loss_dict = criterion(outputs_loss, labels)
            # Fourth, compute total loss as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        output = (logits, pred_boxes) + outputs
        return ((loss, loss_dict) + output) if loss is not None else (logits, pred_boxes)


image = torch.rand((2, 3, 640, 640))
model = DetrForObjectDetection(d_model=64, num_labels=0)
model.eval()
with torch.no_grad():
    result = model(image)
print(result)
print(len(result))

result[0].shape
result[1].shape

