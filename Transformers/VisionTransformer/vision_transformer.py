from torch import nn
from typing import Tuple, Union, Optional
import torch
import math
from enforce_typing import enforce_type_checking


class ViTPatchEmbeddings(nn.Module):
    """
    This class turns pixel values of shape (batch_size, num_channels, height, width) into the initial 
    hidden states(i.e. patch embeddings) of shape (batch_size, seq_length, hidden_size) to be consumed 
    by a Transformer model. 
    #TODO 
    It is assumed now that patch size is a square. Rewrite to assume that patch size is not a square.
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int], 
                 patch_size: int,
                 num_channels: int,
                 hidden_size: int,
                 ) -> None:
        super().__init__()
        self.image_size = image_size 
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (self.image_size[1] // self.patch_size) * (self.image_size[0] // self.patch_size)

        self.hidden_size = hidden_size
        
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    
    def forward(self, image: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = image.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that channel dimension of the pixel values match with the one set in the configuration"
                f" Expected {self.num_channels} but god {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width} doesn't match model)"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(image).flatten(2).transpose(1, 2)
        # Returns for example (1, 3200, 512) 3200 being the number of patches
        return embeddings


class ViTEmbeddings(nn.Module):
    """
    Constructs the CLS token, position and patch embeddings.
    """
    
    def __init__(self,
                 image_size: Tuple[int, int],
                 patch_size: int,
                 num_channels: int,
                 hidden_size: int,
                 hidden_dropout_prob: float,
                 ) -> None:
        super().__init__()

        # in cls token 0th dimension is for batch, 1st dimension is for the cls token, 2nd dimension is for embedding size
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.patch_embedding_generator = ViTPatchEmbeddings(image_size, patch_size, num_channels, hidden_size)
        num_patches = self.patch_embedding_generator.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(hidden_dropout_prob)

    
    def forward(
            self,
            image: torch.Tensor,
    ):
        batch_size, num_channels, height, width = image.shape
        embeddings = self.patch_embedding_generator(image)
        # Expand to accomodate the batch size for cls token 
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # add positional encoding to each token
        embeddings += self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings


class ViTSelfAttention(nn.Module):
    
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 qkv_bias: bool,
                 attention_probs_dropout_prob: float,
                 ) -> None:
        super().__init__()
        if hidden_size & num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size} is not a multiple of the number of attention "
                f"heads {num_attention_heads}"
            )
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=qkv_bias)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    
    def forward(
            self,
            hidden_states: int,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor]:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between query and key to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to which might seem a bit unusual 
        # but its taken from the original Transformer paper 
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer, )
        
        return outputs
    

class ViTSelfOutput(nn.Module):
    
    def __init__(self, 
                 hidden_size:int,
                 hidden_dropout_prob: float,
                 ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(nn.Module):
    
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 qkv_bias: bool, 
                 attention_probs_dropout_prob: float,
                 hidden_dropout_prob: float,
                 ) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(hidden_size, num_attention_heads, qkv_bias, attention_probs_dropout_prob)
        self.output = ViTSelfOutput(hidden_size, hidden_dropout_prob)

    
    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states=hidden_states, output_attentions=False)
        attention_output = self.output(self_outputs[0])

        outputs = (attention_output, ) + self_outputs[1:]
        return outputs


class ViTIntermediate(nn.Module):
    
    def __init__(self, 
                 hidden_size: int, 
                 intermediate_size: int,
                 hidden_act: nn.Module,
                 ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = hidden_act

    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTOutput(nn.Module):
    
    def __init__(self,
                 intermediate_size: int,
                 hidden_size: int,
                 hidden_dropout_prob: float,
                 ) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    
    def forward(self, 
                hidden_states: torch.Tensor, 
                input_tensor: torch.Tensor
                ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states += input_tensor

        return hidden_states


class ViTLayer(nn.Module):
    
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 qkv_bias: bool,
                 attention_probs_dropout_prob: float,
                 hidden_dropout_prob: float,
                 intermediate_size: int,
                 hidden_act: nn.Module,
                 layer_norm_eps: float,
                 ) -> None:
        super().__init__()
        self.attention = ViTAttention(hidden_size, num_attention_heads, qkv_bias, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = ViTIntermediate(hidden_size, intermediate_size, hidden_act)
        self.output = ViTOutput(intermediate_size, hidden_size, hidden_dropout_prob)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    
    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states), # in ViT, layernorm is applied before self attention
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:] # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is applied also after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        outputs = (layer_output, ) + outputs
        return outputs
    

class ViTEncoder(nn.Module):
    
    def __init__(self, 
                 hidden_size: int, 
                 num_attention_heads: int, 
                 qkv_bias : bool,
                 attention_probs_dropout_prob: float,
                 hidden_dropout_prob: float,
                 intermediate_size: int,
                 hidden_act: nn.Module,
                 layer_norm_eps: float,
                 ) -> None:
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer(hidden_size, num_attention_heads, qkv_bias,
                                             attention_probs_dropout_prob, hidden_dropout_prob, 
                                             intermediate_size, hidden_act,
                                             layer_norm_eps)])

    
    def forward(
            self, 
            hidden_states: torch.Tensor,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_outputs = layer_module(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1], )
            
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return hidden_states
    

class ViTPooler(nn.Module):
    
    def __init__(self, 
                 hidden_size
                 ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    
    def forward(self, hidden_states: torch.Tensor):
        # Pooling is done by simply taking the hidden state corresponding to the first token(i.e. CLS token)
        cls_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(cls_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ViTModel(nn.Module):
    
    def __init__(self, 
                 image_size: Tuple[int, int],
                 patch_size: int,
                 num_channels: int,
                 hidden_size: int,
                 hidden_dropout_prob: float,
                 num_attention_heads: int,
                 qkv_bias: bool,
                 attention_probs_dropout_prob: float,
                 intermediate_size: int,
                 hidden_act: nn.Module,
                 layer_norm_eps: float,
                 add_pooling_layer: bool = True):
        super().__init__()
        self.embeddings = ViTEmbeddings(image_size=image_size,
                                        patch_size=patch_size,
                                        num_channels=num_channels,
                                        hidden_size=hidden_size,
                                        hidden_dropout_prob=hidden_dropout_prob,
                                        )
        self.encoder = ViTEncoder(hidden_size, 
                                  num_attention_heads,
                                  qkv_bias,
                                  attention_probs_dropout_prob,
                                  hidden_dropout_prob,
                                  intermediate_size,
                                  hidden_act, 
                                  layer_norm_eps,
                                  )
        
        self.layernorm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.pooler = ViTPooler(hidden_size) if add_pooling_layer else None
    
    
    def forward(self,
                image: torch.Tensor,
                ) -> Union[Tuple, torch.Tensor]:
        if image is None:
            raise ValueError("You have to specify an image for forward function to work")
        
        expected_dtype = self.embeddings.patch_embedding_generator.projection.weight.dtype
        if image.dtype != expected_dtype:
            image = image.to(expected_dtype)

        embedding_output = self.embeddings(
            image, 
        )

        encoder_output = self.encoder(
            embedding_output,
        )

        sequence_output = self.layernorm(encoder_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return sequence_output, pooled_output


class ViTImageClassification(nn.Module):
    def __init__(self,
                 num_classes,
                 image_size,
                 patch_size,
                 num_channels,
                 hidden_size,
                 hidden_dropout_prob=0.0,
                 num_attention_heads=8,
                 qkv_bias=False,
                 attention_probs_dropout_prob=0.0,
                 intermediate_size=768,
                 hidden_act=nn.ReLU(),
                 layer_norm_eps=0.001,
                 add_pooling_layer=True,
                 ) -> None:
        super().__init__()
        self.vit_model = ViTModel(image_size=(800, 1024),
                                  patch_size=16,
                                  num_channels=3,
                                  hidden_size=512,
                                  hidden_dropout_prob=0.0,
                                  num_attention_heads=8,
                                  qkv_bias=False,
                                  attention_probs_dropout_prob=0.0,
                                  intermediate_size=768,
                                  hidden_act=nn.ReLU(),
                                  layer_norm_eps=0.001,
                                  add_pooling_layer=True,
                                  )
        self.classifier = nn.Linear(512, num_classes)
        self.softmax_func = nn.Softmax(dim=1)

    def forward(self, image: torch.Tensor):
        _, cls_token_embedding = self.vit_model(image)
        cls_embedding = self.classifier(cls_token_embedding)
        cls_probas = self.softmax_func(cls_embedding)
        return cls_probas
