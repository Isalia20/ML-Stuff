from Vision_Transformer.vision_transformer import ViTEmbeddings, ViTEncoder, ViTImageClassification
import torch
from torch import nn


vit_embedder = ViTEmbeddings((800, 1024), 16, 3, hidden_size=512, hidden_dropout_prob=0.0)
vit_encoder = ViTEncoder(512, 8, qkv_bias=False,attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0, 
                         intermediate_size=768, hidden_act=nn.ReLU(), layer_norm_eps=0.001)
model = ViTImageClassification(num_classes=10,
                               image_size=(800, 1024),
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

x = torch.rand((1, 3, 800, 1024))
with torch.no_grad():
    output = model(x)
    embedding = vit_embedder(x)
    a = vit_encoder(embedding)
    print(a.shape)


print(output.shape)
print(output)
