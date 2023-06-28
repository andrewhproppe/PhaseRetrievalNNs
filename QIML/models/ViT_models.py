import torch
import torch.nn as nn
from einops import rearrange


class VisionTransformerAutoencoder(nn.Module):
    def __init__(
            self,
            nframe=100,
            npixel=64,
            patch_size=4,
            hidden_dim=256,
            num_heads=4,
            num_layers=6,
            dropout=0.1,
            dimension=3
    ):
        super(VisionTransformerAutoencoder, self).__init__()

        self.patch_size = patch_size
        self.embedding_dim = hidden_dim
        self.nframe = nframe
        self.npixel = npixel
        self.dimension = dimension

        if dimension == 3:
            self.frame_embedding = nn.Conv3d(1, self.embedding_dim, (nframe, 1, 1))
            self.pixel_embedding = nn.Conv2d(self.embedding_dim, self.embedding_dim, self.patch_size, stride=self.patch_size)
        elif dimension == 2:
            self.frame_embedding = nn.Identity()
            self.pixel_embedding = nn.Conv2d(1, self.embedding_dim, self.patch_size, stride=self.patch_size)

        self.positional_encoding = nn.Parameter(torch.zeros(1, (npixel//self.patch_size)**2, self.embedding_dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=num_heads, dropout=dropout), num_layers=num_layers
        )
        # self.decoder = nn.ConvTranspose2d(self.embedding_dim, 1, self.patch_size, stride=self.patch_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.embedding_dim//2, self.embedding_dim//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.embedding_dim//4, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Sigmoid activation function in the last layer
        )

    def forward(self, x):
        x = x.unsqueeze(1) # add channel dimension
        x = self.frame_embedding(x)
        if self.dimension == 3:
            x = x.squeeze(2) # remove frame dimension
        x = self.pixel_embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x += self.positional_encoding
        x = self.transformer_encoder(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.npixel//self.patch_size, w=self.npixel//self.patch_size)
        x = self.decoder(x)
        x = x.squeeze(1) # remove channel dimension
        return x


if __name__ == '__main__':
    # Example usage
    nframe = 100
    npixel = 32
    hidden_dim = 256
    num_heads = 4
    num_layers = 6
    dropout = 0.1

    model = VisionTransformerAutoencoder(
        nframe=nframe,
        npixel=npixel,
        dimension=3
    )
    input_tensor = torch.randn(12, nframe, npixel, npixel)  # For 3D (multi-frame) input
    # input_tensor = torch.randn(12, npixel**2, npixel**2)  # For 2D (correlation matrix) input
    output_tensor = model(input_tensor)
    print(input_tensor.shape)
    print(output_tensor.shape)  # Should print: torch.Size([1, 32, 64, 64])

    #
    # input_tensor = input_tensor.unsqueeze(1)
    # frame_embedding = nn.Conv3d(1, hidden_dim, (nframes, 1, 1))
    # pixel_embedding = nn.Conv2d(hidden_dim, hidden_dim, 4, 4)
    # output_tensor = frame_embedding(input_tensor)
    # output_tensor = output_tensor.squeeze(2)
    # output_tensor = pixel_embedding(output_tensor)
    # print(input_tensor.shape)
    # print(output_tensor.shape)

    # class ViTAE2D(nn.Module):
    #     def __init__(self, input_dim, output_dim, num_heads=8, hidden_dim=512, num_layers=6, dropout=0.1):
    #         super(ViTAE2D, self).__init__()
    #
    #         self.encoder = VisionTransformer(input_dim, hidden_dim, num_heads, num_layers, dropout)
    #         self.decoder = VisionTransformer(hidden_dim, output_dim, num_heads, num_layers, dropout)
    #
    #     def forward(self, x):
    #         encoded = self.encoder(x)
    #         decoded = self.decoder(encoded)
    #         return decoded
    #
    #
    # class VisionTransformer(nn.Module):
    #     def __init__(self, input_dim, output_dim, num_heads, num_layers, dropout):
    #         super(VisionTransformer, self).__init__()
    #
    #         self.embedding = nn.Linear(input_dim, output_dim)
    #         self.positional_encoding = PositionalEncoding(output_dim)
    #         self.transformer_blocks = nn.ModuleList([
    #             TransformerBlock(output_dim, num_heads, dropout)
    #             for _ in range(num_layers)
    #         ])
    #
    #     def forward(self, x):
    #         x = self.embedding(x)
    #         x = self.positional_encoding(x)
    #
    #         for transformer_block in self.transformer_blocks:
    #             x = transformer_block(x)
    #
    #         return x
    #
    #
    # class PositionalEncoding(nn.Module):
    #     def __init__(self, d_model, max_length=1024):
    #         super(PositionalEncoding, self).__init__()
    #
    #         position = torch.arange(0, max_length).unsqueeze(1)
    #         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    #         pe = torch.zeros(1, max_length, d_model)
    #         pe[0, :, 0::2] = torch.sin(position * div_term)
    #         pe[0, :, 1::2] = torch.cos(position * div_term)
    #         self.register_buffer('pe', pe)
    #
    #     def forward(self, x):
    #         x = x + self.pe[:, :x.size(1)]
    #         return x
    #
    #
    # class TransformerBlock(nn.Module):
    #     def __init__(self, d_model, num_heads, dropout):
    #         super(TransformerBlock, self).__init__()
    #
    #         self.multihead_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
    #         self.dropout1 = nn.Dropout(dropout)
    #         self.norm1 = nn.LayerNorm(d_model)
    #         self.feed_forward = nn.Sequential(
    #             nn.Linear(d_model, 4 * d_model),
    #             nn.ReLU(),
    #             nn.Linear(4 * d_model, d_model)
    #         )
    #         self.dropout2 = nn.Dropout(dropout)
    #         self.norm2 = nn.LayerNorm(d_model)
    #
    #     def forward(self, x):
    #         attended, _ = self.multihead_attention(x, x, x)
    #         x = x + self.dropout1(attended)
    #         x = self.norm1(x)
    #
    #         feed_forward_output = self.feed_forward(x)
    #         x = x + self.dropout2(feed_forward_output)
    #         x = self.norm2(x)
    #
    #         return x
    #
    #
    # # Example usage
    # input_dim = 1024
    # output_dim = 64
    # model = ViTAE2D(input_dim, output_dim)
    #
    # # Generate a random input tensor of size (batch_size, input_dim)
    # batch_size = 10
    # input_tensor = torch.randn(batch_size, input_dim, input_dim)
    #
    # # Pass the input through the model
    # output_tensor = model(input_tensor)
    #
    # print("Input tensor shape:", input_tensor.shape)
    # print("Output tensor shape:", output_tensor.shape)