import torch
import torch.nn as nn
from einops import rearrange


class VisionTransformerAutoencoder(nn.Module):
    def __init__(
            self,
            nframe=100,
            input_dim=64,
            output_dim=64,
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
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dimension = dimension

        if dimension == 3:
            self.frame_embedding = nn.Conv3d(1, self.embedding_dim, (nframe, 1, 1))
            self.pixel_embedding = nn.Conv2d(self.embedding_dim, self.embedding_dim, self.patch_size, stride=self.patch_size)
        elif dimension == 2:
            self.frame_embedding = nn.Identity()
            self.pixel_embedding = nn.Conv2d(1, self.embedding_dim, self.patch_size, stride=self.patch_size)

        self.positional_encoding = nn.Parameter(torch.zeros(1, (input_dim//self.patch_size)**2, self.embedding_dim))
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
        # x = x.unsqueeze(1) # add channel dimension
        x = self.frame_embedding(x)
        if self.dimension == 3:
            x = x.squeeze(2) # remove frame dimension
        x = self.pixel_embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x += self.positional_encoding
        x = self.transformer_encoder(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.input_dim//self.patch_size, w=self.input_dim//self.patch_size)
        x = self.decoder(x)
        # x = x.squeeze(1) # remove channel dimension
        return x


class VisionTransformerEncoder2D(nn.Module):
    def __init__(
            self,
            input_dim=64,
            output_dim=64,
            patch_dim=16,
            hidden_dim=256,
            num_heads=2,
            num_layers=2,
            dropout=0.1,
    ):
        super(VisionTransformerEncoder2D, self).__init__()

        self.patch_dim = patch_dim
        self.embedding_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_patches = input_dim//patch_dim

        self.embedding = nn.Conv2d(1, self.embedding_dim, self.patch_dim, stride=self.patch_dim)

        self.linear_projection = (self.num_patches**2, hidden_dim)

        self.positional_encoding = nn.Parameter(torch.zeros(1, (input_dim//self.patch_dim)**2, self.embedding_dim))


        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )

    def forward(self, x):
        x = x.unsqueeze(1) # add channel dimension
        x = self.embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x += self.positional_encoding
        x = self.transformer_encoder(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.input_dim//self.patch_dim, w=self.input_dim//self.patch_dim)
        return x


class VisTransformerEncoder2D(nn.Module):
    def __init__(
            self,
            input_dim=1024,
            output_dim=32,
            patch_dim=32,
            hidden_dim=256,
            num_heads=2,
            num_layers=2,
            dropout=0.1,
    ):
        super(VisTransformerEncoder2D, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim
        self.num_patches = input_dim//patch_dim

        self.embedding = nn.Conv2d(1, self.num_patches, self.patch_dim, stride=self.patch_dim)
        self.flatten_patches = nn.Flatten(start_dim=-2, end_dim=-1)
        self.linear_projection = nn.Linear(self.num_patches**2, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_dim))

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.positional_encoding)
        nn.init.xavier_uniform_(self.linear_projection.weight)
        nn.init.constant_(self.linear_projection.bias, 0.0)

    def forward(self, x):
        x = x.unsqueeze(1) # add channel dimension
        x = self.embedding(x)
        x = self.flatten_patches(x)
        x = self.linear_projection(x)
        x += self.positional_encoding
        x = self.transformer_encoder(x)
        return x


if __name__ == '__main__':
    nframe = 16
    output_dim = 32
    input_dim = output_dim**2
    hidden_dim = 100
    num_heads = 4
    num_layers = 6
    dropout = 0.1

    model = VisTransformerEncoder2D(
        # input_dim=input_dim,
        # output_dim=output_dim,
        # hidden_dim=hidden_dim,
        # patch_dim=output_dim//2,
    )

    # input_tensor = torch.randn(12, nframe, npixel, npixel)  # For 3D (multi-frame) input
    input_tensor = torch.randn(12, input_dim, input_dim)  # For 2D (correlation matrix) input
    output_tensor = model(input_tensor)
    print(input_tensor.shape)
    print(output_tensor.shape)  # Should print: torch.Size([1, 32, 64, 64])

    #
    # final_conv = nn.Conv2d(hidden_dim, 1, 3, 2, 1)
    # final_out = final_conv(output_tensor)
    # print(final_out.shape)

