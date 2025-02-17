# This code was adapted from: https://github.com/openvinotoolkit/anomalib

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

from typing import List, Tuple, Union, Dict, Optional, Any

from models.basic_trainer import BasicTrainer


class Encoder(nn.Module):
    """Encoder Network.

    Args:
        input_size (Tuple[int, int]): Size of input image
        latent_vec_size (int): Size of latent vector z
        num_input_channels (int): Number of input channels in the image
        n_features (int): Number of features per convolution layer
        extra_layers (int): Number of extra layers since the network uses only a single encoder layer by default.
            Defaults to 0.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ):
        super().__init__()

        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{num_input_channels}-{n_features}",
            nn.Conv2d(num_input_channels, n_features, kernel_size=4, stride=2, padding=4, bias=False),
        )
        self.input_layers.add_module(f"initial-relu-{n_features}", nn.LeakyReLU(0.2, inplace=True))

        # Extra Layers
        self.extra_layers = nn.Sequential()

        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_features}-conv",
                nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-batchnorm", nn.BatchNorm2d(n_features))
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-relu", nn.LeakyReLU(0.2, inplace=True))

        # Create pyramid features to reach latent vector
        self.pyramid_features = nn.Sequential()
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_features
            out_features = n_features * 2
            self.pyramid_features.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.pyramid_features.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.pyramid_features.add_module(f"pyramid-{out_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            n_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Final conv
        self.final_conv_layer = None
        if add_final_conv_layer:
            self.final_conv_layer = nn.Conv2d(
                n_features,
                latent_vec_size,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )

    def forward(self, input_tensor: Tensor):
        """Return latent vectors."""

        output = self.input_layers(input_tensor)
        output = self.extra_layers(output)
        output = self.pyramid_features(output)
        if self.final_conv_layer is not None:
            output = self.final_conv_layer(output)

        return output


class Decoder(nn.Module):
    """Decoder Network.

    Args:
        input_size (Tuple[int, int]): Size of input image
        latent_vec_size (int): Size of latent vector z
        num_input_channels (int): Number of input channels in the image
        n_features (int): Number of features per convolution layer
        extra_layers (int): Number of extra layers since the network uses only a single encoder layer by default.
            Defaults to 0.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
    ):
        super().__init__()

        self.latent_input = nn.Sequential()

        # Calculate input channel size to recreate inverse pyramid
        exp_factor = math.ceil(math.log(min(input_size) // 2, 2)) - 2
        n_input_features = n_features * (2**exp_factor)

        # CNN layer for latent vector input
        self.latent_input.add_module(
            f"initial-{latent_vec_size}-{n_input_features}-convt",
            nn.ConvTranspose2d(
                latent_vec_size,
                n_input_features,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        self.latent_input.add_module(f"initial-{n_input_features}-batchnorm", nn.BatchNorm2d(n_input_features))
        self.latent_input.add_module(f"initial-{n_input_features}-relu", nn.ReLU(True))

        # Create inverse pyramid
        self.inverse_pyramid = nn.Sequential()
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_input_features
            out_features = n_input_features // 2
            self.inverse_pyramid.add_module(
                f"pyramid-{in_features}-{out_features}-convt",
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-relu", nn.ReLU(True))
            n_input_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Extra Layers
        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-conv",
                nn.Conv2d(n_input_features, n_input_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-batchnorm", nn.BatchNorm2d(n_input_features)
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-relu", nn.LeakyReLU(0.2, inplace=True)
            )

        # Final layers
        self.final_layers = nn.Sequential()
        self.final_layers.add_module(
            f"final-{n_input_features}-{num_input_channels}-convt",
            nn.ConvTranspose2d(
                n_input_features,
                num_input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        self.final_layers.add_module(f"final-{num_input_channels}-tanh", nn.Tanh())

    def forward(self, input_tensor):
        """Return generated image."""
        output = self.latent_input(input_tensor)
        output = self.inverse_pyramid(output)
        output = self.extra_layers(output)
        output = self.final_layers(output)
        return output


class Discriminator(nn.Module):
    """Discriminator.

        Made of only one encoder layer which takes x and x_hat to produce a score.

    Args:
        input_size (Tuple[int,int]): Input image size.
        num_input_channels (int): Number of image channels.
        n_features (int): Number of feature maps in each convolution layer.
        extra_layers (int, optional): Add extra intermediate layers. Defaults to 0.
    """

    def __init__(self, input_size: Tuple[int, int], num_input_channels: int, n_features: int, extra_layers: int = 0):
        super().__init__()
        encoder = Encoder(input_size, 1, num_input_channels, n_features, extra_layers)
        layers = []
        for block in encoder.children():
            if isinstance(block, nn.Sequential):
                layers.extend(list(block.children()))
            else:
                layers.append(block)

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, input_tensor):
        """Return class of object and features."""
        features = self.features(input_tensor)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


class Generator(nn.Module):
    """Generator model.

    Made of an encoder-decoder-encoder architecture.

    Args:
        input_size (Tuple[int,int]): Size of input data.
        latent_vec_size (int): Dimension of latent vector produced between the first encoder-decoder.
        num_input_channels (int): Number of channels in input image.
        n_features (int): Number of feature maps in each convolution layer.
        extra_layers (int, optional): Extra intermediate layers in the encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add a final convolution layer in the decoder. Defaults to True.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ):
        super().__init__()
        self.encoder1 = Encoder(
            input_size, latent_vec_size, num_input_channels, n_features, extra_layers, add_final_conv_layer
        )
        self.decoder = Decoder(input_size, latent_vec_size, num_input_channels, n_features, extra_layers)
        self.encoder2 = Encoder(
            input_size, latent_vec_size, num_input_channels, n_features, extra_layers, add_final_conv_layer
        )

    def forward(self, input_tensor):
        """Return generated image and the latent vectors."""
        latent_i = self.encoder1(input_tensor)
        gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return gen_image, latent_i, latent_o


class GanomalyModel(nn.Module):
    """Ganomaly Model.

    Args:
        input_size (Tuple[int,int]): Input dimension.
        num_input_channels (int): Number of input channels.
        n_features (int): Number of features layers in the CNNs.
        latent_vec_size (int): Size of autoencoder latent vector.
        extra_layers (int, optional): Number of extra layers for encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add convolution layer at the end. Defaults to True.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        num_input_channels: int,
        n_features: int,
        latent_vec_size: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()
        self.generator: Generator = Generator(
            input_size=input_size,
            latent_vec_size=latent_vec_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
        )
        self.discriminator: Discriminator = Discriminator(
            input_size=input_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
        )
        self.weights_init(self.generator)
        self.weights_init(self.discriminator)

    @staticmethod
    def weights_init(module: nn.Module):
        """Initialize DCGAN weights.

        Args:
            module (nn.Module): [description]
        """
        classname = module.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, batch: Tensor) -> Union[Tuple[Tensor, Tensor, Tensor], Tensor]:
        """Get scores for batch.

        Args:
            batch (Tensor): Images

        Returns:
            Tensor: Regeneration scores.
        """
        fake, latent_i, latent_o = self.generator(batch)
        return fake, latent_i, latent_o
        # return torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)  # convert nx1x1 to n


class GeneratorLoss(nn.Module):
    def __init__(self, wadv=1, wcon=50, wenc=1):
        super().__init__()

        self.loss_enc = nn.SmoothL1Loss()
        self.loss_adv = nn.MSELoss()
        # self.loss_con = nn.L1Loss()
        self.loss_con = nn.MSELoss()
        # self.loss_con = ComboLoss(
        #     weights={'bce': 3, 'dice': 1, 'focal': 4, 'jaccard': 0, 'lovasz': 0, 'lovasz_sigmoid': 0})

        self.wadv = wadv
        self.wcon = wcon
        self.wenc = wenc

    def forward(self, latent_i, latent_o, images, fake, pred_real, pred_fake, real_images=None):
        error_enc = self.loss_enc(latent_i, latent_o)

        if real_images is not None:
            not_nan = ~torch.isnan(real_images)
            error_con = self.loss_con(images[not_nan], fake[not_nan])
        else:
            error_con = self.loss_con(images, fake)
        error_adv = self.loss_adv(pred_real, pred_fake)

        loss = error_adv * self.wadv + error_con * self.wcon + error_enc * self.wenc
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_bce = nn.BCELoss()

    def forward(self, pred_real, pred_fake):
        error_discriminator_real = self.loss_bce(
            pred_real, torch.ones(size=pred_real.shape, dtype=torch.float32, device=pred_real.device)
        )
        error_discriminator_fake = self.loss_bce(
            pred_fake, torch.zeros(size=pred_fake.shape, dtype=torch.float32, device=pred_fake.device)
        )
        loss_discriminator = (error_discriminator_fake + error_discriminator_real) * 0.5
        return loss_discriminator


class GanomalyTrainer(BasicTrainer):
    def __init__(self, args, train_loader, test_loader, logger=None):
        super().__init__(args, train_loader, test_loader, logger)

        n_features = 64
        latent_vec_size = 100
        # latent_vec_size = 64
        extra_layers = 0
        add_final_conv = True

        self.model = GanomalyModel(input_size=(self.image_size, self.image_size), num_input_channels=self.in_channels,
                                   n_features=n_features, latent_vec_size=latent_vec_size, extra_layers=extra_layers,
                                   add_final_conv_layer=add_final_conv)

        # self.optimizer_d = optim.Adam(self.model.discriminator.parameters(), lr=args.lr)
        self.optimizer_d = optim.AdamW(self.model.discriminator.parameters(), 1e-4, weight_decay=25 * 1e-5)
        # self.optimizer_g = optim.Adam(self.model.generator.parameters(), lr=args.lr)
        self.optimizer_g = optim.AdamW(self.model.generator.parameters(), 1e-4, weight_decay=25 * 1e-5)

        self.generator_loss = GeneratorLoss()
        self.discriminator_loss = DiscriminatorLoss()

        self.epochs = args.epochs

    def prepare_train(self):
        pass

    def train_step(self, batch_idx, data, labels, path) -> torch.Tensor:
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()

        # Discriminator
        fake, latent_i, latent_o = self.model(data)
        pred_real, _ = self.model.discriminator(data)

        pred_fake, _ = self.model.discriminator(fake.detach())
        d_loss = self.discriminator_loss(pred_real, pred_fake)
        d_loss.backward()
        self.optimizer_d.step()

        # Generator
        fake, latent_i, latent_o = self.model(data)
        pred_real, _ = self.model.discriminator(data)

        pred_fake, _ = self.model.discriminator(fake)
        # g_loss = self.generator_loss(latent_i, latent_o, data[mask], fake[mask], pred_real, pred_fake, real_data[mask])
        g_loss = self.generator_loss(latent_i, latent_o, data, fake, pred_real, pred_fake)
        g_loss.backward()
        self.optimizer_g.step()

        return g_loss

    def prepare_test(self):
        pass

    def test_step(self, index, x, labels, path) -> (torch.Tensor, torch.Tensor):
        fake, latent_i, latent_o = self.model(x)
        # pred_real, _ = self.model.discriminator(x)

        # pred_fake, _ = self.model.discriminator(fake)

        loss = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)

        return loss, fake
