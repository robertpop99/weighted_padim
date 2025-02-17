# This code was adapted from: https://github.com/openvinotoolkit/anomalib

from random import sample

from torch import nn
import warnings

import timm
from typing import List, Tuple, Union, Dict, Optional, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kornia.filters import get_gaussian_kernel2d
from kornia.filters.filter import _compute_padding
from kornia.filters.kernels import normalize_kernel2d

from models.basic_trainer import BasicTrainer


DIMS = {
    "resnet18": {"orig_dims": 448, "reduced_dims": 100, "emb_scale": 4},
    # "resnet18": {"orig_dims": 896, "reduced_dims": 200, "emb_scale": 8},
    "wide_resnet50_2": {"orig_dims": 1792, "reduced_dims": 550, "emb_scale": 4},
}


class FeatureExtractor(nn.Module):
    """Extract features from a CNN.

    Args:
        backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        layers (Iterable[str]): List of layer names of the backbone to which the hooks are attached.

    """

    def __init__(self, backbone: str, layers: List[str], pre_trained: bool = True):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.idx = self._map_layer_to_idx()
        self.feature_extractor = timm.create_model(
            backbone,
            pretrained=pre_trained,
            features_only=True,
            exportable=True,
            out_indices=self.idx,
        )
        self.out_dims = self.feature_extractor.feature_info.channels()
        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self, offset: int = 3) -> List[int]:
        """Maps set of layer names to indices of model.

        Args:
            offset (int) `timm` ignores the first few layers when indexing please update offset based on need

        Returns:
            Feature map extracted from the CNN
        """
        idx = []
        features = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=False,
            exportable=True,
        )
        for i in self.layers:
            try:
                idx.append(list(dict(features.named_children()).keys()).index(i) - offset)
            except ValueError:
                warnings.warn(f"Layer {i} not found in model {self.backbone}")
                # Remove unfound key from layer dict
                self.layers.remove(i)

        return idx

    def forward(self, input_tensor: Tensor) -> Dict[str, Tensor]:
        """Forward-pass input tensor into the CNN.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Feature map extracted from the CNN
        """
        features = dict(zip(self.layers, self.feature_extractor(input_tensor)))
        return features


class GaussianBlur2d(nn.Module):
    """Compute GaussianBlur in 2d.

    Makes use of kornia functions, but most notably the kernel is not computed
    during the forward pass, and does not depend on the input size. As a caveat,
    the number of channels that are expected have to be provided during initialization.
    """

    def __init__(
        self,
        kernel_size: Union[Tuple[int, int], int],
        sigma: Union[Tuple[float, float], float],
        channels: int,
        normalize: bool = True,
        border_type: str = "reflect",
        padding: str = "same",
    ) -> None:
        """Initialize model, setup kernel etc..

        Args:
            kernel_size (Union[Tuple[int, int], int]): size of the Gaussian kernel to use.
            sigma (Union[Tuple[float, float], float]): standard deviation to use for constructing the Gaussian kernel.
            channels (int): channels of the input
            normalize (bool, optional): Whether to normalize the kernel or not (i.e. all elements sum to 1).
                Defaults to True.
            border_type (str, optional): Border type to use for padding of the input. Defaults to "reflect".
            padding (str, optional): Type of padding to apply. Defaults to "same".
        """
        super().__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        sigma = sigma if isinstance(sigma, tuple) else (sigma, sigma)
        self.kernel: Tensor
        self.register_buffer("kernel", get_gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma))
        self.kernel = self.kernel.view(kernel_size)
        if normalize:
            self.kernel = normalize_kernel2d(self.kernel)
        self.channels = channels
        self.kernel.unsqueeze_(0).unsqueeze_(0)
        self.kernel = self.kernel.expand(self.channels, -1, -1, -1)
        self.border_type = border_type
        self.padding = padding
        self.height, self.width = self.kernel.shape[-2:]
        self.padding_shape = _compute_padding([self.height, self.width])

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Blur the input with the computed Gaussian.

        Args:
            input_tensor (Tensor): Input tensor to be blurred.

        Returns:
            Tensor: Blurred output tensor.
        """
        batch, channel, height, width = input_tensor.size()

        if self.padding == "same":
            input_tensor = F.pad(input_tensor, self.padding_shape, mode=self.border_type)

        # convolve the tensor with the kernel.
        output = F.conv2d(input_tensor, self.kernel, groups=self.channels, padding=0, stride=1)

        if self.padding == "same":
            out = output.view(batch, channel, height, width)
        else:
            out = output.view(batch, channel, height - self.height + 1, width - self.width + 1)

        return out


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    Args:
        image_size (Union[ListConfig, Tuple]): Size of the input image. The anomaly map is upsampled to this dimension.
        sigma (int, optional): Standard deviation for Gaussian Kernel. Defaults to 4.
    """

    def __init__(self, image_size, sigma: int = 4, distance_type='mahalanobis'):
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)
        self.distance_type = distance_type

    def compute_distance(self, embedding: Tensor, stats: List[Tensor]) -> Tensor:
        """Compute anomaly score to the patch in position(i,j) of a test image.

        Ref: Equation (2), Section III-C of the paper.

        Args:
            embedding (Tensor): Embedding Vector
            stats (List[Tensor]): Mean and Covariance Matrix of the multivariate Gaussian distribution

        Returns:
            Anomaly score of a test image via mahalanobis distance.
        """
        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        # calculate mahalanobis distances
        mean, inv_covariance, covariance, matching_likelihood = stats

        if self.distance_type == 'mahalanobis':
            delta = (embedding - mean).permute(2, 0, 1)
            distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
            distances = distances.clamp(0).sqrt()
        elif self.distance_type == 'pdf':
            distances = torch.zeros(batch, height * width).to(embedding.device)
            for i in range(height * width):
                dist = matching_likelihood[i]
                distances[:, i] = -dist.log_prob(embedding[:, :, i])
            distances = distances.clamp(0)

        # delta = (embedding - mean).permute(2, 0, 1)
        # distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
        # distances = distances.clamp(0).sqrt()

        distances = distances.reshape(batch, 1, height, width)
        return distances

    def up_sample(self, distance: Tensor) -> Tensor:
        """Up sample anomaly score to match the input image size.

        Args:
            distance (Tensor): Anomaly score computed via the mahalanobis distance.

        Returns:
            Resized distance matrix matching the input image size
        """

        score_map = F.interpolate(
            distance,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return score_map

    def smooth_anomaly_map(self, anomaly_map: Tensor) -> Tensor:
        """Apply gaussian smoothing to the anomaly map.

        Args:
            anomaly_map (Tensor): Anomaly score for the test image(s).

        Returns:
            Filtered anomaly scores
        """

        blurred_anomaly_map = self.blur(anomaly_map)
        return blurred_anomaly_map

    def compute_anomaly_map(self, embedding: Tensor, mean: Tensor, inv_covariance: Tensor, covariance: Tensor,
                            matching_likelihood) -> Tensor:
        """Compute anomaly score.

        Scores are calculated based on embedding vector, mean and inv_covariance of the multivariate gaussian
        distribution.

        Args:
            embedding (Tensor): Embedding vector extracted from the test set.
            mean (Tensor): Mean of the multivariate gaussian distribution
            inv_covariance (Tensor): Inverse Covariance matrix of the multivariate gaussian distribution.
            covariance (Tensor): Covariance matrix of the multivariate gaussian distribution.

        Returns:
            Output anomaly score.
        """

        score_map = self.compute_distance(
            embedding=embedding,
            stats=[mean.to(embedding.device), inv_covariance.to(embedding.device), covariance.to(embedding.device),
                   matching_likelihood],
        )
        up_sampled_score_map = self.up_sample(score_map)
        smoothed_anomaly_map = self.smooth_anomaly_map(up_sampled_score_map)

        return smoothed_anomaly_map

    def forward(self, **kwargs):
        if not ("embedding" in kwargs and "mean" in kwargs and "inv_covariance" in kwargs):
            raise ValueError(f"Expected keys `embedding`, `mean` and `covariance`. Found {kwargs.keys()}")

        embedding: Tensor = kwargs["embedding"]
        mean: Tensor = kwargs["mean"]
        inv_covariance: Tensor = kwargs["inv_covariance"]
        covariance: Tensor = kwargs["covariance"]
        matching_likelihood = kwargs["matching_likelihood"]

        return self.compute_anomaly_map(embedding, mean, inv_covariance, covariance, matching_likelihood)


class MultiVariateGaussian(nn.Module):
    """Multi Variate Gaussian Distribution."""

    def __init__(self, n_features, n_patches):
        super().__init__()

        self.register_buffer("mean", torch.zeros(n_features, n_patches))
        self.register_buffer("inv_covariance", torch.eye(n_features).unsqueeze(0).repeat(n_patches, 1, 1))

        self.mean: Tensor
        self.inv_covariance: Tensor

    @staticmethod
    def _cov(
        observations: Tensor,
        rowvar: bool = False,
        bias: bool = False,
        ddof: Optional[int] = None,
        aweights: Tensor = None,
    ) -> Tensor:
        """Estimates covariance matrix like numpy.cov.

        Args:
            observations (Tensor): A 1-D or 2-D array containing multiple variables and observations.
                 Each row of `m` represents a variable, and each column a single
                 observation of all those variables. Also see `rowvar` below.
            rowvar (bool): If `rowvar` is True (default), then each row represents a
                variable, with observations in the columns. Otherwise, the relationship
                is transposed: each column represents a variable, while the rows
                contain observations. Defaults to False.
            bias (bool): Default normalization (False) is by ``(N - 1)``, where ``N`` is the
                number of observations given (unbiased estimate). If `bias` is True,
                then normalization is by ``N``. These values can be overridden by using
                the keyword ``ddof`` in numpy versions >= 1.5. Defaults to False
            ddof (Optional, int): If not ``None`` the default value implied by `bias` is overridden.
                Note that ``ddof=1`` will return the unbiased estimate, even if both
                `fweights` and `aweights` are specified, and ``ddof=0`` will return
                the simple average. See the notes for the details. The default value
                is ``None``.
            aweights (Tensor): 1-D array of observation vector weights. These relative weights are
                typically large for observations considered "important" and smaller for
                observations considered less "important". If ``ddof=0`` the array of
                weights can be used to assign probabilities to observation vectors. (Default value = None)


        Returns:
          The covariance matrix of the variables.
        """
        # ensure at least 2D
        if observations.dim() == 1:
            observations = observations.view(-1, 1)

        # treat each column as a data point, each row as a variable
        if rowvar and observations.shape[0] != 1:
            observations = observations.t()

        if ddof is None:
            if bias == 0:
                ddof = 1
            else:
                ddof = 0

        weights = aweights
        weights_sum: Any

        if weights is not None:
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=torch.float)  # pylint: disable=not-callable
            weights_sum = torch.sum(weights)
            avg = torch.sum(observations * (weights / weights_sum)[:, None], 0)
        else:
            avg = torch.mean(observations, 0)

        # Determine the normalization
        if weights is None:
            fact = observations.shape[0] - ddof
        elif ddof == 0:
            fact = weights_sum
        elif aweights is None:
            fact = weights_sum - ddof
        else:
            fact = weights_sum - ddof * torch.sum(weights * weights) / weights_sum

        observations_m = observations.sub(avg.expand_as(observations))

        if weights is None:
            x_transposed = observations_m.t()
        else:
            x_transposed = torch.mm(torch.diag(weights), observations_m).t()

        covariance = torch.mm(x_transposed, observations_m)
        covariance = covariance / fact

        return covariance.squeeze()

    def forward(self, embedding: Tensor) -> List[Tensor]:
        """Calculate multivariate Gaussian distribution.

        Args:
          embedding (Tensor): CNN features whose dimensionality is reduced via either random sampling or PCA.

        Returns:
          mean and inverse covariance of the multi-variate gaussian distribution that fits the features.
        """
        device = embedding.device

        batch, channel, height, width = embedding.size()
        embedding_vectors = embedding.view(batch, channel, height * width)
        self.mean = torch.mean(embedding_vectors, dim=0)
        self.covariance = torch.zeros(size=(channel, channel, height * width), device=device)
        self.matching_likelihood = [None] * height * width
        identity = torch.eye(channel).to(device)
        for i in range(height * width):
            self.covariance[:, :, i] = self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * identity
            self.matching_likelihood[i] = torch.distributions.multivariate_normal.MultivariateNormal(self.mean[:, i].to('cuda'), self.covariance[:, :, i].to('cuda'))

        # calculate inverse covariance as we need only the inverse
        self.inv_covariance = torch.linalg.inv(self.covariance.permute(2, 0, 1))

        return [self.mean, self.inv_covariance]

    def fit(self, embedding: Tensor) -> List[Tensor]:
        """Fit multi-variate gaussian distribution to the input embedding.

        Args:
            embedding (Tensor): Embedding vector extracted from CNN.

        Returns:
            Mean and the covariance of the embedding.
        """
        return self.forward(embedding)


class PadimModel(nn.Module):
    def __init__(self, image_size, layers, backbone: str = "resnet18", pre_trained: bool = True,
                 distance_type='mahalanobis'):
        super().__init__()
        self.tiler = None

        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = FeatureExtractor(backbone=self.backbone, layers=layers, pre_trained=pre_trained)
        self.dims = DIMS[backbone]
        # pylint: disable=not-callable
        # Since idx is randomly selected, save it with model to get same results
        self.register_buffer(
            "idx",
            torch.tensor(sample(range(0, DIMS[backbone]["orig_dims"]), DIMS[backbone]["reduced_dims"])),
        )
        self.idx: Tensor
        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=image_size, distance_type=distance_type)

        n_features = DIMS[backbone]["reduced_dims"]
        patches_dims = torch.tensor(image_size) / DIMS[backbone]["emb_scale"]
        n_patches = patches_dims.ceil().prod().int().item()
        self.gaussian = MultiVariateGaussian(n_features, n_patches)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward-pass image-batch (N, C, H, W) into model to extract features.

        Args:
            input_tensor: Image-batch (N, C, H, W)
            input_tensor: Tensor:

        Returns:
            Features from single/multiple layers.

        """

        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        if self.training:
            output = embeddings
        else:
            output = self.anomaly_map_generator(
                embedding=embeddings, mean=self.gaussian.mean, inv_covariance=self.gaussian.inv_covariance,
                covariance=self.gaussian.covariance, matching_likelihood=self.gaussian.matching_likelihood
            )
        return output

    def generate_embedding(self, features: Dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features (Dict[str, Tensor]): Hierarchical feature map from a CNN (ResNet18 or WideResnet)

        Returns:view(batch, channel, width, height)
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        # idx = self.idx.to(embeddings.device)
        # idx = torch.tensor(sample(range(64, 448), 300)).to(embeddings.device)
        # idx = torch.tensor(range(64, 448)).to(embeddings.device)
        # embeddings = torch.index_select(embeddings, 1, idx)
        idx = self.idx.to(embeddings.device)
        embeddings = torch.index_select(embeddings, 1, idx)
        return embeddings


class PadimTrainer(BasicTrainer):
    def __init__(self, args, train_loader, test_loader, logger=None):
        super().__init__(args, train_loader, test_loader, logger)

        layers = ['layer1', 'layer2', 'layer3']
        if 'wide' in self.model_type:
            backbone = 'wide_resnet50_2'
        else:
            backbone = "resnet18"
        pre_trained = True

        if 'pdf' in self.model_type:
            distance_type = 'pdf'
        else:
            distance_type = 'mahalanobis'

        self.model = PadimModel((self.image_size, self.image_size), layers, backbone, pre_trained, distance_type)
        self.embeddings = []
        self.stats = []

        self.epochs = 5

    def prepare_train(self):
        pass
    
    def train_step(self, batch_idx, data, labels, path) -> torch.Tensor:
        data_expanded = torch.cat([data, data, data], dim=1)

        self.model.feature_extractor.eval()
        embeddings = self.model(data_expanded)

        self.embeddings.append(embeddings.cpu())

        return torch.tensor([0])
    
    def prepare_test(self):
        embeddings = torch.vstack(self.embeddings)

        # self.stats = self.model.gaussian.fit(embeddings.to(self.device))
        self.stats = self.model.gaussian.fit(embeddings)

        # self.model.train()
        # self.model.feature_extractor.eval()
    
    def test_step(self, index, x, labels, path) -> (torch.Tensor, torch.Tensor):
        x_expanded = torch.cat([x, x, x], dim=1)
        anomaly_maps = self.model(x_expanded)

        # batch, channel, width, height = anomaly_maps.shape
        # anomaly_maps = anomaly_maps.view(batch, channel, width * height)
        # outputs = torch.zeros(batch).to(self.device)
        # for i in range(batch):
        #     outputs[i] = torch.dist(self.stats[0], anomaly_maps[i], p=0.2)

        outputs = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(dim=1).values

        return outputs, anomaly_maps




