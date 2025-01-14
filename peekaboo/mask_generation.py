import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from rp import gaussian_kernel, as_numpy_array

def nans_like(tensor: torch.Tensor) -> torch.Tensor:
    # shape 맞춰서 nan elts로 이뤄진 텐서 생성
    return torch.full_like(tensor, torch.nan)

def shifted_image(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    '''
    image는 torch image로, 경계는 nan으로 패딩하여 경계 밖 픽셀을 추적하고 edge-case를 처리. 암튼 Augmentation인걸까?
    '''
    C_, H, W = image.shape

    if abs(dx) >= W or abs(dy) >= H:
        return nans_like(image)

    def shift_along_dim(tensor, shift, dim):
        if shift == 0:
            return tensor
        nans = nans_like(tensor.narrow(dim, 0, abs(shift)))
        if shift > 0:
            return torch.concat((nans, tensor.narrow(dim, 0, tensor.size(dim) - shift)), dim=dim)
        else:
            return torch.concat((tensor.narrow(dim, abs(shift), tensor.size(dim) - abs(shift)), nans), dim=dim)
    
    output = shift_along_dim(image, dx, 2)  # Shift along width
    output = shift_along_dim(output, dy, 1)  # Shift along height

    return output
  
def test_shifted_image():
    import icecream
    from rp import (
        as_float_image,
        as_numpy_image,
        as_torch_image,
        bordered_image_solid_color,
        cv_resize_image,
        display_image,
        load_image,
    )

    def process_and_display(image, dx, dy):
        """Helper function to process and display shifted images."""
        im = shifted_image(image, dx, dy)
        im = torch.nan_to_num(im)
        im = as_numpy_image(im)
        im = bordered_image_solid_color(im)
        icecream.ic(dx, dy)
        display_image(im)

    image_url = "https://nationaltoday.com/wp-content/uploads/2020/02/doggy-date-night.jpg"
    image = load_image(image_url)
    image = as_float_image(image)
    image = cv_resize_image(image, (64, 64))
    image = as_torch_image(image)
    shifts = [-64, -32, 0, 32, 64]
    for dx in shifts:
        for dy in shifts:
            process_and_display(image, dx, dy)

def get_weight_matrix(image: torch.Tensor, sigma: float, kernel_size: int, tolerance: float):
    # Return a 4d tensor corresponding to the weights needed to perform a bilateral blur

    C, H, W = image.shape
    R = kernel_size // 2  # Kernel radius

    device, dtype = image.device, image.dtype

    # Create Gaussian kernel matrix
    kernel = gaussian_kernel(size=kernel_size, sigma=sigma, dim=2)
    kernel = torch.tensor(kernel, dtype=dtype, device=device)

    # Create shifts tensor to store shifted images
    shifts = torch.empty((kernel_size, kernel_size, C, H, W), dtype=dtype, device=device)
    for u in range(kernel_size):
        for v in range(kernel_size):
            shifts[u, v] = shifted_image(image, u - R, v - R)

    # Compute color deltas and distances
    color_deltas = shifts - image[None, None]
    color_dists = (color_deltas ** 2).sum(dim=2).sqrt()

    # Compute color weights using a Gaussian function
    color_weights = torch.exp(-0.5 * (color_dists / tolerance) ** 2)

    # Combine spatial weights with color weights
    weights = kernel[:, :, None, None] * color_weights
    weights = weights.nan_to_num(0)  # Replace NaNs with 0
    weights = weights / weights.sum((0, 1), keepdim=True)  # Normalize weights

    return weights

def apply_weight_matrix(image: torch.Tensor, weights: torch.Tensor, iterations: int = 1):

    # If iterations > 1, apply recursively for faster performance
    if iterations > 1:
        for _ in range(iterations):
            image = apply_weight_matrix(image, weights)
        return image

    C, H, W = image.shape
    K = weights.shape[0]
    R = K // 2

    # Pre-allocate tensor for weighted color calculations
    weighted_colors = torch.empty((K, K, C, H, W), dtype=image.dtype, device=image.device)

    # Apply weights to shifted images
    for u in range(K):
        for v in range(K):
            shifted = shifted_image(image, u - R, v - R).nan_to_num()  # Replace nans with 0s
            weighted_colors[u, v] = shifted * weights[u, v][None, :, :]

    # Sum across all shifts to generate the output image
    return weighted_colors.sum(dim=(0, 1))

class BilateralProxyBlur:
    """
    BilateralProxyBlur 클래스는 이미지를 Bilateral Blur로 처리하는 객체입니다.

    Parameters:
    - image: 입력 이미지 (torch.Tensor, CHW 형식)
    - kernel_size: 커널 크기 (기본값: 5)
    - tolerance: 색상 차이에 따른 가중치 감쇠 (기본값: 0.08)
    - sigma: 공간 가중치의 표준편차 (기본값: 5)
    - iterations: 필터 반복 적용 횟수 (기본값: 10)
    """
    def __init__(self, image: torch.Tensor, *,
                 kernel_size: int = 5,
                 tolerance: float = 0.08,
                 sigma: float = 5,
                 iterations: int = 10):
        self.image = image
        self.kernel_size = kernel_size
        self.tolerance = tolerance
        self.sigma = sigma
        self.iterations = iterations
        self.weights = get_weight_matrix(image, sigma, kernel_size, tolerance)

    def __call__(self, image: torch.Tensor):
        """블러링을 적용한 이미지를 반환합니다."""
        return apply_weight_matrix(image, self.weights, self.iterations)

######## HELPER FUNCTIONS ########

class GaussianFourierFeatureTransform(nn.Module):
    """
    다음 논문을 참고하여 Gaussian Fourier feature mapping을 구현했음:
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
    https://arxiv.org/abs/2006.10739

    입력 텐서 크기를
    [batches, num_channels, width, height]
    --> [batches, num_features*2, width, height]
    크기로 변환합니다.
    """

    def __init__(self, num_channels, num_features=256, scale=10):
        """
        Gaussian 분포를 이용해 랜덤한 Fourier components를 생성.
        'scale'이 높을수록 더 높은 frequency features를 생성하여,
        간단한 MLP로도 detailed 이미지를 학습할 수 있지만,
        너무 높을 경우 high-frequency noise만 학습될 위험이 있음.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features

        # 주파수는 Gaussian 분포에서 생성되며, scale로 조정됨
        self.freqs = nn.Parameter(torch.randn(num_channels, num_features) * scale, requires_grad=False)
        # -> 학습되지 않는 파라미터
    
    def forward(self, x):
        batch_size, num_channels, height, width = x.shape

        # Reshape input for matrix multiplication with freqs: [B, C, H, W] -> [(B*H*W), C]
        x = x.permute(0, 2, 3, 1).reshape(-1, num_channels)

        # Multiply with freqs: [(B*H*W), C] x [C, F] -> [(B*H*W), F]
        x = x @ self.freqs

        # Reshape back to [B, H, W, F] and permute to [B, F, H, W]
        x = x.view(batch_size, height, width, self.num_features).permute(0, 3, 1, 2)

        # Apply sin and cos transformations
        x = 2 * torch.pi * x
        output = torch.cat([torch.sin(x), torch.cos(x)], dim=1)

        return output
    
def get_uv_grid(height: int, width: int, batch_size: int = 1) -> torch.Tensor:
    """
    (batch_size, 2, height, width) 크기의 UV grid torch cpu 텐서를 생성합니다.
    UV 좌표는 [0, 1) 사이의 값을 가지며, 1은 포함하지 않아 좌표가 텍스처를 360도로 감싸지 않도록 합니다.
    """

    # 0에서 1까지의 y와 x 좌표를 생성, 끝점은 포함하지 않음
    y_coords = np.linspace(0, 1, height, endpoint=False)  # Shape: (height,)
    x_coords = np.linspace(0, 1, width, endpoint=False)   # Shape: (width,)

    # UV grid 생성 후 예상되는 출력 형태로 reshape
    uv_grid = np.stack(np.meshgrid(y_coords, x_coords), -1)  # Shape: (width, height, 2)    
    uv_grid = torch.tensor(uv_grid).permute(2, 1, 0).unsqueeze(0).float().contiguous()  # Shape: (1, 2, height, width)
    uv_grid = uv_grid.repeat(batch_size, 1, 1, 1)  # batch_size만큼 반복

    return uv_grid

######## LEARNABLE IMAGES ########

class LearnableImage(nn.Module):
    '''
    Abstract class이며, subclassing을 통해 사용해야함.
    forward() 호출 시 (num_channels, height, width) 형태의 텐서를 반환해야 합니다.
    '''
    def __init__(self, height: int, width: int, num_channels: int):
        super().__init__()
        self.height = height
        self.width = width
        self.num_channels = num_channels
    
    def as_numpy_image(self):
        # 이미지를 Numpy 배열 형식으로 변환하여 반환합니다.
        image=self()
        image=as_numpy_array(image)
        image=image.transpose(1,2,0)
        return image

class LearnableImageRaster(LearnableImage):
    """
    픽셀로 파라미터화된 학습 가능한 이미지를 생성하는 클래스입니다.
    """
    def __init__(self, height: int, width: int, num_channels: int = 1):
        super().__init__(height, width, num_channels)
        # 랜덤 초기화된 이미지를 학습 가능한 파라미터로 설정
        self.image = nn.Parameter(torch.randn(num_channels, height, width))                 #####
        # self.image = nn.Parameter(torch.full((num_channels, height, width), 0.8))
        
    def forward(self):
        # 이미지 파라미터를 복사하여 반환
        output = self.image.clone()
        return torch.sigmoid(output)

class LearnableImageFourier(LearnableImage):
    '''
    Fourier features와 MLP를 통해 학습 가능한 이미지를 생성하는 클래스.
    출력 이미지는 [0, 1] 범위 내의 값을 가지며, 각 픽셀 값은 해당 범위를 벗어나지 않습니다.
    '''
    def __init__(self, 
                 height: int = 256, 
                 width: int = 256, 
                 num_channels: int = 3, 
                 hidden_dim: int = 256, 
                 num_features: int = 128, 
                 scale: int = 10):
        super().__init__(height, width, num_channels)

        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.scale = scale
        
        # 학습 중에 변경되지 않는 고정된 UV grid와 Fourier features 추출기
        self.uv_grid = nn.Parameter(get_uv_grid(height, width, batch_size=1), requires_grad=False)
        self.feature_extractor = GaussianFourierFeatureTransform(2, num_features, scale)
        self.features = nn.Parameter(self.feature_extractor(self.uv_grid), requires_grad=False)

        # MLP 구조를 갖는 1x1 Conv2d로 구성된 모델
        H, C, M = hidden_dim, num_channels, 2 * num_features
        self.model = nn.Sequential(
            nn.Conv2d(M, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
            nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
            nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
            nn.Conv2d(H, C, kernel_size=1),
            nn.Sigmoid(),
        )

    def get_features(self, condition=None):
        '''
        Fourier features를 반환하며, 조건(condition)이 주어지면 일부 features를 대체하여 반환합니다.
        !!!:TODO: Don't keep this! Condition should be CONCATENATED! Not replacing features...this is just for testing...
        '''
        features = self.features

        if condition is not None:
            # 첫 n개의 features를 condition으로 대체 (n = len(condition))
            features = features.clone()
            features = rearrange(features, 'B C H W -> B H W C')
            features[..., :len(condition)] = condition
            features = rearrange(features, 'B H W C -> B C H W')

        return features

    def forward(self, condition=None):
        # 학습된 이미지를 반환합니다.
        features = self.get_features(condition)
        output = self.model(features).squeeze(0)
        return output

######## TEXTURE PACKS ########

class LearnableTexturePack(nn.Module):
    """
    여러 개의 학습 가능한 이미지를 관리하는 abstract class. To be subclassed before use.
    이 클래스를 상속하여 사용해야 하며, get_learnable_image는 학습 가능한 이미지를 반환하는 함수입니다.
    """
    #TODO: Inherit from some list class, such as nn.ModuleList. That way we can access learnable_images by indexing them from self...

    def __init__(self, 
                 height: int, 
                 width: int, 
                 num_channels: int, 
                 num_textures: int, 
                 get_learnable_image):
        super().__init__()

        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.num_textures = num_textures

        # 학습 가능한 이미지를 nn.ModuleList로 저장
        self.learnable_images = nn.ModuleList([get_learnable_image() for _ in range(num_textures)])

    def as_numpy_images(self):
        # 모든 이미지를 Numpy 배열로 변환하여 반환
        return [x.as_numpy_image() for x in self.learnable_images]
       
    def forward(self):
        # 텐서 크기 (num_textures, num_channels, height, width)의 이미지를 반환
        output = torch.stack([x() for x in self.learnable_images])

        return output

    def __len__(self):
        # 텍스처 팩의 이미지 개수를 반환합니다.
        return len(self.learnable_images)

class LearnableImageRasterBilateral(LearnableImageRaster):
    """
    Bilateral blur를 적용한 Learnable Image Raster 클래스.
    """
    def __init__(self, bilateral_blur, num_channels: int = 3):
        _, height, width = bilateral_blur.image.shape
        super().__init__(height, width, num_channels)
        self.bilateral_blur = bilateral_blur

    def forward(self):
        # Bilateral blur를 적용한 학습 가능한 이미지를 반환합니다.
        output = self.image.clone()
        output = self.bilateral_blur(output)
        return torch.sigmoid(output)
    
class LearnableImageFourierBilateral(LearnableImageFourier):
    """
    Bilateral blur를 적용한 Learnable Image Fourier 클래스.
    """
    def __init__(self, bilateral_blur, num_channels: int = 3, hidden_dim: int = 256,
                 num_features: int = 128, scale: int = 10):
        _, height, width = bilateral_blur.image.shape
        super().__init__(height, width, num_channels, hidden_dim, num_features, scale)

        # 모델 초기화: 1x1 Conv2d를 사용하여 MLP 구조로 구성
        H, C, M = self.hidden_dim, self.num_channels, 2 * self.num_features
        self.model = nn.Sequential(
            nn.Conv2d(M, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
            nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
            nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
            nn.Conv2d(H, C, kernel_size=1))
        
        self.bilateral_blur = bilateral_blur

    def forward(self, condition=None):
        # Fourier features와 Bilateral blur를 적용한 학습 가능한 이미지를 반환합니다.
        features = self.get_features(condition)
        output = self.model(features).squeeze(0)

        output = self.bilateral_blur(output)
        output = torch.sigmoid(output)

        return output