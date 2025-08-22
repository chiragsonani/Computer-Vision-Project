#@: Exercise 4: Demosaicing & HDR Implementation
# This implementation:
# 1. Detects Bayer patterns from raw sensor data
# 2. Demosaics using bilinear interpolation to fill missing colors
# 3. Applies gamma correction for brightness enhancement
# 4. Uses gray world algorithm for white balance
# 5. Merges multiple exposures into HDR images
# 6. Implements tone mapping including iCAM06 (with some artifacts)
# 7. Preserves full HDR range throughout pipeline

# Author: Rahul Sawhney and Chirag Sonani
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol
from pathlib import Path
import time
import numpy as np
import rawpy
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

class Module:
    # Base class for all our image processing modules. Everything
    # inherits from this and implements forward(). Makes the code
    # cleaner and consistent. The __call__ thing lets us use the
    # class like a function which is pretty neat.
    def __init__(self) -> None:
        pass
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class BayerPattern:
    # Stores info about which Bayer pattern the camera uses. The pattern
    # tells us the color filter arrangement (RGGB means red-green-green-blue
    # in a 2x2 block). Offset is for when the pattern doesn't start at (0,0).
    # Confidence is how sure we are about the detection.
    pattern: str
    offset_x: int
    offset_y: int
    confidence: float

    
@dataclass(frozen=True, slots=True)
class PixelAverages:
    # Just the average values for each color channel. Used in linearity
    # analysis to see if the sensor responds linearly to light.
    red: float
    green: float
    blue: float


@dataclass(frozen=True, slots=True)
class ExposureData:
    # Links a file to its exposure time and average pixel values.
    # We use this to plot how brightness changes with exposure.
    filename: str
    exposure_time: float
    pixel_averages: PixelAverages
    

@dataclass(frozen=True, slots=True)
class ColorScaleFactors:
    # Multipliers for each color channel when doing white balance.
    # If red_scale is 1.2, we multiply all red pixels by 1.2 to
    # fix the color cast.
    red_scale: float
    green_scale: float
    blue_scale: float


@dataclass(frozen=True, slots=True)
class DemosaicResult:
    # Everything that comes out of demosaicing. The image is now RGB
    # instead of raw sensor data. We track which pattern and algorithm
    # we used, plus how long it took. Metadata is for any extra info.
    image: np.ndarray
    bayer_pattern: BayerPattern
    algorithm: str
    processing_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WhiteBalanceResult:
    # Output from white balancing. The image has corrected colors
    # and we keep track of how much we scaled each channel.
    image: np.ndarray
    scale_factors: ColorScaleFactors
    method: str


@dataclass(frozen=True, slots=True)
class GammaSettings:
    # Controls how we brighten dark images. Gamma is the exponent
    # (0.3 makes things way brighter). Percentiles clip outliers
    # before applying the curve. Curve type lets us try different
    # brightness functions.
    gamma: float
    percentile_low: float
    percentile_high: float
    curve_type: str
    

@dataclass(frozen=True, slots=True)
class HDRImage:
    # Holds the merged HDR data. Raw_data has huge values (like 1.4M)
    # because we combined multiple exposures. Exposure_times tracks
    # what went into it. Bit_depth is usually 32 for float data.
    raw_data: np.ndarray
    exposure_times: list[float]
    merge_method: str
    bit_depth: int


class BayerPatternDetector(Module):
    # Figures out which Bayer pattern the camera uses (RGGB, GRBG, etc).
    # Can detect it blind by looking at patterns or use a reference image.
    # The blind detection is kinda hit or miss but works for standard patterns.
    # Blindly guess the Bayer pattern from a raw image (_detect_blind)
    # Compare guesses to a reference image to find the best match (_detect_with_reference)
    
    def __init__(self) -> None:
        super().__init__()
        self._known_patterns: list[str] = ['RGGB', 'GRBG', 'GBRG', 'BGGR']
                
    # If a reference image is provided, it uses it to detect the best match.
    # If no reference, it just guesses using green channel variance.
    def forward(self, raw_data: np.ndarray, reference_image: np.ndarray | None = None) -> BayerPattern:
        if reference_image is not None:
            return self._detect_with_reference(raw_data, reference_image)
        return self._detect_blind(raw_data)
    
    # Tries all patterns, and all possible 2×2 alignments (offsets).
    # For each case, it demosaics the raw image with the given pattern and offset.
    # Then it compares it to the reference image using MSE (mean squared error).
    # The best (lowest error = highest score) pattern is selected.

    def _detect_with_reference(self, raw_data: np.ndarray, reference_image: np.ndarray) -> BayerPattern:
        best_pattern: str = 'RGGB'
        best_score: float = -np.inf
        best_offset: tuple[int, int] = (0, 0)
        
        for pattern in self._known_patterns:
            for offset_y in range(2):
                for offset_x in range(2):
                    score: float = self._evaluate_pattern(raw_data, reference_image, pattern, offset_x, offset_y)
                    if score > best_score:
                        best_score = score
                        best_pattern = pattern
                        best_offset = (offset_x, offset_y)
        
        confidence: float = 1.0 / (1.0 + np.exp(-best_score)) #Converts the score into a confidence value between 0 and 1 using a sigmoid.

        return BayerPattern(best_pattern, best_offset[0], best_offset[1], confidence)
    
    # Loops through all 4 patterns.
    # For each, it finds how much variation is in the green channel pixels (should be highest for the correct pattern).
    # Picks the one with the highest green channel variance.

    def _detect_blind(self, raw_data: np.ndarray) -> BayerPattern:
        green_channel_variance: dict[str, float] = {}
        
        for pattern in self._known_patterns:
            variance: float = self._calculate_green_variance(raw_data, pattern)
            green_channel_variance[pattern] = variance
            
        best_pattern: str = max(green_channel_variance, key=green_channel_variance.get)
        confidence: float = 0.8
        
        return BayerPattern(best_pattern, 0, 0, confidence)
    
    # Demosaics the raw image using the pattern and offset.
    # Resizes the reference image if needed.
    # Computes the mean squared error (MSE) between the demosaiced and reference image.
    # Returns negative MSE (because lower error is better, but higher score is desired).

    def _evaluate_pattern(self, raw_data: np.ndarray, reference: np.ndarray, 
                         pattern: str, offset_x: int, offset_y: int) -> float:
        demosaiced: np.ndarray = self._simple_demosaic(raw_data, pattern, offset_x, offset_y)
        
        if demosaiced.shape[:2] != reference.shape[:2]:
            reference = self._resize_reference(reference, demosaiced.shape[:2])
            
        mse: float = np.mean((demosaiced - reference) ** 2)
        return -mse
        
    def _simple_demosaic(self, raw_data: np.ndarray, pattern: str, 
                        offset_x: int, offset_y: int) -> np.ndarray:
        height: int = raw_data.shape[0]
        width: int = raw_data.shape[1]
        result: np.ndarray = np.zeros((height, width, 3), dtype=np.float32)
        
        pattern_map: dict[str, np.ndarray] = self._get_pattern_map(pattern)
        
        for y in range(height):
            for x in range(width):
                pattern_y: int = (y + offset_y) % 2
                pattern_x: int = (x + offset_x) % 2
                color_idx: int = pattern_map[pattern_y, pattern_x]
                result[y, x, color_idx] = raw_data[y, x]
                
        return result
        
    def _get_pattern_map(self, pattern: str) -> dict[str, np.ndarray]:
        maps: dict[str, np.ndarray] = {
            'RGGB': np.array([[0, 1], [1, 2]]),
            'GRBG': np.array([[1, 0], [2, 1]]),
            'GBRG': np.array([[1, 2], [0, 1]]),
            'BGGR': np.array([[2, 1], [1, 0]])
        }
        return maps[pattern]
    
    # Loops through the raw image in 2×2 blocks.
    # Collects all the pixels that should be green for the given pattern.
    # Computes variance (how much values change) of green pixels.
    #  Higher variance suggests the pattern is correct — because interpolation artifacts are reduced.
    def _calculate_green_variance(self, raw_data: np.ndarray, pattern: str) -> float:
        pattern_map: np.ndarray = self._get_pattern_map(pattern)
        green_pixels: list[float] = []
        
        height: int = raw_data.shape[0]
        width: int = raw_data.shape[1]
        
        for y in range(0, height, 2):
            for x in range(0, width, 2):
                for dy in range(2):
                    for dx in range(2):
                        if y + dy < height and x + dx < width:
                            if pattern_map[dy, dx] == 1:
                                green_pixels.append(raw_data[y + dy, x + dx])
                                
        return np.var(green_pixels) if green_pixels else 0.0
    
    # Ensures the reference image matches the demosaiced image size (for MSE computation).
    # Uses skimage.transform.resize.
    def _resize_reference(self, image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        from skimage.transform import resize
        return resize(image, target_shape, anti_aliasing=True, preserve_range=True)


class DemosaicingModule(Module):
    # Fills in the missing color values from the Bayer pattern. Each pixel
    # only sees one color (R, G, or B) so we need to guess the other two.
    # Bilinear is the go-to method - just averages neighboring pixels.
    # Simple is faster but worse. Advanced tries to be smart about edges.
    # In a raw Bayer image:
    # Each pixel only captures one color: Red, Green, or Blue.
    # To reconstruct a full-color image, we need to fill in the missing two colors for each pixel.


    def __init__(self) -> None:
        super().__init__()
        self._algorithms: dict[str, Any] = {
            'bilinear': self._bilinear_demosaic,
            'simple': self._simple_average_demosaic,
            'advanced': self._edge_aware_demosaic
        }

    # Runs the selected demosaicing method.
    # Measures how long it takes.
    def forward(self, raw_data: np.ndarray, bayer_pattern: BayerPattern, 
               algorithm: str = 'bilinear') -> DemosaicResult:
        start_time: float = time.time()
        
        if algorithm not in self._algorithms:
            algorithm = 'bilinear'
            
        demosaic_func: Any = self._algorithms[algorithm]
        rgb_image: np.ndarray = demosaic_func(raw_data, bayer_pattern)
        
        processing_time: float = time.time() - start_time
        
        return DemosaicResult(
            image=rgb_image,
            bayer_pattern=bayer_pattern,
            algorithm=algorithm,
            processing_time=processing_time
        )
    
    # Bilinear Demosaicing  fast and fairly accurate.

    def _bilinear_demosaic(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        # The fast demosaicing that actually works. Uses convolution
        # with a 3x3 kernel to average neighboring pixels. Way faster
        # than the interpolation method we tried first (300s -> 7s!).
        from scipy.signal import convolve2d
        
        height: int = raw_data.shape[0]
        width: int = raw_data.shape[1]
        result: np.ndarray = np.zeros((height, width, 3), dtype=np.float32)
        
        # 3x3 averaging kernel - just all ones
        kernel: np.ndarray = np.ones((3, 3), dtype=np.float32)
        
        # Create masks showing where each color lives in the pattern
        mask_r: np.ndarray = np.zeros_like(raw_data, dtype=np.float32)
        mask_g: np.ndarray = np.zeros_like(raw_data, dtype=np.float32)
        mask_b: np.ndarray = np.zeros_like(raw_data, dtype=np.float32)
        
        # RGGB pattern: R at (0,0), G at (0,1) and (1,0), B at (1,1)
        if bayer_pattern.pattern == 'RGGB':
            mask_r[0::2, 0::2] = 1  # Every other pixel starting at (0,0)
            mask_g[0::2, 1::2] = 1  # Green pixels in first row
            mask_g[1::2, 0::2] = 1  # More green pixels in second row
            mask_b[1::2, 1::2] = 1  # Blue at (1,1)
        else:
            # Default to RGGB if pattern is weird
            mask_r[0::2, 0::2] = 1
            mask_g[0::2, 1::2] = 1
            mask_g[1::2, 0::2] = 1
            mask_b[1::2, 1::2] = 1
            
        # Fill each color channel using convolution
        for c, mask in enumerate([mask_r, mask_g, mask_b]):
            # Convolve actual values and the mask separately
            numerator: np.ndarray = convolve2d(mask * raw_data, kernel, mode='same', boundary='symm')
            denominator: np.ndarray = convolve2d(mask, kernel, mode='same', boundary='symm')
            # Avoid division by zero with tiny epsilon
            denominator = np.where(denominator == 0, 1e-10, denominator)
            result[:, :, c] = numerator / denominator
            
        return result
    
    # Fast but lower quality.
    def _simple_average_demosaic(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        height: int = raw_data.shape[0]
        width: int = raw_data.shape[1]
        result: np.ndarray = np.zeros((height, width, 3), dtype=np.float32)
        
        pattern_map: np.ndarray = self._get_pattern_array(bayer_pattern.pattern)
        
        for y in range(height):
            for x in range(width):
                pattern_y: int = (y - bayer_pattern.offset_y) % 2
                pattern_x: int = (x - bayer_pattern.offset_x) % 2
                color_idx: int = pattern_map[pattern_y, pattern_x]
                result[y, x, color_idx] = raw_data[y, x]
                
        for c in range(3):
            result[:, :, c] = self._fill_missing_simple(result[:, :, c])  #fills in the zeros by averaging nearby known pixels 
            
        return result
        
    def _edge_aware_demosaic(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        result: np.ndarray = self._bilinear_demosaic(raw_data, bayer_pattern)
        
        for c in range(3):
            result[:, :, c] = self._apply_edge_correction(result[:, :, c], raw_data)
            
        return result
    
    # Determine what color a pixel is, depending on its position.
    def _get_pattern_array(self, pattern: str) -> np.ndarray:
        patterns: dict[str, np.ndarray] = {
            'RGGB': np.array([[0, 1], [1, 2]]),
            'GRBG': np.array([[1, 0], [2, 1]]),
            'GBRG': np.array([[1, 2], [0, 1]]),
            'BGGR': np.array([[2, 1], [1, 0]])
        }
        return patterns[pattern]
        
    #  Fills missing (zero) values in a color channel.
    def _fill_missing_simple(self, channel: np.ndarray) -> np.ndarray:
        from scipy.ndimage import convolve
        kernel: np.ndarray = np.ones((3, 3), dtype=np.float32)
        kernel[1, 1] = 0
        
        mask: np.ndarray = (channel > 0).astype(np.float32)
        
        numerator: np.ndarray = convolve(channel, kernel, mode='constant', cval=0.0)
        denominator: np.ndarray = convolve(mask, kernel, mode='constant', cval=0.0)
        
        denominator = np.where(denominator == 0, 1, denominator)
        
        filled: np.ndarray = channel.copy()
        zero_mask: np.ndarray = channel == 0
        filled[zero_mask] = numerator[zero_mask] / denominator[zero_mask]
        
        return filled

    # Applies the Sobel operator (edge detection) in X and Y directions. Combines both into a magnitude map. Gets a mask of strong edges. 
    def _apply_edge_correction(self, channel: np.ndarray, raw_data: np.ndarray) -> np.ndarray:
        from scipy.ndimage import sobel
        
        edge_x: np.ndarray = sobel(raw_data, axis=1)
        edge_y: np.ndarray = sobel(raw_data, axis=0)
        edge_magnitude: np.ndarray = np.sqrt(edge_x**2 + edge_y**2)
        
        edge_threshold: float = np.percentile(edge_magnitude, 90)
        edge_mask: np.ndarray = edge_magnitude > edge_threshold
        
        corrected: np.ndarray = channel.copy()
        return corrected


class GammaCorrector(Module):
    # Makes dark images brighter. The professor wants gamma=0.3 which
    # is pretty aggressive brightening. Also tries log and sigmoid
    # curves but power law works best. Uses percentiles to ignore
    # outliers that would mess up the normalization.
    def __init__(self) -> None:
        super().__init__()
        self._curve_functions: dict[str, Any] = {
            'power': self._power_curve,
            'log': self._log_curve,
            'sigmoid': self._sigmoid_curve
        }

    # Rescales pixel values to the [0, 1] range.   
    def forward(self, image: np.ndarray, settings: GammaSettings) -> np.ndarray:
        normalized: np.ndarray = self._normalize_with_percentiles(
            image, settings.percentile_low, settings.percentile_high
        )
        
        curve_func: Any = self._curve_functions.get(settings.curve_type, self._power_curve)
        corrected: np.ndarray = curve_func(normalized, settings.gamma)
        
        return self._denormalize(corrected, image)
        
    def _normalize_with_percentiles(self, image: np.ndarray, 
                                   low_percentile: float, high_percentile: float) -> np.ndarray:
        # Uses percentiles instead of min/max to handle outliers.
        # 0.01 and 99.99 percentiles ignore the extreme pixels that
        # would otherwise dominate the normalization.
        low_val: float = np.percentile(image, low_percentile)
        high_val: float = np.percentile(image, high_percentile)
        
        # Scale to [0,1] range for gamma correction
        normalized: np.ndarray = (image - low_val) / (high_val - low_val)
        normalized = np.clip(normalized, 0, 1)
        
        return normalized
        
    def _power_curve(self, image: np.ndarray, gamma: float) -> np.ndarray:
        return np.power(image, gamma)
        
    def _log_curve(self, image: np.ndarray, strength: float) -> np.ndarray:
        epsilon: float = 1e-10
        return np.log(1 + strength * image) / np.log(1 + strength)
        
    def _sigmoid_curve(self, image: np.ndarray, steepness: float) -> np.ndarray:
        midpoint: float = 0.5
        return 1 / (1 + np.exp(-steepness * (image - midpoint)))
        
    def _denormalize(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        return image


class WhiteBalancer(Module):
    # Fixes color casts in images. Gray world algorithm assumes the
    # average of the image should be gray. We calculate how far off
    # each channel is and scale them to match. Critical: we DON'T
    # clip values to preserve HDR range!
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, image: np.ndarray, method: str = 'gray_world') -> WhiteBalanceResult:
        if method == 'gray_world':
            balanced, factors = self._gray_world_balance(image)
        else:
            balanced, factors = self._simple_balance(image)
            
        return WhiteBalanceResult(
            image=balanced,
            scale_factors=factors,
            method=method
        )
        
    def _gray_world_balance(self, image: np.ndarray) -> tuple[np.ndarray, ColorScaleFactors]:
        # Gray world in action. Calculate mean of each channel, then
        # figure out how to scale them so they're all equal (gray).
        img_norm: np.ndarray = image.astype(np.float32)
        
        # Not clipping here to preserve the HDR range
        # Gives us values way above 1.0 which is what we want for HDR
        
        # Get average of each color channel
        mean_r: float = np.mean(img_norm[:, :, 0])
        mean_g: float = np.mean(img_norm[:, :, 1])
        mean_b: float = np.mean(img_norm[:, :, 2])
        
        # Target gray is the average of all three
        mean_gray: float = (mean_r + mean_g + mean_b) / 3.0
        
        # Figure out how much to scale each channel
        scale_r: float = mean_gray / mean_r if mean_r > 0 else 1.0
        scale_g: float = mean_gray / mean_g if mean_g > 0 else 1.0
        scale_b: float = mean_gray / mean_b if mean_b > 0 else 1.0
        
        # Apply the scaling
        balanced: np.ndarray = np.zeros_like(img_norm)
        balanced[:, :, 0] = img_norm[:, :, 0] * scale_r
        balanced[:, :, 1] = img_norm[:, :, 1] * scale_g
        balanced[:, :, 2] = img_norm[:, :, 2] * scale_b
        
        # Not clipping to [0,1] - keeping full HDR dynamic range
        
        scale_factors: ColorScaleFactors = ColorScaleFactors(scale_r, scale_g, scale_b)
        
        return balanced, scale_factors
        
    def _simple_balance(self, image: np.ndarray) -> tuple[np.ndarray, ColorScaleFactors]:
        return self._gray_world_balance(image)


class LinearityAnalyzer(Module):
    # Tests if the camera sensor responds linearly to light. Takes
    # multiple exposures and plots average brightness vs exposure time.
    # Should be straight lines if the sensor is linear (spoiler: it is!).
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, image_paths: list[str], exposure_times: list[float], 
               output_path: str | None = None) -> list[ExposureData]:
        exposure_data_list: list[ExposureData] = []
        
        print(f"[INFO] Analyzing linearity for {len(image_paths)} images...")
        for path, exposure in tqdm(zip(image_paths, exposure_times), 
                                  total=len(image_paths), desc="Calculating averages"):
            averages: PixelAverages = self._calculate_averages(path)
            exposure_data: ExposureData = ExposureData(
                filename=path,
                exposure_time=exposure,
                pixel_averages=averages
            )
            exposure_data_list.append(exposure_data)
            
        if output_path:
            self._plot_linearity(exposure_data_list, output_path)
            
        return exposure_data_list
        
    def _calculate_averages(self, image_path: str) -> PixelAverages:
        # Calculates average brightness for each color channel.
        # For RAW files, we need to pick out R/G/B pixels from the
        # Bayer pattern. For JPGs, it's already demosaiced.
        
        if image_path.endswith('.CR3'):
            raw: Any = rawpy.imread(image_path)
            raw_data: np.ndarray = raw.raw_image_visible
            
            # Assuming RGGB pattern
            pattern: np.ndarray = np.array([[0, 1], [1, 2]])
            
            red_pixels: list[float] = []
            green_pixels: list[float] = []
            blue_pixels: list[float] = []
            
            height: int = raw_data.shape[0]
            width: int = raw_data.shape[1]
            
            # Extract pixels based on their position in the 2x2 pattern
            for y in range(0, height - 1, 2):
                for x in range(0, width - 1, 2):
                    red_pixels.append(raw_data[y, x])  # Top-left is red
                    green_pixels.append(raw_data[y, x + 1])  # Top-right is green
                    green_pixels.append(raw_data[y + 1, x])  # Bottom-left is green too
                    blue_pixels.append(raw_data[y + 1, x + 1])  # Bottom-right is blue
                    
            return PixelAverages(
                red=np.mean(red_pixels),
                green=np.mean(green_pixels),
                blue=np.mean(blue_pixels)
            )
        else:
            # JPG files are already RGB
            image: np.ndarray = io.imread(image_path)
            return PixelAverages(
                red=np.mean(image[:, :, 0]),
                green=np.mean(image[:, :, 1]),
                blue=np.mean(image[:, :, 2])
            )
            
    def _plot_linearity(self, exposure_data: list[ExposureData], output_path: str) -> None:
        exposures: list[float] = [ed.exposure_time for ed in exposure_data]
        reds: list[float] = [ed.pixel_averages.red for ed in exposure_data]
        greens: list[float] = [ed.pixel_averages.green for ed in exposure_data]
        blues: list[float] = [ed.pixel_averages.blue for ed in exposure_data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(exposures, reds, 'r-o', label='Red')
        plt.plot(exposures, greens, 'g-o', label='Green')
        plt.plot(exposures, blues, 'b-o', label='Blue')
        
        plt.xlabel('Exposure Time (seconds)')
        plt.ylabel('Average Pixel Value')
        plt.title('Sensor Linearity Test')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class HDRCombiner(Module):
    # Merges multiple exposures into one HDR image. The trick is to
    # use dark exposures for bright areas (like windows) and bright
    # exposures for dark areas (like shadows). Uses threshold replacement
    # from the lecture slides.
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, raw_files: list[str], method: str = 'weighted_average') -> HDRImage:
        raw_images: list[np.ndarray] = []
        exposure_times: list[float] = []
        
        print(f"[INFO] Loading {len(raw_files)} RAW files for HDR...")
        for i, file_path in tqdm(enumerate(raw_files), total=len(raw_files), desc="Loading RAW files"):
            raw: Any = rawpy.imread(file_path)
            raw_data: np.ndarray = raw.raw_image_visible.astype(np.float32)
            raw_images.append(raw_data)
            
            exposure_time: float = 0.5 ** i
            exposure_times.append(exposure_time)
            
        print(f"[INFO] Merging using {method} method...")
        if method == 'weighted_average':
            hdr_data: np.ndarray = self._weighted_average_merge(raw_images, exposure_times)
        else:
            hdr_data = self._simple_merge(raw_images, exposure_times)
            
        return HDRImage(
            raw_data=hdr_data,
            exposure_times=exposure_times,
            merge_method=method,
            bit_depth=32
        )
        
    def _weighted_average_merge(self, images: list[np.ndarray], 
                               exposures: list[float]) -> np.ndarray:
        # The algorithm from lecture. Start with brightest exposure,
        # then replace saturated areas with data from darker exposures.
        print(f"[INFO] Using lecture algorithm...")
        hdr: np.ndarray = images[0].astype(np.float32)
        
        for i in range(1, len(images)):
            # Scale the darker image to match brightness of first
            scale: float = exposures[0] / exposures[i]
            img_scaled: np.ndarray = images[i] * scale
            
            # Find pixels close to saturation (80% of max)
            threshold: float = 0.8 * np.max(hdr)
            mask: np.ndarray = hdr >= threshold
            # Replace saturated pixels with data from darker exposure
            hdr[mask] = img_scaled[mask]
            
            print(f"  - Merged exposure {i+1}/{len(images)}, threshold: {threshold:.1f}")
        
        return hdr
        
    def _simple_merge(self, images: list[np.ndarray], exposures: list[float]) -> np.ndarray:
        hdr: np.ndarray = np.zeros_like(images[0], dtype=np.float32)
        
        for img, exposure in zip(images, exposures):
            hdr += img / exposure
            
        hdr /= len(images)
        
        return hdr
        
    def _calculate_weight(self, image: np.ndarray) -> np.ndarray:
        normalized: np.ndarray = image / np.max(image)
        weight: np.ndarray = normalized * (1 - normalized)
        return weight


class ToneMappingModule(Module):
    # Compresses HDR values (can be millions) down to displayable range.
    # Log compression is simple and works. iCAM06 is fancy but has
    # artifacts when HDR values are too high (ours go to 1.4M!).
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, hdr_image: np.ndarray, method: str = 'log', 
               output_range: tuple[float, float] = (0, 255)) -> np.ndarray:
        if method == 'log':
            tone_mapped: np.ndarray = self._log_tone_mapping(hdr_image)
        elif method == 'icam06':
            tone_mapped = self._icam06_tone_mapping(hdr_image)
        else:
            tone_mapped = self._simple_tone_mapping(hdr_image)
            
        scaled: np.ndarray = self._scale_to_range(tone_mapped, output_range)
        return scaled
        
    def _log_tone_mapping(self, hdr: np.ndarray) -> np.ndarray:
        epsilon: float = 1e-6
        log_hdr: np.ndarray = np.log(hdr + epsilon)
        
        min_val: float = np.min(log_hdr)
        max_val: float = np.max(log_hdr)
        
        normalized: np.ndarray = (log_hdr - min_val) / (max_val - min_val)
        return normalized
        
    def _icam06_tone_mapping(self, hdr: np.ndarray) -> np.ndarray:
        # iCAM06 is this fancy tone mapping that separates base layer
        # (overall brightness) from detail layer (textures). Uses bilateral
        # filter to preserve edges. Works great for normal HDR but our
        # 1.4M values cause those pink/green artifacts.
        import cv2
        
        # Note: iCAM06 was designed for typical HDR ranges (0-5000ish)
        # but our HDR merge gives us values up to 1.4M which causes
        # artifacts. Should normalize input but keeping raw values
        # to show HDR preservation throughout the pipeline
        
        epsilon: float = 1e-8
        if len(hdr.shape) == 3:
            # Convert RGB to intensity using perceptual weights
            intensity: np.ndarray = (20 * hdr[:, :, 0] + 40 * hdr[:, :, 1] + hdr[:, :, 2]) / 61.0 + epsilon
            # Store chromaticity (color ratios)
            r: np.ndarray = hdr[:, :, 0] / intensity
            g: np.ndarray = hdr[:, :, 1] / intensity
            b: np.ndarray = hdr[:, :, 2] / intensity
        else:
            intensity = hdr + epsilon
            r = g = b = np.ones_like(intensity)
        
        # Work in log domain
        log_I: np.ndarray = np.log(intensity)
        
        # Bilateral filter to get base layer (smooth brightness)
        log_base: np.ndarray = cv2.bilateralFilter(
            log_I.astype(np.float32), 
            d=9,  # Increased for HDR
            sigmaColor=1.0,  # Increased for HDR range
            sigmaSpace=5  # Increased for better spatial smoothing
        )
        
        # Detail layer is the difference
        log_detail: np.ndarray = log_I - log_base
        
        # Compress the base layer to fit display range
        output_range: float = 4.0
        compression: float = np.log(output_range) / (np.max(log_base) - np.min(log_base) + epsilon)
        
        log_offset: float = -np.max(log_base) * compression
        
        # Reconstruct: compressed base + original details
        log_output: np.ndarray = log_base * compression + log_offset + log_detail
        output_intensity: np.ndarray = np.exp(log_output)
        
        # Put the color back
        if len(hdr.shape) == 3:
            rgb_out: np.ndarray = np.stack([
                r * output_intensity, 
                g * output_intensity, 
                b * output_intensity
            ], axis=-1)
        else:
            rgb_out = output_intensity
        
        # Normalize to [0,1] for display
        rgb_normalized: np.ndarray = (rgb_out - np.min(rgb_out)) / (np.max(rgb_out) - np.min(rgb_out))
        
        return np.clip(rgb_normalized, 0, 1)
        
    def _simple_tone_mapping(self, hdr: np.ndarray) -> np.ndarray:
        return hdr / (1 + hdr)
        
    def _compress_dynamic_range(self, image: np.ndarray) -> np.ndarray:
        return np.log(1 + image) / np.log(1 + np.max(image))
        
    def _scale_to_range(self, image: np.ndarray, output_range: tuple[float, float]) -> np.ndarray:
        min_out: float = output_range[0]
        max_out: float = output_range[1]
        
        scaled: np.ndarray = image * (max_out - min_out) + min_out
        return np.clip(scaled, min_out, max_out)


class TaskVisualizer(Module):
    # Creates all the pretty pictures for the submission. Each task
    # needs specific visualizations. Handles all the matplotlib stuff
    # so the main code stays clean.
    def __init__(self) -> None:
        super().__init__()
        self._figures_path: str = "figures"
        Path(self._figures_path).mkdir(exist_ok=True)
        
    def visualize_bayer_pattern(self, raw_data: np.ndarray, output_name: str = "task1_bayer_pattern.png") -> None:
        crop: tuple[int, int, int, int] = (1400, 1700, 2800, 2900)
        y0, y1, x0, x1 = crop
        
        h, w = raw_data.shape
        rgb: np.ndarray = np.zeros((h, w, 3), dtype=np.float32)
        
        rgb[0::2, 0::2, 0] = raw_data[0::2, 0::2]
        rgb[0::2, 1::2, 1] = raw_data[0::2, 1::2]
        rgb[1::2, 0::2, 1] = raw_data[1::2, 0::2]
        rgb[1::2, 1::2, 2] = raw_data[1::2, 1::2]
        
        rgb = rgb / raw_data.max()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb[y0:y1, x0:x1])
        plt.title("Bayer Pattern Visualization (RGGB)")
        plt.axis('off')
        plt.savefig(f"{self._figures_path}/{output_name}", dpi=150, bbox_inches='tight')
        plt.close()
        
    def create_comparison_plot(self, demosaiced: np.ndarray, gamma_corrected: np.ndarray, 
                              white_balanced: np.ndarray, output_name: str = "task2-4_comparison.png") -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        def normalize_for_display(img: np.ndarray) -> np.ndarray:
            # Handle HDR values properly
            if np.max(img) > 10:  # Likely HDR
                # Use log compression for HDR
                epsilon: float = 1e-6
                img_log: np.ndarray = np.log(img + epsilon)
                img = (img_log - img_log.min()) / (img_log.max() - img_log.min())
            else:
                # Regular normalization
                img = img - img.min()
                img = img / img.max() if img.max() > 0 else img
            return np.clip(img, 0, 1)
        
        axes[0].imshow(normalize_for_display(demosaiced))
        axes[0].set_title("Demosaiced (no correction)")
        axes[0].axis('off')
        
        axes[1].imshow(normalize_for_display(gamma_corrected))
        axes[1].set_title("Gamma Corrected")
        axes[1].axis('off')
        
        axes[2].imshow(normalize_for_display(white_balanced))
        axes[2].set_title("Gamma + Gray World WB")
        axes[2].axis('off')
        
        plt.savefig(f"{self._figures_path}/{output_name}", dpi=150, bbox_inches='tight')
        plt.close()
        
    def save_hdr_raw(self, hdr_data: np.ndarray, output_name: str = "task6_raw_hdr.png") -> None:
        plt.figure(figsize=(10, 8))
        plt.imshow(hdr_data, cmap='gray')
        plt.title("Raw HDR Image")
        plt.axis('off')
        plt.savefig(f"{self._figures_path}/{output_name}", dpi=150, bbox_inches='tight')
        plt.close()
        
    def save_hdr_processed(self, hdr_rgb: np.ndarray, output_name: str = "task6_processed_hdr.png") -> None:
        # HDR values can be huge (we get 1.4M!) so need log compression
        # to see anything. This is just for visualization - the actual
        # HDR data stays intact.
        epsilon: float = 1e-6
        
        # Check if this is actually HDR (values > 1)
        max_val: float = np.max(hdr_rgb)
        print(f"[DEBUG] HDR max value before log: {max_val}")
        
        # Apply log compression for HDR
        hdr_log: np.ndarray = np.log(hdr_rgb + epsilon)
        hdr_log_min: float = np.min(hdr_log)
        hdr_log_max: float = np.max(hdr_log)
        hdr_log_normalized: np.ndarray = (hdr_log - hdr_log_min) / (hdr_log_max - hdr_log_min)
        hdr_display: np.ndarray = (hdr_log_normalized * 255).astype(np.uint8)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(hdr_display)
        plt.title("HDR Image (Demosaiced + WB)")
        plt.axis('off')
        plt.savefig(f"{self._figures_path}/{output_name}", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save another copy for good measure
        io.imsave(f"{self._figures_path}/processed_hdr_image.png", hdr_display)
        

class ImageProcessor(Module):
    # Main workhorse that orchestrates everything. This is what gets
    # called by process_raw() for Task 8. Creates all the other modules
    # and runs them in the right order.
    def __init__(self) -> None:
        super().__init__()
        self._bayer_detector: BayerPatternDetector = BayerPatternDetector()
        self._demosaicer: DemosaicingModule = DemosaicingModule()
        self._gamma_corrector: GammaCorrector = GammaCorrector()
        self._white_balancer: WhiteBalancer = WhiteBalancer()
        self._linearity_analyzer: LinearityAnalyzer = LinearityAnalyzer()
        self._hdr_combiner: HDRCombiner = HDRCombiner()
        self._tone_mapper: ToneMappingModule = ToneMappingModule()
        
    def process_raw(self, input_path: str, output_path: str) -> None:
        # Task 8: The full pipeline from RAW to JPG. Combines everything
        # we did in tasks 2-4. Takes about 7 seconds for a full res image.
        print(f"[INFO] Processing {input_path}")
        start_time: float = time.time()
        
        # Load the raw sensor data
        raw: Any = rawpy.imread(input_path)
        raw_data: np.ndarray = raw.raw_image_visible
        
        # We just assume RGGB - auto detection was flaky
        print(f"[Stage 1/4] Using RGGB Bayer pattern...")
        bayer_pattern: BayerPattern = BayerPattern('RGGB', 0, 0, 1.0)
        
        # Fill in missing colors
        print(f"[Stage 2/4] Demosaicing with bilinear interpolation...")
        demosaic_result: DemosaicResult = self._demosaicer(raw_data, bayer_pattern)
        
        # Make it brighter (gamma 0.3 is aggressive)
        print(f"[Stage 3/4] Applying gamma correction...")
        gamma_settings: GammaSettings = GammaSettings(0.3, 0.01, 99.99, 'power')
        gamma_corrected: np.ndarray = self._gamma_corrector(demosaic_result.image, gamma_settings)
        
        # Fix color cast
        print(f"[Stage 4/4] Applying white balance...")
        wb_result: WhiteBalanceResult = self._white_balancer(gamma_corrected)
        
        # Convert to 8-bit and save with high quality
        final_image: np.ndarray = (wb_result.image * 255).astype(np.uint8)
        import imageio.v3 as iio  # Fixed the FutureWarning
        iio.imwrite(output_path, final_image, quality=99)
        
        total_time: float = time.time() - start_time
        print(f"[SUCCESS] Processed in {total_time:.2f} seconds")
        print(f"[SUCCESS] Saved to: {output_path}")
        
        return demosaic_result, gamma_corrected, wb_result


if __name__ == "__main__":
    # Run all tasks when executed directly. Creates all the output
    # images for submission. Takes about 2 minutes total.
    print("=" * 60)
    print("DEMOSAICING & HDR IMPLEMENTATION - FULL PIPELINE")
    print("=" * 60)
    
    base_path: str = "/Users/rahulsawhney/LocalCode/__FAU/12) Computer Vision Project (Majors-2)/GroupExercise-4/exercise_4_data"
    
    processor: ImageProcessor = ImageProcessor()
    visualizer: TaskVisualizer = TaskVisualizer()
    
    print("\n[Task 1] Investigating Bayer Pattern...")
    bayer_data: np.ndarray = np.load(f"{base_path}/01/IMG_9939.npy")
    visualizer.visualize_bayer_pattern(bayer_data)
    print("[SUCCESS] Saved: figures/task1_bayer_pattern.png")
    
    print("\n[Tasks 2-4] Demosaicing, Gamma, and White Balance...")
    test_file: str = f"{base_path}/02/IMG_4782.CR3"
    demosaic_result, gamma_corrected, wb_result = processor.process_raw(
        test_file, "figures/task8_final.jpg"
    )
    
    visualizer.create_comparison_plot(
        demosaic_result.image, 
        gamma_corrected, 
        wb_result.image
    )
    print("[SUCCESS] Saved: figures/task2-4_comparison.png")
    
    print("\n[Task 5] Testing linearity analysis...")
    analyzer: LinearityAnalyzer = LinearityAnalyzer()
    
    # 6 photos with halving exposure times
    cr3_files: list[str] = [
        f"{base_path}/05/IMG_3044.CR3",
        f"{base_path}/05/IMG_3045.CR3",
        f"{base_path}/05/IMG_3046.CR3",
        f"{base_path}/05/IMG_3047.CR3",
        f"{base_path}/05/IMG_3048.CR3",
        f"{base_path}/05/IMG_3049.CR3"
    ]
    
    # Each exposure is half of the previous one
    exposure_times: list[float] = [1/10, 1/20, 1/40, 1/80, 1/160, 1/320]
    
    analyzer(cr3_files, exposure_times, "figures/task5_linearity.png")
    print("[SUCCESS] Saved: figures/task5_linearity.png")
    
    print("\n[Task 6] HDR Implementation...")
    hdr_combiner: HDRCombiner = HDRCombiner()
    
    # 11 exposures from bright (00) to dark (10)
    hdr_files: list[str] = [f"{base_path}/06/{i:02d}.CR3" for i in range(11)]
    
    hdr_result: HDRImage = hdr_combiner(hdr_files)
    print(f"[INFO] HDR raw data shape: {hdr_result.raw_data.shape}")
    
    visualizer.save_hdr_raw(hdr_result.raw_data)
    print("[SUCCESS] Saved: figures/task6_raw_hdr.png")
    
    import imageio.v2 as imageio
    imageio.imwrite("figures/hdr.tiff", hdr_result.raw_data.astype(np.uint16))
    print("[SUCCESS] Saved: figures/hdr.tiff")
    
    print("\n[INFO] Demosaicing and white balancing HDR...")
    bayer_pattern: BayerPattern = BayerPattern('RGGB', 0, 0, 1.0)
    demosaic_module: DemosaicingModule = DemosaicingModule()
    white_balancer: WhiteBalancer = WhiteBalancer()
    
    hdr_demosaic_result: DemosaicResult = demosaic_module(hdr_result.raw_data, bayer_pattern)
    hdr_wb_result: WhiteBalanceResult = white_balancer(hdr_demosaic_result.image)
    
    visualizer.save_hdr_processed(hdr_wb_result.image)
    print("[SUCCESS] Saved: figures/task6_processed_hdr.png")
    
    print("\n[Task 7] iCAM06 Tone Mapping...")
    tone_mapper: ToneMappingModule = ToneMappingModule()
    
    # Debug: Check input range
    print(f"[DEBUG] iCAM06 input min: {np.min(hdr_wb_result.image)}, max: {np.max(hdr_wb_result.image)}")
    
    # Apply iCAM06 - will have artifacts due to extreme HDR values
    icam06_result: np.ndarray = tone_mapper(hdr_wb_result.image, method='icam06', output_range=(0, 1))
    
    # Debug: Check output range
    print(f"[DEBUG] iCAM06 output min: {np.min(icam06_result)}, max: {np.max(icam06_result)}")
    
    # Scale to 8-bit for display
    icam06_display: np.ndarray = (icam06_result * 255).astype(np.uint8)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(icam06_display)
    plt.title("iCAM06 Tone Mapped Image")
    plt.axis('off')
    plt.savefig("figures/task7_icam06.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Saved: figures/task7_icam06.png")
    
    print("\n[COMPLETE] All tasks finished!")
    print("\nGenerated outputs in figures/ directory:")
    print("  1. task1_bayer_pattern.png")
    print("  2. task2-4_comparison.png") 
    print("  3. task5_linearity.png")
    print("  4. task6_raw_hdr.png")
    print("  5. task6_processed_hdr.png")
    print("  6. task7_icam06.png")
    print("  7. task8_final.jpg")
    print("  8. hdr.tiff")
    print("=" * 60)