# 📦 Computer Vision: Demosaicing & High Dynamic Range (HDR)

This task implements omplete raw image processing pipeline from scratch. It covers the fundamental steps of converting raw sensor data into a viewable image, including demosaicing, color correction, and handling high dynamic range (HDR) scenes.
---

## 🚀 The Task Pipeline

The pipeline processes raw sensor data through a series of sequential steps to produce a final, high-quality image.

### **Raw Sensor Data & Bayer Pattern Investigation**
The process begins by loading the raw data and identifying its unique Bayer filter pattern.

### **Demosaicing**
The raw, single-channel data is converted into a full-color RGB image by interpolating the missing color values for each pixel.

### **Luminosity Correction**
The image's brightness is adjusted using a non-linear curve, such as gamma correction, to make it visually appealing.

### **White Balance**
Color casts are removed by applying an algorithm like "Gray World" to ensure that the colors in the image are accurate and neutral.

###**HDR Merging**
For scenes with a wide range of light, multiple exposures are combined to create a single HDR image that captures detail in both shadows and highlights.

###**Tone Mapping**
The high dynamic range of the combined image is compressed into a displayable range, allowing it to be viewed on a standard monitor.

###**Final Output**
The processed image is saved to a standard format like JPG.

---

## 🛠️ Implementation Details

The implementation is structured into a series of interconnected Python modules, each handling a specific part of the pipeline. This modular design makes the code clean and easy to maintain.

* **`Module`**: A base class that all other components inherit from, providing a consistent structure for each processing step.

* **`BayerPatternDetector`**: This module analyzes raw data to identify the **Bayer pattern** using both a "blind" method (based on the green channel's statistical properties) and a reference-based comparison.

* **`DemosaicingModule`**: Handles color interpolation using multiple algorithms. The **bilinear demosaicing** method is the primary implementation and relies on `scipy.signal.convolve2d` for efficient processing, significantly outperforming a basic loop-based approach.

* **`GammaCorrector`**: This class applies luminosity adjustments. It uses **percentile-based normalization** (`np.percentile`) to handle outliers and implements different tone curves, including power ($y=x^{\gamma}$), logarithmic, and sigmoid functions.

* **`WhiteBalancer`**: Implements the **Gray World algorithm** to correct color casts. It calculates the average color for each channel and scales the channels to match, importantly **without clipping values** to preserve the full dynamic range.

* **`LinearityAnalyzer`**: A utility module that verifies the linear response of the camera sensor. It loads raw `.CR3` files and plots the average red, green, and blue values against exposure time using `matplotlib`.

* **`HDRCombiner`**: This module merges multiple exposures. It uses an algorithm from the course lecture that replaces overexposed regions in a bright image with the corresponding, well-exposed data from a darker image.

* **`ToneMappingModule`**: Compresses the HDR data for display. It provides a simple **logarithmic** tone mapping and a more complex **iCAM06** method. The code notes that extreme HDR values (up to 1.4 million) can cause visual artifacts with the iCAM06 method.

* **`ImageProcessor`**: The main class that orchestrates the entire pipeline. Its `process_raw` function takes a raw file, runs it through the full pipeline of demosaicing, correction, and white balance, and saves the final result as a high-quality JPG using the `imageio` library.
