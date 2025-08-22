# 📦 Computer Vision: Box Detection

This task involved estimating the dimensions of a box from a 3D point cloud. The solution is implemented in Python and demonstrates key skills in data processing, geometric algorithms, and object measurement.

---

## 🛠️ Implementation & Code Structure

The task follows a **modular, class-based architecture** built on a `Module` base class, with each component handling a specific part of the processing pipeline. The core libraries used are **NumPy**, **SciPy**, and **Matplotlib**.

### `BoxDataset`

This class handles the **data loading and management**. It automatically discovers and loads the provided `.mat` files, structuring the data into a dictionary containing the amplitude image, distance image, and 3D point cloud.

### `Ransac`

I implemented a **custom RANSAC (Random Sample Consensus) algorithm** from scratch for robust plane detection. The algorithm:
- Randomly samples three points to propose a candidate plane model.
- Determines inliers by checking if points are within a specified distance threshold.
- Iterates to find the model with the largest number of inliers.
- This module is used to identify both the **floor plane** and, subsequently, the **box's top plane** after removing the floor points.

### `Mask`

This class is dedicated to **refining the binary masks** created by the RANSAC process. It leverages **SciPy's morphological operations** (`ndimage.binary_opening`, `ndimage.binary_closing`) to filter out noise and small, disconnected components. It also includes functionality to find and extract the **largest connected component** in a mask, ensuring the measurements are based on the primary box surface.

### `Geometry`

This module performs the final **dimensional calculations**. It computes the box's **height** by finding the perpendicular distance between the detected floor and box planes. For **length and width**, it analyzes the 3D coordinates of the points at the corners of the box mask and uses a diagonal averaging method for a more stable result.

### `Viz`

The class handles all **visualization** for the project. It generates and saves a series of figures, including:
- The raw input data (amplitude and distance images).
- The intermediate floor and box masks, showing the effect of morphological filtering.
- A final color-coded visualization of the scene, highlighting the floor, box, and background. This visualization also includes the calculated dimensions and corner labels.

---

## 📝 Discussion

The task's final phase involved a critical analysis of the implemented algorithm. I identified potential weaknesses, such as its sensitivity to specific data assumptions and a lack of robustness against significant occlusions. A discussion of these limitations and suggestions for future improvements is included as part of the project deliverables.
