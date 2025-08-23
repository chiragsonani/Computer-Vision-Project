# 📦 Computer Vision: Face Recognition System

This task is a comprehensive implementation of a basic **face recognition system** capable of both identification and re-identification. The system is divided into two core parts: a **Training Module** for capturing and processing facial data to build models, and a **Testing Module** for evaluating those models on video streams.

---

## Training Module

The training module (`training.py`) handles data acquisition and model building for both supervised face identification and unsupervised face clustering.

### Face Detection, Tracking, and Alignment

This section focuses on the preprocessing pipeline for video-based face recognition, using a modular, object-oriented design.

* **`FaceDetectorModule`**: Utilizes the **MTCNN** (Multi-task Convolutional Neural Network) to find the largest face bounding box in a frame.
* **`FaceTrackerModule`**: Employs a **hybrid approach** using template matching for efficient tracking, re-initializing with MTCNN when the correlation score drops below a set threshold. A `TrackingState` dataclass manages the state of the tracked face.
* **Alignment**: Normalizes detected faces to a fixed size of 224x224 pixels before feature extraction.

### Face Identification and Verification

This section implements a distance-based face identification system using deep features extracted by **FaceNet**. The `FaceRecognizerModule` is a custom k-NN classifier.

* **Feature Extraction**: A pre-trained **ResNet-50 CNN** (`cv2.dnn.readNetFromONNX`) generates 128-dimensional embeddings from aligned faces. The system uses L2-normalization for valid distance computations.
* **`partial_fit`**: Adds **dual embeddings** (color and grayscale) for a given identity to the gallery to improve robustness against illumination changes.
* **`predict`**: The core method for identification, which computes **Euclidean distances** to find the nearest neighbor.
* **Open-Set Protocol**: A face is classified as "unknown" if its distance to the nearest neighbor or its posterior probability falls below a set threshold.

### Face Clustering

This exercise explores unsupervised learning for face recognition by implementing **k-Means clustering from scratch**.

* **`KMeansEngine`**: Implements **Lloyd's algorithm**. It uses random initialization from the data points themselves to prevent empty clusters and iteratively assigns points to the nearest centroid until convergence.
* **`FaceClusteringModule`**: Orchestrates the clustering process. The `partial_fit` method collects embeddings without labels, and the `fit` method runs the k-Means algorithm.
* **Re-Identification**: The `predict` method assigns a new face embedding to the closest cluster, enabling person re-identification.

---

## Testing Module

The testing module (`test.py`) evaluates the performance of the trained models and visualizes the results.

### Evaluation of Face Recognition

This section focuses on evaluating the open-set face identification system's performance using **Detection and Identification Rate (DIR) curves**. The `EvaluationModule` handles this process.

* **Data Preparation**: The module loads pre-computed embeddings and labels from pickle files for standardized training and testing.
* **`run`**: Orchestrates the evaluation by iterating through a range of false alarm rates.
* **`select_similarity_threshold`**: For each false alarm rate, this method determines the corresponding similarity threshold using a **percentile-based selection**.
* **`calc_identification_rate`**: Computes the identification rate by comparing predictions for known subjects against their ground truth labels.
* **Analysis**: The final `run` method prints the optimal thresholds for both **security-critical** (low false alarms) and **user-friendly** (high identification rates) scenarios.

The task uses Python's `dataclasses` for immutable state management, with `logging` and `matplotlib` for detailed analysis and visualization.
