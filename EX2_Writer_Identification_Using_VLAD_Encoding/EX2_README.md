# 📦 Computer Vision: Writer Identification System

This task implements a writer identification system using the **Bag of Visual Words** model, evaluated on the ICDAR17 Historical WI benchmark dataset. The core of the system is a pipeline that transforms local image descriptors into a global vector representation for classification.

---

## 🚀 Overview of the Pipeline

The project follows a structured approach to identify writers by analyzing the unique characteristics of their handwriting. The pipeline consists of the following key stages:

### **1. Codebook Generation**
A **codebook** (or vocabulary) is created to serve as a set of visual "words." This is done by:
* Randomly sampling a large number of local descriptors from the training set.
* Applying the **k-means** clustering algorithm to these descriptors to create a dictionary of visual words.

### **2. VLAD Encoding**
With the codebook established, each image is encoded into a single, global vector representation using the **Vector of Locally Aggregated Descriptors (VLAD)** method. This process involves:
* Calculating residuals between each local descriptor and its nearest cluster center in the codebook.
* Aggregating these residuals to form the final VLAD vector.
* Evaluating the performance of this raw encoding using **Mean Average Precision (mAP)**.

### **3. VLAD Normalization**
To improve performance and account for visual "burstiness," the VLAD vectors are normalized. This involves:
* Applying **power normalization** to each element of the VLAD vector.
* Performing a subsequent **l2 normalization** of the entire vector.
* Comparing the new mAP score against the previous result to evaluate the impact of normalization.

### **4. Exemplar Classification**
The final step uses the normalized VLAD vectors for classification. This is done by:
* Training an individual **LinearSVC** for each test image's encoding, using it as a positive example against all training encodings as negatives.
* Using the SVM's weight vector as a new, highly discriminative global descriptor.
* Evaluating the final performance based on the mAP of these new descriptors, completing the pipeline.

---

## 🛠️ Implementation & Code Structure

The task follows a **modular, class-based architecture**, with each component inheriting from a `Module` base class. This ensures a clean and scalable pipeline.

### `SIFTDataset`
This class handles the **data loading** and management. It reads pre-computed SIFT descriptors from the provided `.pkl.gz` files.

### `DictionaryLearner`
This class handles the creation of the visual codebook. It uses `sklearn.cluster.MiniBatchKMeans` to efficiently cluster a large number of SIFT descriptors into a predefined number of "visual words."

### `VLADEncoder`
As the central part of the pipeline, this module computes the **VLAD vector** for each document. It aggregates the residual vectors for each cluster and also handles the crucial normalization steps:
* **Power Normalization**: Applied to reduce the impact of visually frequent patterns.
* **L2 Normalization**: Applied to ensure all encodings have a unit length, which is essential for accurate distance comparisons.

### `ESVMClassifier`
This module implements the **Exemplar SVM (E-SVM)** approach. It trains a `sklearn.svm.LinearSVC` for each test sample, treating it as the sole positive example against all training samples. To manage the high computational load, the training of these thousands of individual SVMs is **parallelized** using `joblib` and `multiprocessing`.

### `DistanceComputer`
This class is responsible for calculating the **cosine distance** between all pairs of encoded documents. This metric is used to rank documents and evaluate the system's performance via **Mean Average Precision (mAP)**.

### `VLADProcessor`
Acting as the main orchestrator, this class ties all the above modules together, providing a high-level interface to run the complete pipeline from codebook learning to final distance computation.
