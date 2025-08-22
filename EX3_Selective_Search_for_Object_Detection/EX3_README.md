# 📦 Computer Vision: Selective Search for Object Detection

This task implements the **Selective Search** algorithm for object detection. The goal is to generate high-quality object proposals (bounding boxes) for images across different domains: Art History, Christian Archaeology, and Classical Archaeology. The implementation involves a hierarchical process that merges initial image segments based on various similarity metrics.

---

## 🚀 The Selective Search Pipeline

The tasks are structured as a step-by-step pipeline to generate object proposals:

### **Initial Segmentation**
The process begins by using a graph-based algorithm, specifically the **Felzenszwalb algorithm**, to generate an initial set of small, homogeneous regions. This provides the base for the subsequent merging process.

### **Region Similarity Metrics**
Once the initial regions are obtained, the algorithm computes four distinct similarity metrics between all neighboring regions. These metrics are crucial for deciding which regions should be merged:
* **Color Similarity**: Based on color histograms.
* **Texture Similarity**: Based on texture histograms.
* **Size Similarity**: Prefers merging smaller regions.
* **Fill Similarity**: Measures how well two regions fit into their combined bounding box.

### **Hierarchical Merging**
The algorithm iteratively merges regions in a greedy, hierarchical fashion.
* A list of all neighboring regions is created.
* In each step of the hierarchy, the two most similar neighboring regions are **merged**.
* The old similarities involving the two merged regions are **removed**.
* New similarity values between the newly formed region and its neighbors are **calculated**.
This process continues until all regions are merged into a single region, resulting in a hierarchy of proposals.

### **Final Proposals**
The final task involves generating the full set of region proposals (bounding boxes) from the merging hierarchy. These proposals represent potential objects in the image.

---

## 🛠️ Implementation Details

The task is structured with a modular, class-based architecture, with each component inheriting from a `Module` base class. This ensures a clean and scalable pipeline.

### `SegmentationModule`
This class handles the initial image segmentation using the `skimage.segmentation.felzenszwalb` function, which produces hundreds of small, uniform regions to start the selective search process.

### `TextureGradientCalculator`
This module computes texture features using the `skimage.feature.local_binary_pattern` algorithm. It calculates a texture "signature" for each pixel, which is then used to create texture histograms for each region.

### `RegionExtractor`
This crucial module processes the initial segments to extract the properties needed for merging. For each region, it identifies its bounding box, pixel count, and generates both color and texture histograms. The process is **parallelized** using `ThreadPoolExecutor` to handle a large number of regions efficiently.

### `SimilarityCalculator`
This class is responsible for calculating the four similarity metrics. It uses a custom histogram intersection method to compare color and texture histograms. Size and fill similarities are calculated based on the regions' properties to guide the merging process.

### `NeighborFinder`
This module identifies which regions are adjacent to each other. The implementation is optimized and **parallelized** to avoid memory and performance bottlenecks that arise from checking every possible pair of regions in a large image.

### `RegionMerger`
This module handles the core hierarchical process. It takes two regions and combines them into a single, new region by merging their bounding boxes, summing their sizes, and creating a weighted average of their histograms.

### `BatchSimilarityProcessor`
During the iterative merging process, this class re-calculates similarities for all new neighbors. It uses a **parallel processing** approach with `ThreadPoolExecutor` to perform these calculations on large batches of pairs, significantly speeding up the algorithm's runtime.

### `SelectiveSearchProcessor`
This is the main orchestrator of the entire system. It coordinates all the other modules, from initial segmentation to the final hierarchical merging, to produce the complete set of object proposals for an image.
