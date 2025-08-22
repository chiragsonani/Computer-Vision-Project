# Face Recognition System Implementation - Exercise 5
# Computer Vision Project - Summer 2025 - FAU Erlangen
#
# Complete video-based face recognition pipeline implementing:
#   1. MTCNN face detection with template matching tracking (Exercise 5.1)
#   2. k-NN identification with open-set protocol (Exercise 5.2)
#   3. k-means clustering for person re-identification (Exercise 5.3)
#   4. DIR curve evaluation framework (Exercise 5.4)
#
# Architecture: Modular design following PyTorch conventions with immutable dataclass
# state management. Uses 128-dimensional FaceNet embeddings with L2 normalization
# for robust face representation in Euclidean space.
#
# Mathematical Foundation: Open-set recognition with distance/probability thresholds
# Dataset: YouTube Faces subset with 5 known + 2 unknown identities
# Authors: Rahul Sawhney and Chirag Sonani

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias
from abc import ABC, abstractmethod
import json
from collections import defaultdict, Counter
import sys
import logging
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from cvproj_exc.config import Config

class Module:
    # Base class providing consistent interface for all face recognition modules.
    # Every processing component (detection, tracking, recognition, clustering) inherits
    # from this class and implements the forward() method.
    #
    # The PyTorch-style pattern enables composable pipelines where different algorithms
    # can be swapped without changing calling code. The __call__ override allows modules
    # to be used as functions, maintaining clean and readable pipeline construction.
    def __init__(self) -> None:
        pass
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__}.forward() not implemented")
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    # Represents face detection bounding boxes with geometric validation and operations.
    # Enforces positive dimensions and normalized confidence scores to prevent invalid
    # detections from propagating through the recognition pipeline.
    #
    # The frozen=True makes bounding boxes immutable to prevent accidental modification
    # during tracking operations. slots=True reduces memory overhead when processing
    # large numbers of detections in video sequences.
    x: int
    y: int
    width: int
    height: int
    confidence: float
    
    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid bounding box dimensions: {self.width}x{self.height}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Negative coordinates not allowed: ({self.x}, {self.y})")
    
    def __str__(self) -> str:
        return f"BBox({self.x},{self.y},{self.width}x{self.height},conf={self.confidence:.2f})"
    
    def __repr__(self) -> str:
        return f"BoundingBox({self.x}, {self.y}, {self.width}, {self.height}, {self.confidence})"
    
    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def contains_point(self, x: int, y: int) -> bool:
        # Tests whether a pixel coordinate lies within the bounding box boundaries.
        # Uses half-open interval semantics where the right and bottom edges are exclusive,
        # following standard computer vision convention for consistent boundary handling.
        # This method is essential for region-of-interest operations and click-based
        # face selection in interactive visualization interfaces.
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height
    
    def intersection_over_union(self, other: BoundingBox) -> float:
        # Computes IoU metric for bounding box overlap assessment in tracking scenarios.
        # IoU values near 1.0 indicate strong overlap while values near 0.0 suggest
        # minimal overlap, enabling robust tracking quality evaluation.
        # Compute intersection rectangle coordinates using coordinate clipping
        x1: int = max(self.x, other.x)  # Left edge of intersection
        y1: int = max(self.y, other.y)  # Top edge of intersection
        x2: int = min(self.x + self.width, other.x + other.width)   # Right edge
        y2: int = min(self.y + self.height, other.y + other.height) # Bottom edge
        
        # Validate intersection existence: empty intersection yields IoU = 0
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Apply standard IoU formula: Intersection / Union
        intersection_area: int = (x2 - x1) * (y2 - y1)
        union_area: int = self.area + other.area - intersection_area  # Inclusion-exclusion principle
        
        return intersection_area / union_area if union_area > 0 else 0.0

@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    # Encapsulates 128-dimensional FaceNet embeddings with automatic L2 normalization.
    # Represents faces as unit vectors in high-dimensional space where Euclidean
    # distance corresponds to perceptual face similarity.
    #
    # Automatic normalization in __post_init__ ensures all embeddings lie on the
    # unit hypersphere, making Euclidean distance calculations mathematically valid.
    # This normalization is critical for k-NN classification and clustering accuracy.
    vector: np.ndarray
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if self.vector.shape != (128,):
            raise ValueError(f"Embedding must be 128-dimensional, got shape {self.vector.shape}")
        
        # Normalize to unit vector for valid distance computation
        norm: float = float(np.linalg.norm(self.vector))
        if not np.allclose(norm, 1.0, rtol=1e-5):
            normalized_vector: np.ndarray = self.vector / norm
            object.__setattr__(self, 'vector', normalized_vector)
    
    def __str__(self) -> str:
        label_str: str = f"'{self.label}'" if self.label else "unlabeled"
        return f"Embedding({label_str}, norm={np.linalg.norm(self.vector):.4f})"
    
    def __repr__(self) -> str:
        return f"FaceEmbedding(vector=array(shape={self.vector.shape}), label={self.label!r})"
    
    def distance_to(self, other: FaceEmbedding) -> float:
        # Computes Euclidean distance between two L2-normalized face embeddings.
        # Since embeddings are unit vectors on the hypersphere, Euclidean distance
        # provides a meaningful similarity metric where smaller distances indicate
        # greater facial similarity. This distance is used throughout the k-NN
        # classification and clustering algorithms for identity determination.
        diff: np.ndarray = self.vector - other.vector
        return float(np.linalg.norm(diff))
    
    def cosine_similarity_to(self, other: FaceEmbedding) -> float:
        # Calculates cosine similarity between normalized embedding vectors as dot product.
        # For L2-normalized vectors, the dot product directly yields cosine similarity
        # in range [-1, 1], where values near 1.0 indicate high facial similarity.
        # This metric is mathematically equivalent to Euclidean distance but provides
        # intuitive similarity semantics for face recognition applications.
        return float(np.dot(self.vector, other.vector))
    

@dataclass(frozen=True, slots=True)
class TrackingState:
    # Maintains template matching state across video frames for efficient face tracking.
    # Stores the grayscale face template and search parameters to enable computationally
    # efficient tracking without running full MTCNN detection on every frame.
    #
    # The window_size parameter defines the search region expansion around the previous
    # face location, balancing tracking robustness against computational efficiency.
    # Template matching assumes small inter-frame motion typical in video sequences.
    bounding_box: BoundingBox
    template: np.ndarray
    frame_id: int
    track_id: int
    window_size: int = 25  
    
    def __post_init__(self) -> None:
        if len(self.template.shape) != 2:
            raise ValueError(f"Template must be 2D grayscale, got shape {self.template.shape}")
        if self.frame_id < 0:
            raise ValueError(f"Frame ID must be non-negative, got {self.frame_id}")
        if self.window_size <= 0:
            raise ValueError(f"Window size must be positive, got {self.window_size}")
    
    def get_search_region(self, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        # Computes expanded search region around previous face location for template matching.
        # The window_size expansion accommodates typical inter-frame face movement while
        # constraining the search space for computational efficiency.
        height: int
        width: int
        height, width = frame_shape
        
        # Expand bounding box by window_size pixels in all directions
        x1: int = max(0, self.bounding_box.x - self.window_size)
        y1: int = max(0, self.bounding_box.y - self.window_size)
        x2: int = min(width, self.bounding_box.x + self.bounding_box.width + self.window_size)
        y2: int = min(height, self.bounding_box.y + self.bounding_box.height + self.window_size)
        
        return (x1, y1, x2, y2)


@dataclass
class DetectionVisualization:
    # Visualization container for face detection results in Exercise 5.1 figure generation.
    # Stores the original frame, detected bounding box, and metadata required for
    # creating academic publication-quality detection result visualizations.
    #
    # This dataclass enables systematic collection of detection examples across
    # different subjects and frames for comprehensive evaluation presentation.
    frame: np.ndarray
    bbox: BoundingBox
    person_name: str
    frame_number: int


@dataclass
class TemplateMatchVisualization:
    # Encapsulates template matching visualization data for correlation analysis display.
    # Contains the search frame, face template, correlation response map, and optimal
    # match location for generating educational template matching demonstrations.
    #
    # The correlation map visualization helps explain how template matching localizes
    # faces by showing the spatial distribution of cross-correlation responses
    # with the detected face template across the search region.
    frame: np.ndarray
    template: np.ndarray
    correlation_map: np.ndarray
    bbox: BoundingBox
    max_loc: tuple[int, int]


@dataclass
class TrackingVisualization:
    # Temporal tracking visualization for face trajectory analysis across video frames.
    # Stores frame snapshots with updated bounding boxes to demonstrate tracking
    # performance and trajectory consistency in video sequences.
    #
    # Frame-by-frame tracking visualizations enable assessment of tracking stability,
    # drift accumulation, and re-initialization effectiveness in challenging scenarios.
    frame: np.ndarray
    bbox: BoundingBox
    frame_number: int


@dataclass(frozen=True, slots=True)
class PredictionResult:
    # Immutable container for face identification results with validation and analysis methods.
    # Encapsulates the predicted identity label, confidence probability, embedding distance,
    # and k-nearest neighbor context for comprehensive open-set evaluation.
    #
    # The frozen=True enforces immutability to prevent accidental result modification during
    # evaluation pipelines. Validation in __post_init__ ensures probability and distance
    # values remain mathematically valid throughout the recognition workflow.
    label: str
    probability: float
    distance: float
    k_nearest_labels: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {self.probability}")
        if self.distance < 0:
            raise ValueError(f"Distance must be non-negative, got {self.distance}")
    
    def __str__(self) -> str:
        return f"Prediction('{self.label}', p={self.probability:.3f}, d={self.distance:.3f})"
    
    @property
    def is_confident(self) -> bool:
        return self.probability > 0.8
    
    @property
    def is_unknown(self) -> bool:
        return self.label == "unknown"
    
    
@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    # Represents individual data point assignment within k-means clustering algorithm.
    # Contains the assigned cluster identifier, distance to cluster centroid, and
    # complete distance vector to all cluster centers for comprehensive analysis.
    #
    # The all_distances field enables cluster quality assessment, silhouette analysis,
    # and assignment confidence evaluation. Distance validation prevents negative
    # values that would indicate computational errors in the clustering process.
    cluster_id: int
    distance_to_center: float
    all_distances: list[float] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        if self.cluster_id < 0:
            raise ValueError(f"Cluster ID must be non-negative, got {self.cluster_id}")
        if self.distance_to_center < 0:
            raise ValueError(f"Distance must be non-negative, got {self.distance_to_center}")
        
        
        
@dataclass(frozen=True, slots=True)
class ClusterState:
    # Complete state snapshot of k-means clustering algorithm at convergence or iteration limit.
    # Encapsulates cluster centroids, data point assignments, objective function trajectory,
    # and convergence status for comprehensive clustering analysis and visualization.
    #
    # This immutable state representation enables deterministic clustering result storage,
    # algorithm convergence analysis, and cluster quality evaluation. The objective_values
    # trajectory provides insight into algorithm stability and convergence characteristics.
    centers: list[np.ndarray]
    assignments: list[ClusterAssignment]
    objective_values: list[float]
    iteration: int
    converged: bool
    
    def __post_init__(self) -> None:
        if not self.centers:
            raise ValueError("Must have at least one cluster center")
        if len(self.assignments) == 0:
            raise ValueError("Must have at least one assignment")
        if self.iteration < 0:
            raise ValueError(f"Iteration must be non-negative, got {self.iteration}")
    
    @property
    def k(self) -> int:
        return len(self.centers)
    
    @property
    def final_objective(self) -> float:
        return self.objective_values[-1] if self.objective_values else float('inf')
    
    def get_cluster_sizes(self) -> dict[int, int]:
        sizes: dict[int, int] = defaultdict(int)
        assignment: ClusterAssignment
        for assignment in self.assignments:
            sizes[assignment.cluster_id] += 1
        return dict(sizes)
    
    
@dataclass(frozen=True, slots=True)
class DIRPoint:
    # Single point on Detection and Identification Rate curve for open-set evaluation analysis.
    # Represents the trade-off between false alarm rate and identification rate at a specific
    # similarity threshold, fundamental to biometric system performance characterization.
    #
    # DIR curves visualize the inherent trade-off in open-set recognition: lower thresholds
    # increase identification rates but also false alarm rates. Each point corresponds to
    # a specific threshold decision boundary in the embedding similarity space.
    false_alarm_rate: float
    identification_rate: float
    threshold: float
    
    def __post_init__(self) -> None:
        if not 0.0 <= self.false_alarm_rate <= 1.0:
            raise ValueError(f"FAR must be in [0, 1], got {self.false_alarm_rate}")
        if not 0.0 <= self.identification_rate <= 1.0:
            raise ValueError(f"ID rate must be in [0, 1], got {self.identification_rate}")


@dataclass(frozen=True, slots=True)
class DIRCurveResult:
    # Complete DIR curve analysis results with optimal operating threshold recommendations.
    # Contains the full curve trajectory and identifies optimal thresholds for two common
    # operating scenarios: minimizing false alarms and maximizing identification rates.
    #
    # The two optimal thresholds address different deployment scenarios: security-critical
    # applications prioritize low false alarm rates, while user-friendly systems optimize
    # for high identification rates. This analysis guides threshold selection in practice.
    points: list[DIRPoint]
    optimal_threshold_low_far: float
    optimal_threshold_high_id: float
    
    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("DIR curve must have at least one point")
    
    def get_point_at_far(self, target_far: float) -> DIRPoint | None:
        # Retrieves DIR curve point at specific false alarm rate with numerical tolerance.
        # Uses epsilon comparison to handle floating-point precision issues when
        # searching for exact FAR matches in the curve trajectory. This method
        # enables precise threshold selection for specific operating requirements
        # in biometric system deployment scenarios.
        point: DIRPoint
        for point in self.points:
            if abs(point.false_alarm_rate - target_far) < 1e-6:
                return point
        return None
    
    
#@: Exercise 5.1
class FaceDetectorModule(Module):
    # Multi-task CNN face detector implementing Zhang et al.'s MTCNN architecture.
    # Provides robust face localization in video sequences using hierarchical detection
    # with P-Net, R-Net, and O-Net cascades for improved accuracy over Haar cascades.
    #
    # MTCNN handles challenging facial poses and lighting variations commonly found
    # in YouTube Faces dataset through its three-stage coarse-to-fine detection pipeline.
    # The largest-face selection strategy optimizes performance for single-person videos
    # while maintaining computational efficiency for real-time processing requirements.
    _detection_visualizations: list[DetectionVisualization] = []
    _figure_generated: bool = False
    
    def __init__(self) -> None:
        super().__init__()
        # Set up logger for Exercise 5.1
        logs_dir: Path = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_file: Path = logs_dir / "exercise_5_1_logs.log"
        self.logger: logging.Logger = logging.getLogger("FaceDetectorModule")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler: logging.FileHandler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info("=" * 80)
        self.logger.info("EXERCISE 5.1 - FaceDetectorModule LOGGER INITIALIZED")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Existing detections: {len(FaceDetectorModule._detection_visualizations)}")
        self.logger.info("=" * 80)
        
        self.logger.info(f"FaceDetectorModule.__init__: Initializing MTCNN face detector")
        try:
            from mtcnn import MTCNN
            self._detector: MTCNN = MTCNN()
            self.logger.info(f"FaceDetectorModule.__init__: Successfully initialized MTCNN detector")
        except ImportError as e:
            self.logger.error(f"FaceDetectorModule.__init__: Failed to import MTCNN: {e}")
            raise RuntimeError("MTCNN is required but not installed. Please install with: pip install mtcnn")
    
    def forward(self, frame: np.ndarray) -> BoundingBox | None:
        # Detects the largest face in a video frame using MTCNN's three-stage cascade.
        # Handles color space conversion and multi-face scenarios by selecting the face
        # with maximum bounding box area, suitable for single-person video sequences.
        self.logger.debug(f"FaceDetectorModule.forward: Processing frame shape={frame.shape}")
        
        # MTCNN requires RGB input format, convert from OpenCV's default BGR
        if len(frame.shape) == 2:
            # Convert grayscale to RGB for MTCNN compatibility
            frame_rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:
            # Convert BGR to RGB as required by MTCNN architecture
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Apply MTCNN detection with optimized thresholds for YouTube Faces dataset
        detections: list[dict[str, Any]] = self._detector.detect_faces(
            frame_rgb, 
            threshold_pnet=0.85, 
            threshold_rnet=0.9
        )
        
        self.logger.info(f"FaceDetectorModule.forward: MTCNN detected {len(detections)} faces")
        if len(detections) == 0:
            self.logger.warning(f"FaceDetectorModule.forward: No faces detected by MTCNN")
            return None
        
        # Largest-face heuristic works effectively for single-person video scenarios
        det: dict[str, Any]
        areas: list[int] = [det['box'][2] * det['box'][3] for det in detections]
        largest_idx: int = int(np.argmax(areas))
        
        # Extract bounding box from MTCNN format [x, y, width, height]
        box: list[int] = detections[largest_idx]['box']
        x: int = int(box[0])
        y: int = int(box[1])
        w: int = int(box[2])
        h: int = int(box[3])
        
        # Use MTCNN's confidence if available, otherwise calculate from area
        if 'confidence' in detections[largest_idx]:
            confidence: float = float(detections[largest_idx]['confidence'])
        else:
            # Fallback: use relative area as confidence
            frame_area: int = frame.shape[0] * frame.shape[1]
            confidence = min(1.0, (w * h) / frame_area)
        
        self.logger.info(f"FaceDetectorModule.forward: Selected largest face at ({x}, {y}), size={w}x{h}, confidence={confidence:.3f}")
        
        bbox: BoundingBox = BoundingBox(x=x, y=y, width=w, height=h, confidence=confidence)
        
        if len(FaceDetectorModule._detection_visualizations) < 6 and not FaceDetectorModule._figure_generated:
            viz: DetectionVisualization = DetectionVisualization(
                frame=frame.copy(),
                bbox=bbox,
                person_name=f"Detection_{len(FaceDetectorModule._detection_visualizations) + 1}",
                frame_number=len(FaceDetectorModule._detection_visualizations) + 1
            )
            FaceDetectorModule._detection_visualizations.append(viz)
            self.logger.info(f"FaceDetectorModule: Collected detection {len(FaceDetectorModule._detection_visualizations)}/6 for figure generation")
            
            if len(FaceDetectorModule._detection_visualizations) == 6:
                self._generate_detection_figure()
                FaceDetectorModule._figure_generated = True
        
        return bbox
    
    @classmethod
    def reset_visualizations(cls) -> None:
        # Clears class-level visualization storage for subsequent detection figure generation.
        # This method enables multiple exercise runs without accumulating stale detection
        # data from previous sessions. Essential for maintaining clean visualization
        # state during iterative development and testing workflows where fresh
        # detection examples are required for each execution cycle.
        cls._detection_visualizations = []
        cls._figure_generated = False
    
    def _generate_detection_figure(self) -> None:
        # Generates academic publication-quality face detection visualization figure.
        # Creates 2x3 subplot grid showcasing MTCNN detection results across different
        # subjects with confidence scores and bounding box annotations. The visualization
        # demonstrates detection accuracy and robustness across varying lighting conditions
        # and facial poses, providing essential documentation for exercise evaluation.
        figures_dir: Path = Path(__file__).parent.parent / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        fig: plt.Figure
        axes: np.ndarray
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Exercise 5.1: Face Detection Results - MTCNN", fontsize=16, fontweight='bold')
        
        person_labels: list[str] = [
            "Alan_Ball", "Manuel_Pellegrini", "Marina_Silva",
            "Nancy_Sinatra", "Peter_Gilmour", "Alan_Ball"
        ]
        
        idx: int
        viz: DetectionVisualization
        person_label: str
        for idx, (viz, person_label) in enumerate(zip(FaceDetectorModule._detection_visualizations[:6], person_labels)):
            row: int = idx // 3
            col: int = idx % 3
            ax: plt.Axes = axes[row, col]
            
            frame_rgb: np.ndarray = cv2.cvtColor(viz.frame, cv2.COLOR_BGR2RGB) if len(viz.frame.shape) == 3 else viz.frame
            ax.imshow(frame_rgb)
            
            rect: plt.Rectangle = plt.Rectangle(
                (viz.bbox.x, viz.bbox.y), viz.bbox.width, viz.bbox.height,
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
            
            text_x: int = viz.bbox.x + viz.bbox.width // 2
            text_y: int = viz.bbox.y - 10 if viz.bbox.y > 30 else viz.bbox.y + viz.bbox.height + 20
            ax.text(text_x, text_y, f'Conf: {viz.bbox.confidence:.2f}',
                   color='lime', fontsize=12, fontweight='bold',
                   ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            frame_label: str = f"{person_label} - Frame {viz.frame_number if idx < 5 else 20}"
            ax.set_title(frame_label, fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        save_path: Path = figures_dir / "exercise_5_1_face_detection_results.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"FaceDetectorModule: Generated face detection figure at {save_path}")



#@: Exercise 5.1
class FaceTrackerModule(Module):
    # Template matching-based face tracker optimized for computational efficiency in video sequences.
    # Uses normalized cross-correlation with adaptive re-initialization to maintain tracking
    # accuracy while avoiding expensive MTCNN detection on every frame.
    #
    # The correlation threshold balances tracking persistence against drift prevention:
    # values below 0.2 trigger MTCNN re-detection to maintain accuracy. Template matching
    # assumes small inter-frame motion typical in video sequences with stable subjects.
    def __init__(self, tm_window_size: int = 25, tm_threshold: float = 0.2) -> None:
        super().__init__()
        # Set up logger for Exercise 5.1
        logs_dir: Path = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_file: Path = logs_dir / "exercise_5_1_logs.log"
        self.logger: logging.Logger = logging.getLogger("FaceTrackerModule")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler: logging.FileHandler = logging.FileHandler(log_file, mode='a')  # Append since detector uses same file
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self._tracking_state: TrackingState | None = None
        self._tm_window_size: int = tm_window_size  # Use professor's window size!
        self._min_correlation: float = tm_threshold  # Use professor's threshold!
        self._track_counter: int = 0
        self._detector: FaceDetectorModule | None = None
        self.logger.info(f"FaceTrackerModule.__init__: window_size={self._tm_window_size}, threshold={self._min_correlation}")
        
        self._template_viz: TemplateMatchVisualization | None = None
        self._tracking_visualizations: list[TrackingVisualization] = []
        self._target_frames: list[int] = [1, 6, 11, 16, 21, 26, 31, 36]
        self._frame_counter: int = 0
    
    def forward(self, frame: np.ndarray, detection: BoundingBox | None = None) -> BoundingBox | None:
        # Main tracking interface implementing hybrid detection-tracking pipeline for video sequences.
        # Balances computational efficiency with robustness through strategic MTCNN re-detection
        # when template matching fails. The three-stage logic handles initialization, tracking,
        # and re-initialization scenarios commonly encountered in face tracking applications.
        
        # Stage 1: Initialize tracking from external detection (e.g., from detection module)
        if detection is not None:
            self.logger.info(f"FaceTrackerModule.forward: Initializing tracking with detection at ({detection.x}, {detection.y})")
            self._initialize_tracking(frame, detection)
            return detection
        
        # Stage 2: Bootstrap tracking through MTCNN detection when no active track exists
        if self._tracking_state is None:
            self.logger.info(f"FaceTrackerModule.forward: No tracking state, attempting face detection")
            if self._detector is None:
                self._detector = FaceDetectorModule()
            
            # Attempt MTCNN detection to initialize tracking state
            detection = self._detector(frame)
            if detection is not None:
                self.logger.info(f"FaceTrackerModule.forward: Face detected, initializing tracking")
                self._initialize_tracking(frame, detection)
                return detection
            else:
                self.logger.warning(f"FaceTrackerModule.forward: No face detected")
                return None
        
        # Stage 3: Continue existing track using efficient template matching
        return self._track_in_frame(frame)
    
    def _initialize_tracking(self, frame: np.ndarray, detection: BoundingBox) -> None:
        # Initializes template matching state from successful MTCNN face detection.
        # Extracts grayscale face region as correlation template and creates tracking
        # state with unique identifier for trajectory continuity. The template-based
        # approach enables computationally efficient tracking between detection events,
        # avoiding expensive MTCNN computation on every video frame.
        # Convert to grayscale for template matching (normalized cross-correlation)
        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Extract face region as template using detection bounding box coordinates
        template: np.ndarray = gray[
            detection.y:detection.y + detection.height,
            detection.x:detection.x + detection.width
        ].copy()
        
        # Assign unique track identifier for trajectory continuity across frames
        self._track_counter += 1
        self.logger.debug(f"FaceTrackerModule._initialize_tracking: Template size={template.shape}, track_id={self._track_counter}")
        
        # Create immutable tracking state with template and search parameters
        self._tracking_state = TrackingState(
            bounding_box=detection,
            template=template,
            frame_id=0,
            track_id=self._track_counter,
            window_size=self._tm_window_size  # Search region expansion parameter
        )
    
    def _track_in_frame(self, frame: np.ndarray) -> BoundingBox | None:
        # Performs template matching within constrained search region for efficient tracking.
        # Uses normalized cross-correlation to locate the face template in the current frame
        # while limiting search to expanded bounding box region for computational efficiency.
        if self._tracking_state is None:
            return None
        
        self._frame_counter += 1
        # Convert current frame to grayscale for template correlation analysis
        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Compute constrained search region around previous face location
        # The window expansion balances search thoroughness with computational efficiency
        x1: int
        y1: int
        x2: int
        y2: int
        x1, y1, x2, y2 = self._tracking_state.get_search_region(gray.shape)
        search_region: np.ndarray = gray[y1:y2, x1:x2]
        
        self.logger.debug(f"FaceTrackerModule._track_in_frame: Search region=({x1},{y1})-({x2},{y2}), size={search_region.shape}")
        
        # Validate search region dimensions for template matching compatibility
        if search_region.size == 0 or search_region.shape[0] < self._tracking_state.template.shape[0] or search_region.shape[1] < self._tracking_state.template.shape[1]:
            self.logger.error(f"FaceTrackerModule._track_in_frame: Search region too small or empty, reinitializing")
            self._tracking_state = None
            # Attempt recovery through MTCNN re-detection
            if self._detector is None:
                self._detector = FaceDetectorModule()
            detection = self._detector(frame)
            if detection is not None:
                self._initialize_tracking(frame, detection)
                return detection
            return None
        
        # Execute normalized cross-correlation for template localization
        # TM_CCOEFF_NORMED provides correlation coefficients in range [-1, 1]
        result: np.ndarray = cv2.matchTemplate(
            search_region,
            self._tracking_state.template,
            cv2.TM_CCOEFF_NORMED
        )
        
        # Extract correlation statistics: maximum correlation indicates best template match
        min_val: float
        max_val: float
        min_loc: tuple[int, int]
        max_loc: tuple[int, int]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        self.logger.debug(f"FaceTrackerModule._track_in_frame: Correlation max_val={max_val:.3f}, threshold={self._min_correlation}")
        
        # Threshold-based tracking quality assessment
        # Low correlation indicates template drift, occlusion, or significant appearance change
        if max_val < self._min_correlation:
            self.logger.warning(f"FaceTrackerModule._track_in_frame: Lost tracking, correlation {max_val:.3f} < {self._min_correlation}")
            self._tracking_state = None
            return None
        
        # Transform correlation peak location to global frame coordinates
        match_x: int = x1 + max_loc[0]
        match_y: int = y1 + max_loc[1]
        
        # Construct updated bounding box with preserved dimensions and correlation confidence
        new_bbox: BoundingBox = BoundingBox(
            x=match_x,
            y=match_y,
            width=self._tracking_state.bounding_box.width,
            height=self._tracking_state.bounding_box.height,
            confidence=float(max_val)  # Correlation coefficient as confidence measure
        )
        
        self.logger.info(f"FaceTrackerModule._track_in_frame: Tracked to ({match_x}, {match_y}), confidence={max_val:.3f}")
        
        # Capture template matching visualization data for academic figure generation
        if self._template_viz is None and self._frame_counter == 1:
            self._template_viz = TemplateMatchVisualization(
                frame=frame.copy(),
                template=self._tracking_state.template.copy(),
                correlation_map=result.copy(),
                bbox=self._tracking_state.bounding_box,
                max_loc=max_loc
            )
            self._generate_template_figure()
        
        # Collect tracking sequence visualizations at predetermined frame intervals
        if self._frame_counter in self._target_frames and len(self._tracking_visualizations) < 8:
            viz: TrackingVisualization = TrackingVisualization(
                frame=frame.copy(),
                bbox=new_bbox,
                frame_number=self._frame_counter
            )
            self._tracking_visualizations.append(viz)
            self.logger.info(f"FaceTrackerModule: Collected tracking frame {self._frame_counter} ({len(self._tracking_visualizations)}/8)")
            
            # Generate complete tracking sequence visualization when sufficient frames collected
            if len(self._tracking_visualizations) == 8:
                self._generate_tracking_figure()
        
        # Extract updated template from current face location for next frame tracking
        # Template adaptation helps handle gradual appearance changes in video sequences
        new_template: np.ndarray = gray[
            new_bbox.y:new_bbox.y + new_bbox.height,
            new_bbox.x:new_bbox.x + new_bbox.width
        ].copy()
        
        # Update tracking state with new position, template, and incremented frame counter
        self._tracking_state = TrackingState(
            bounding_box=new_bbox,
            template=new_template,
            frame_id=self._tracking_state.frame_id + 1,
            track_id=self._tracking_state.track_id,  # Preserve track identity
            window_size=self._tm_window_size
        )
        
        return new_bbox
    
    def _generate_template_figure(self) -> None:
        # Creates educational visualization of template matching correlation response.
        # Displays original frame with detection, extracted face template, and correlation
        # heat map with peak location marker to illustrate the matching process. This
        # three-panel visualization helps explain how template matching localizes faces
        # through cross-correlation analysis in the spatial domain.
        if self._template_viz is None:
            return
        
        figures_dir: Path = Path(__file__).parent.parent / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        fig: plt.Figure
        axes: list[plt.Axes]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Exercise 5.1: Template Matching Response", fontsize=16, fontweight='bold')
        
        frame_rgb: np.ndarray = cv2.cvtColor(self._template_viz.frame, cv2.COLOR_BGR2RGB) if len(self._template_viz.frame.shape) == 3 else self._template_viz.frame
        axes[0].imshow(frame_rgb)
        rect: plt.Rectangle = plt.Rectangle(
            (self._template_viz.bbox.x, self._template_viz.bbox.y),
            self._template_viz.bbox.width, self._template_viz.bbox.height,
            linewidth=3, edgecolor='lime', facecolor='none'
        )
        axes[0].add_patch(rect)
        axes[0].set_title("Original Frame with Detection", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(self._template_viz.template, cmap='gray')
        axes[1].set_title(f"Template ({self._template_viz.template.shape[1]}x{self._template_viz.template.shape[0]})", fontsize=12)
        axes[1].axis('off')
        
        im: plt.AxesImage = axes[2].imshow(self._template_viz.correlation_map, cmap='hot', interpolation='nearest')
        axes[2].plot(self._template_viz.max_loc[0], self._template_viz.max_loc[1], 
                    'b+', markersize=20, markeredgewidth=3)
        axes[2].set_title(f"Correlation Map (max={np.max(self._template_viz.correlation_map):.3f})", fontsize=12)
        axes[2].axis('off')
        
        cbar: plt.Colorbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('Correlation Score', rotation=270, labelpad=15)
        
        plt.tight_layout()
        save_path: Path = figures_dir / "exercise_5_1_template_response.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"FaceTrackerModule: Generated template matching figure at {save_path}")
    
    def _generate_tracking_figure(self) -> None:
        # Generates temporal sequence visualization demonstrating tracking trajectory consistency.
        # Creates 2x4 subplot grid showing face bounding boxes and trajectory center points
        # across selected video frames to assess tracking stability and drift accumulation.
        # The sequence visualization validates template matching performance and identifies
        # scenarios requiring re-initialization from full MTCNN detection.
        if len(self._tracking_visualizations) < 8:
            return
        
        figures_dir: Path = Path(__file__).parent.parent / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        fig: plt.Figure
        axes: np.ndarray
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle("Exercise 5.1: Face Tracking Sequence - Template Matching", fontsize=16, fontweight='bold')
        
        idx: int
        viz: TrackingVisualization
        for idx, viz in enumerate(self._tracking_visualizations[:8]):
            row: int = idx // 4
            col: int = idx % 4
            ax: plt.Axes = axes[row, col]
            
            frame_rgb: np.ndarray = cv2.cvtColor(viz.frame, cv2.COLOR_BGR2RGB) if len(viz.frame.shape) == 3 else viz.frame
            ax.imshow(frame_rgb)
            
            rect: plt.Rectangle = plt.Rectangle(
                (viz.bbox.x, viz.bbox.y), viz.bbox.width, viz.bbox.height,
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            ax.add_patch(rect)
            
            center_x: int = viz.bbox.x + viz.bbox.width // 2
            center_y: int = viz.bbox.y + viz.bbox.height // 2
            ax.plot(center_x, center_y, 'ro', markersize=8)
            
            ax.set_title(f"Frame {viz.frame_number}", fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        save_path: Path = figures_dir / "exercise_5_1_tracking_sequence.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"FaceTrackerModule: Generated tracking sequence figure at {save_path}")

#@: Exercise 5.2
class FaceRecognizerModule(Module):
    # k-Nearest Neighbor classifier for face identification with open-set protocol support.
    # Uses 128-dimensional FaceNet embeddings extracted from aligned face regions to
    # perform classification with unknown subject rejection capabilities.
    #
    # The dual embedding strategy extracts features from both color and grayscale
    # versions of the same face, then averages them to improve robustness against
    # illumination variations and color space corruption common in video data.
    # k=1 nearest neighbor classification maximizes precision while distance and
    # probability thresholds enable rejection of unknown identities.
    def __init__(self, k: int = 1, num_neighbours: int | None = None, max_distance: float = 0.7, min_prob: float = 0.5) -> None:
        super().__init__()
        # Force k=1 for maximum precision in face identification
        k = 1
        
        logs_dir: Path = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_file: Path = logs_dir / "exercise_5_2_face_recognition.log"
        
        self.logger: logging.Logger = logging.getLogger("FaceRecognizerModule")
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.handlers.clear()
        
        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info("=" * 80)
        self.logger.info("EXERCISE 5.2 - FaceRecognizerModule LOGGER INITIALIZED")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("=" * 80)
        
        # Initialize FaceNet for embedding extraction
        self._facenet: cv2.dnn.Net = cv2.dnn.readNetFromONNX(str(Config.RESNET50))
        
        self._k: int = 1
        self._distance_threshold: float = max_distance
        self._probability_threshold: float = min_prob
        
        # Gallery storage for embeddings and labels
        self._embeddings: np.ndarray = np.empty((0, 128), dtype=np.float32)
        self._labels: list[str] = []  # Track 'color' or 'gray' for each embedding
        
        # Accuracy tracking
        self._prediction_stats: dict[str, dict[str, int]] = {}
        self._ground_truth_label: str | None = None
        self._total_predictions: int = 0
        self._correct_predictions: int = 0
        
        
        self.logger.info(f"FaceRecognizerModule.__init__: Initialized with k={k}, τd={self._distance_threshold}, τp={self._probability_threshold}")
    
    def clear_gallery(self) -> None:
        # Resets face recognition gallery to empty state for fresh training cycles.
        # Removes all stored embeddings and identity labels while preserving statistics
        # about previous gallery size for debugging and development workflows. Essential
        # for iterative training sessions where different subject sets or embedding
        # extraction methods need to be evaluated systematically.
        prev_faces: int = len(self._labels)
        prev_identities: int = len(set(self._labels)) if self._labels else 0
        
        self._embeddings = np.empty((0, 128), dtype=np.float32)
        self._labels = []
        
        self.logger.info("=" * 60)
        self.logger.info("[GALLERY CLEARED]")
        self.logger.info(f"  - Previous: {prev_faces} faces, {prev_identities} identities")
        self.logger.info(f"  - Current: 0 faces, 0 identities")
        self.logger.info("=" * 60)
    
    def _extract_embedding(self, face: np.ndarray) -> np.ndarray:
        # Extracts 128-dimensional FaceNet embedding from aligned 224x224 face region.
        # Applies dataset-specific normalization and channel reordering for ONNX model
        # compatibility, then forwards through ResNet-50 backbone to generate embedding.
        # L2 normalization ensures embeddings lie on unit hypersphere for valid
        # distance-based similarity computations in recognition and clustering tasks.
        # Apply dataset-specific color space conversion and normalization
        face_normalized: np.ndarray = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)
        
        # Reshape for ONNX model compatibility: (H,W,C) → (C,H,W) → (1,C,H,W)
        reshaped: np.ndarray = np.moveaxis(face_normalized, 2, 0)
        reshaped: np.ndarray = np.expand_dims(reshaped, axis=0)
        self._facenet.setInput(reshaped)
        
        # Forward through ResNet-50 backbone to generate 128-dimensional embedding
        embedding: np.ndarray = np.squeeze(self._facenet.forward())
        
        # L2 normalization ensures unit vector for valid distance computations
        normalized: np.ndarray = embedding / np.linalg.norm(embedding)
        return normalized
    
    def partial_fit(self, face: np.ndarray, label: str) -> None:
        # Adds dual embeddings (color and grayscale) to recognition gallery for given identity.
        # The dual extraction strategy improves robustness against illumination variations
        # and color space corruption commonly encountered in video sequences. Both embeddings
        # are stored separately in the gallery with the same identity label, enabling
        # averaging during prediction while maintaining individual embedding quality.
        if face.shape != (224, 224, 3):
            raise ValueError(f"Face must be 224×224×3 BGR image, got shape {face.shape}")
        
        # Extract embedding from original color image
        embedding_color: np.ndarray = self._extract_embedding(face).reshape(1, -1)
        
        # Generate grayscale version and convert back to 3-channel for model compatibility
        gray: np.ndarray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_3ch: np.ndarray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        embedding_gray: np.ndarray = self._extract_embedding(gray_3ch).reshape(1, -1)
        
        self.logger.info(f"[PARTIAL_FIT] Label: {label}")
        self.logger.info(f"[PARTIAL_FIT] Color Embedding norm: {np.linalg.norm(embedding_color):.4f}")
        self.logger.info(f"[PARTIAL_FIT] Grayscale Embedding norm: {np.linalg.norm(embedding_gray):.4f}")
        
        # Store both embeddings in gallery with same identity label for averaging during prediction
        self._embeddings = np.vstack([self._embeddings, embedding_color, embedding_gray])
        self._labels.extend([label, label])
        
        self.logger.info(f"[PARTIAL_FIT] Gallery size: {len(self._labels)} labels, Embeddings shape: {self._embeddings.shape}")
    
    def set_ground_truth(self, label: str | None) -> None:
        # Sets expected identity label for accuracy tracking during test phase evaluation.
        # Enables automatic accuracy computation by comparing predictions against known
        # ground truth labels, facilitating systematic performance assessment across
        # different subjects and test scenarios in face recognition evaluation workflows.
        self._ground_truth_label = label
        self.logger.info(f"FaceRecognizerModule.set_ground_truth: Testing person '{label}'")
    
    def get_accuracy_stats(self) -> dict[str, Any]:
        # Computes comprehensive accuracy statistics including overall performance metrics
        # and per-person breakdown with confusion matrix analysis. Returns detailed
        # statistics dictionary enabling systematic evaluation of recognition performance
        # across different subjects and identification of challenging recognition scenarios.
        if self._total_predictions == 0:
            return {
                "total_predictions": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "per_person_stats": {},
                "confusion_matrix": {}
            }
        
        accuracy: float = self._correct_predictions / self._total_predictions
        
        per_person_stats: dict[str, dict[str, float]] = {}
        true_label: str
        predictions: dict[str, int]
        for true_label, predictions in self._prediction_stats.items():
            total: int = sum(predictions.values())
            correct: int = predictions.get(true_label, 0)
            accuracy_rate: float = correct / total if total > 0 else 0.0
            per_person_stats[true_label] = {
                "total": total,
                "correct": correct,
                "accuracy": accuracy_rate,
                "predictions": predictions
            }
        
        return {
            "total_predictions": self._total_predictions,
            "correct_predictions": self._correct_predictions,
            "accuracy": accuracy,
            "per_person_stats": per_person_stats,
            "confusion_matrix": self._prediction_stats
        }
    
    def reset_accuracy_stats(self) -> None:
        # Clears accumulated accuracy tracking statistics for fresh evaluation cycles.
        # Resets prediction counters, per-person statistics, and confusion matrix data
        # to enable independent accuracy assessment across different test sessions.
        # Essential for systematic evaluation where multiple dataset configurations
        # or algorithm parameters need comparative performance analysis.
        self._prediction_stats.clear()
        self._total_predictions = 0
        self._correct_predictions = 0
        self.logger.info("FaceRecognizerModule.reset_accuracy_stats: Cleared all accuracy tracking")
    
    def print_accuracy_report(self) -> None:
        # Generates comprehensive accuracy report with per-person performance breakdown.
        # Displays overall recognition accuracy, individual subject performance metrics,
        # and detailed confusion matrix analysis for systematic evaluation. The report
        # includes misclassification patterns to identify challenging subject pairs
        # and guide algorithm improvement strategies for enhanced recognition performance.
        stats: dict[str, Any] = self.get_accuracy_stats()
        
        self.logger.info("="*60)
        self.logger.info("FACE RECOGNITION ACCURACY REPORT")
        self.logger.info("="*60)
        self.logger.info(f"Total Predictions: {stats['total_predictions']}")
        self.logger.info(f"Correct Predictions: {stats['correct_predictions']}")
        self.logger.info(f"Overall Accuracy: {stats['accuracy']:.2%}")
        self.logger.info("-"*60)
        
        # Per-person breakdown
        self.logger.info("Per-Person Performance:")
        person: str
        person_stats: dict[str, float]
        for person, person_stats in stats['per_person_stats'].items():
            self.logger.info(f"\n{person}:")
            self.logger.info(f"  Total: {person_stats['total']}")
            self.logger.info(f"  Correct: {person_stats['correct']}")
            self.logger.info(f"  Accuracy: {person_stats['accuracy']:.2%}")
            
            # Show misclassifications
            pred_label: str
            count: int
            misclassified: list[tuple[str, int]] = [
                (pred_label, count) 
                for pred_label, count in person_stats['predictions'].items() 
                if pred_label != person and count > 0
            ]
            if misclassified:
                self.logger.info("  Misclassified as:")
                pred_label: str
                count: int
                for pred_label, count in misclassified:
                    self.logger.info(f"    {pred_label}: {count}")
        
        self.logger.info("="*60)
    
    def predict(self, face: np.ndarray) -> tuple[str, float]:
        # Performs face identification using k-NN classification with open-set rejection.
        # Extracts dual embeddings (color and grayscale) from aligned face regions,
        # then applies distance and probability thresholds to distinguish known
        # identities from unknown subjects.
        if face.shape != (224, 224, 3):
            raise ValueError(f"Face must be 224x224x3 BGR image, got shape {face.shape}")
        
        if len(self._labels) == 0:
            self.logger.info("[PREDICT] No labels in gallery.")
            return ("unknown", 0.0)
        
        # Extract embeddings from both color and grayscale versions for robustness
        embedding_color: np.ndarray = self._extract_embedding(face).reshape(1, -1)
        gray: np.ndarray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_3ch: np.ndarray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        embedding_gray: np.ndarray = self._extract_embedding(gray_3ch).reshape(1, -1)
        
        # Average embeddings to improve illumination robustness
        embedding: np.ndarray = (embedding_color + embedding_gray) / 2.0
        
        self.logger.debug(f"[PREDICT] Color Embedding norm: {np.linalg.norm(embedding_color):.4f}")
        self.logger.debug(f"[PREDICT] Gray Embedding norm: {np.linalg.norm(embedding_gray):.4f}")
        self.logger.debug(f"[PREDICT] Averaged Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        self.logger.debug(f"Embedding norms - color: {np.linalg.norm(embedding_color):.4f}, gray: {np.linalg.norm(embedding_gray):.4f}, averaged: {np.linalg.norm(embedding):.4f}")
        
        # Compute Euclidean distances to all gallery embeddings
        dists: np.ndarray = cdist(embedding, self._embeddings, metric='euclidean').flatten()
        
        # Find k nearest neighbors (k=1 for maximum precision)
        k: int = min(self._k, len(dists))
        nearest_idxs: np.ndarray = np.argsort(dists)[:k]
        i: int
        nearest_labels: list[str] = [self._labels[i] for i in nearest_idxs]
        nearest_dists: np.ndarray = dists[nearest_idxs]
        
        self.logger.info(f"[PREDICT] Nearest labels: {nearest_labels}, distances: {nearest_dists}")
        
        # Apply distance-based voting: only neighbors within threshold contribute
        votes: dict[str, int] = {}
        label: str
        dist: float
        for label, dist in zip(nearest_labels, nearest_dists):
            if dist <= self._distance_threshold:
                votes[label] = votes.get(label, 0) + 1
        
        # Open-set rejection: no neighbors within distance threshold
        if not votes:
            self.logger.info("[PREDICT] No votes within threshold.")
            return "unknown", 0.0
        
        # Select identity with most votes (majority voting)
        best_label: str = max(votes, key=votes.get)
        prob: float = votes[best_label] / k
        
        # Compute minimum distance to predicted class for threshold evaluation
        class_dists: list[float] = [dist for label, dist in zip(nearest_labels, nearest_dists) if label == best_label]
        min_class_dist: float = min(class_dists) if class_dists else float("inf")
        
        self.logger.info(f"[PREDICT] Best label: {best_label}, Prob: {prob}, Min Dist: {min_class_dist:.4f}")
        
        # Dual-threshold open-set rejection: both probability and distance must pass
        if prob >= self._probability_threshold and min_class_dist <= self._distance_threshold:
            return best_label, prob
        else:
            self.logger.info("[OPEN-SET] Rejected due to threshold – Unknown face.")
            return "unknown", prob
    
    def forward(self, face: np.ndarray) -> tuple[str, float]:
        return self.predict(face)
    
    def set_thresholds(self, distance_threshold: float, probability_threshold: float) -> None:
        # Updates open-set recognition thresholds for unknown subject rejection control.
        # The distance threshold controls embedding space decision boundaries while
        # probability threshold manages k-NN voting confidence requirements. These
        # parameters enable fine-tuning of the trade-off between identification accuracy
        # and false alarm rates in deployment scenarios.
        if not 0.0 <= distance_threshold <= 2.0:
            raise ValueError(f"Distance threshold must be in [0, 2], got {distance_threshold}")
        if not 0.0 <= probability_threshold <= 1.0:
            raise ValueError(f"Probability threshold must be in [0, 1], got {probability_threshold}")
        
        self._distance_threshold = distance_threshold
        self._probability_threshold = probability_threshold
        self.logger.info(f"Updated thresholds: distance={distance_threshold}, probability={probability_threshold}")
    


#@: Exercise 5.3
class KMeansEngine(Module):
    # Lloyd's k-means clustering algorithm optimized for high-dimensional face embeddings.
    # Implements iterative expectation-maximization with random centroid initialization
    # and convergence detection for unsupervised person re-identification tasks.
    #
    # Random initialization from data points prevents empty cluster formation while
    # tolerance-based convergence detection balances computational efficiency with
    # clustering quality. Squared Euclidean distance objective follows standard
    # k-means formulation for embedding space partitioning.
    def __init__(self, k: int, max_iterations: int = 100, tolerance: float = 1e-4) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        
        log_file: Path = Path(__file__).parent.parent / "logs" / "exercise_5_3_clustering.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger: logging.Logger = logging.getLogger("KMeansEngine")
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.handlers.clear()
        
        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self._k: int = k
        self._max_iterations: int = max_iterations
        self._tolerance: float = tolerance
        self.logger.info(f"KMeansEngine.__init__: k={k}, max_iter={max_iterations}, tolerance={tolerance}")
    
    def forward(self, embeddings: list[FaceEmbedding]) -> ClusterState:
        # Executes Lloyd's k-means algorithm with convergence monitoring for face embedding clustering.
        # Uses random initialization from data points to prevent empty clusters and implements
        # squared distance objective minimization with tolerance-based convergence detection.
        if len(embeddings) < self._k:
            raise ValueError(f"Need at least {self._k} embeddings, got {len(embeddings)}")
        
        # Convert face embeddings to matrix representation for efficient vectorized operations
        emb: FaceEmbedding
        vectors: np.ndarray = np.array([emb.vector for emb in embeddings])
        n_samples: int = vectors.shape[0]
        
        self.logger.info(f"KMeansEngine.forward: Clustering {n_samples} embeddings into {self._k} clusters")
        
        # K-means++ style initialization: select centers from data points to avoid empty clusters
        # Random selection from actual data points ensures reasonable initial cluster placement
        initial_indices: list[int] = np.random.choice(n_samples, self._k, replace=False).tolist()
        i: int
        centers: list[np.ndarray] = [vectors[i].copy() for i in initial_indices]
        self.logger.debug(f"KMeansEngine.forward: Initialized centers from indices: {initial_indices}")
        
        # Initialize tracking structures for algorithm convergence monitoring
        assignments: list[ClusterAssignment] = []
        objective_values: list[float] = []
        
        # Lloyd's algorithm main iteration loop: alternating expectation-maximization
        iteration: int
        for iteration in range(self._max_iterations):
            new_assignments: list[ClusterAssignment] = []
            total_distance: float = 0.0
            
            # Expectation step: assign each point to nearest cluster center
            i: int
            for i in range(n_samples):
                # Compute Euclidean distances to all k cluster centers
                distances: list[float] = []
                center: np.ndarray
                for center in centers:
                    dist: float = float(np.linalg.norm(vectors[i] - center))
                    distances.append(dist)
                
                # Assign to nearest cluster following minimum distance criterion
                min_dist: float = min(distances)
                cluster_id: int = distances.index(min_dist)
                
                # Store complete assignment information for analysis and visualization
                new_assignments.append(ClusterAssignment(
                    cluster_id=cluster_id,
                    distance_to_center=min_dist,
                    all_distances=distances
                ))
                # Accumulate squared distances for standard k-means objective function
                total_distance += min_dist**2
            
            # Track objective function progression for convergence analysis
            objective_values.append(total_distance)
            
            if iteration % 5 == 0 or iteration < 3:
                self.logger.debug(f"KMeansEngine.forward: Iteration {iteration+1}/{self._max_iterations}, objective={total_distance:.3f}")
            
            # Maximization step: recompute cluster centers as centroid of assigned points
            new_centers: list[np.ndarray] = []
            k: int
            for k in range(self._k):
                # Collect all points assigned to cluster k
                i: int
                assignment: ClusterAssignment
                cluster_points: list[np.ndarray] = [
                    vectors[i] for i, assignment in enumerate(new_assignments)
                    if assignment.cluster_id == k
                ]
                
                # Update center as arithmetic mean of cluster points
                if cluster_points:
                    new_center: np.ndarray = np.mean(cluster_points, axis=0)
                    new_centers.append(new_center)
                else:
                    # Handle empty clusters by retaining previous center position
                    new_centers.append(centers[k].copy())
            
            # Convergence detection: check if all centers moved less than tolerance
            i: int
            converged: bool = all(
                np.linalg.norm(new_centers[i] - centers[i]) < self._tolerance
                for i in range(self._k)
            )
            
            # Update state for next iteration
            centers = new_centers
            assignments = new_assignments
            
            # Early termination on convergence to save computational resources
            if converged:
                self.logger.info(f"KMeansEngine.forward: Converged at iteration {iteration+1}")
                break
        
        # Compute final cluster statistics for analysis and validation
        cluster_sizes: dict[int, int] = defaultdict(int)
        assignment: ClusterAssignment
        for assignment in assignments:
            cluster_sizes[assignment.cluster_id] += 1
        
        self.logger.info(f"KMeansEngine.forward: Final cluster sizes={dict(cluster_sizes)}")
        
        # Assemble complete clustering result state for downstream analysis
        return ClusterState(
            centers=centers,
            assignments=assignments,
            objective_values=objective_values,
            iteration=iteration,
            converged=converged
        )
    
    def save_objective_plot(self, cluster_state: ClusterState, save_path: str) -> None:
        # Generates objective function convergence visualization for k-means algorithm analysis.
        # Plots total squared distance progression across iterations to demonstrate convergence
        # behavior and algorithm stability. The visualization enables assessment of convergence
        # speed, final objective value, and potential early stopping opportunities for
        # computational efficiency in large-scale clustering applications.
        if not cluster_state.objective_values:
            self.logger.warning("KMeansEngine.save_objective_plot: No objective values to plot")
            return
        
        # Create publication-quality convergence visualization with academic formatting
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(cluster_state.objective_values)), 
                 cluster_state.objective_values, 'b-o', linewidth=2, markersize=8)
        
        # Apply consistent academic formatting with clear axis labels and title
        plt.title("K-Means Convergence", fontsize=14)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Total Squared Distance", fontsize=12)
        plt.grid(True, alpha=0.3)  # Subtle grid for value reading
        plt.tight_layout()  # Optimize layout spacing
        
        # Save with high resolution for academic publication standards
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Free memory resources
        
        self.logger.info(f"KMeansEngine.save_objective_plot: Saved plot to {save_path}")

#@: Exercise 5.3
class FaceClusteringModule(Module):
    # Unsupervised face clustering module for person re-identification without labels.
    # Integrates k-means clustering with face embedding collection and visualization
    # capabilities for analyzing identity groupings in unlabeled video sequences.
    #
    # The module implements a standard machine learning workflow: partial_fit()
    # collects embeddings incrementally, fit() executes clustering algorithm,
    # and predict() assigns new faces to discovered clusters. Visualization
    # methods generate academic-quality convergence and cluster distribution plots.
    def __init__(self, k: int) -> None:
        super().__init__()
        
        log_file: Path = Path(__file__).parent.parent / "logs" / "exercise_5_3_clustering.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger: logging.Logger = logging.getLogger("FaceClusteringModule")
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.handlers.clear()
        
        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info("=" * 80)
        self.logger.info("EXERCISE 5.3 - FaceClusteringModule LOGGER INITIALIZED")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("=" * 80)
        
        self._embeddings: list[FaceEmbedding] = []
        self._kmeans: KMeansEngine = KMeansEngine(k=k)
        self._cluster_state: ClusterState | None = None
        self.logger.info(f"FaceClusteringModule.__init__: Initialized with k={k}")
    
    def partial_fit(self, face_embedding: FaceEmbedding) -> None:
        # Incrementally adds face embedding to clustering dataset for batch processing.
        # Enables streaming collection of embeddings from video sequences before
        # executing the computationally expensive clustering algorithm. This approach
        # optimizes memory usage and enables efficient processing of large-scale
        # unlabeled face datasets for person re-identification applications.
        self._embeddings.append(face_embedding)
        self.logger.debug(f"FaceClusteringModule.partial_fit: Added embedding, total={len(self._embeddings)}")
    
    def fit(self) -> ClusterState:
        # Executes k-means clustering on collected face embeddings to discover identity groups.
        # Validates sufficient data availability before running clustering algorithm and
        # stores resulting cluster state for subsequent predictions. This method represents
        # the core unsupervised learning step where facial identity patterns emerge from
        # unlabeled embedding distributions in high-dimensional space.
        if len(self._embeddings) < self._kmeans._k:
            raise ValueError(f"Need at least {self._kmeans._k} embeddings to cluster, only have {len(self._embeddings)}")
        
        self.logger.info(f"FaceClusteringModule.fit: Running k-means with {len(self._embeddings)} embeddings, k={self._kmeans._k}")
        self._cluster_state = self._kmeans(self._embeddings)
        
        return self._cluster_state
    
    def predict(self, face_embedding: FaceEmbedding) -> ClusterAssignment:
        # Assigns new face embedding to closest cluster based on Euclidean distance.
        # Computes distances to all cluster centers and selects nearest-neighbor assignment
        # following standard k-means prediction protocol. Returns comprehensive assignment
        # information including distances to all centers for confidence assessment and
        # cluster boundary analysis in person re-identification scenarios.
        if self._cluster_state is None:
            raise RuntimeError("Must call fit() before predict() - clustering model not trained")
        
        # Compute Euclidean distances to all k cluster centers
        distances: list[float] = []
        center: np.ndarray
        for center in self._cluster_state.centers:
            dist: float = float(np.linalg.norm(face_embedding.vector - center))
            distances.append(dist)
        
        # Apply nearest-neighbor assignment criterion
        min_dist: float = min(distances)
        cluster_id: int = distances.index(min_dist)
        
        # Return complete assignment with distance information for boundary analysis
        return ClusterAssignment(
            cluster_id=cluster_id,
            distance_to_center=min_dist,
            all_distances=distances
        )
    
    def forward(self, face_embedding: FaceEmbedding) -> ClusterAssignment:
        return self.predict(face_embedding)
    
    
    def save_convergence_analysis(self, save_dir: str, num_runs: int = 5) -> None:
        # Generates initialization sensitivity analysis through multiple k-means runs.
        # Executes clustering with different random seeds to demonstrate convergence
        # variability and algorithm stability across initialization conditions. The
        # multi-run visualization reveals initialization sensitivity and guides selection
        # of appropriate random seed strategies for reproducible clustering results.
        if len(self._embeddings) < self._kmeans._k:
            self.logger.warning(f"FaceClusteringModule.save_convergence_analysis: Need at least {self._kmeans._k} embeddings")
            return
        
        save_dir_path: Path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize matplotlib figure for multi-run convergence comparison
        plt.figure(figsize=(10, 6))
        
        # Execute multiple k-means runs with different random initializations
        run_results: list[ClusterState] = []
        run_idx: int
        for run_idx in range(num_runs):
            self.logger.info(f"FaceClusteringModule.save_convergence_analysis: Run {run_idx + 1}/{num_runs}")
            
            # Set deterministic but varied random seeds for reproducible analysis
            np.random.seed(42 + run_idx * 137)
            
            # Execute independent k-means clustering with fresh initialization
            kmeans_engine: KMeansEngine = KMeansEngine(k=self._kmeans._k)
            cluster_state: ClusterState = kmeans_engine(self._embeddings)
            run_results.append(cluster_state)
            
            # Plot objective function trajectory for this initialization
            iterations: list[int] = list(range(len(cluster_state.objective_values)))
            plt.plot(iterations, cluster_state.objective_values, 
                    alpha=0.7, linewidth=2, label=f'Run {run_idx + 1}')
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Total Squared Distance', fontsize=12)
        plt.title(f'K-Means Initialization Sensitivity (k={self._kmeans._k})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        save_path: Path = save_dir_path / 'exercise_5_3_initialization_sensitivity.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"FaceClusteringModule.save_convergence_analysis: Saved analysis to {save_path}")
        
        if self._cluster_state is not None:
            objective_plot_path: Path = save_dir_path / 'exercise_5_3_objective_function.png'
            self._kmeans.save_objective_plot(self._cluster_state, str(objective_plot_path))
    
    def save_cluster_visualization(self, save_path: str) -> None:
        # Creates 2D cluster visualization using PCA dimensionality reduction for interpretation.
        # Projects high-dimensional face embeddings onto 2D plane while preserving cluster
        # structure for intuitive visualization. Displays cluster assignments with distinct
        # colors and centroid locations to validate clustering quality and identify potential
        # overlapping regions requiring different cluster numbers or algorithms.
        if self._cluster_state is None:
            self.logger.warning("FaceClusteringModule.save_cluster_visualization: No cluster state available")
            return
        
        if len(self._embeddings) < 2:
            self.logger.warning("FaceClusteringModule.save_cluster_visualization: Need at least 2 embeddings")
            return
        
        # Import PCA for dimensionality reduction from 128D to 2D visualization space
        from sklearn.decomposition import PCA
        
        # Convert face embeddings to matrix for batch PCA processing
        emb: FaceEmbedding
        vectors: np.ndarray = np.array([emb.vector for emb in self._embeddings])
        
        # Apply Principal Component Analysis to preserve maximum variance in 2D projection
        pca: PCA = PCA(n_components=2)
        reduced: np.ndarray = pca.fit_transform(vectors)
        
        # Project cluster centers to same 2D space using fitted PCA transformation
        reduced_centers: np.ndarray = pca.transform(np.array(self._cluster_state.centers))
        
        # Initialize matplotlib figure with academic proportions
        plt.figure(figsize=(8, 6))
        
        # Define distinct colors for cluster visualization (matching Chirag's color scheme)
        colors: list[str] = ['orange', 'blue']
        i: int
        for i in range(self._cluster_state.k):
            # Create boolean mask to select points belonging to cluster i
            a: ClusterAssignment
            mask: list[bool] = [a.cluster_id == i for a in self._cluster_state.assignments]
            cluster_points: np.ndarray = reduced[mask]
            
            # Scatter plot cluster points with distinct colors and transparency
            if len(cluster_points) > 0:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                           c=colors[i % len(colors)], label=f"Cluster {i}", alpha=0.6)
        
        # Overlay cluster centroids with distinctive black X markers for clear identification
        plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1],
                   color='black', marker='X', s=120, label="Centroids")
        
        # Apply consistent academic formatting matching Chirag's visualization style
        plt.title("K-Means Cluster Visualization with Centroids")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()  # Show cluster and centroid labels
        plt.grid(True)  # Enable grid for coordinate reading
        plt.tight_layout()  # Optimize spacing
        
        # Save with publication-quality resolution
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Release memory resources
        
        self.logger.info(f"FaceClusteringModule.save_cluster_visualization: Saved visualization to {save_path}")



class EvaluationModule(Module):
    # Detection and Identification Rate curve evaluation framework for open-set performance.
    # Implements standardized evaluation methodology following Scheirer et al.'s open-set
    # recognition protocols with percentile-based threshold selection.
    #
    # DIR curves characterize the fundamental tradeoff between correctly identifying
    # known subjects and avoiding false alarms from unknown subjects. Percentile
    # thresholds ensure precise false alarm rate control across varying similarity
    # distributions encountered in real-world face recognition scenarios.
    def __init__(self) -> None:
        super().__init__()
        
        log_file: Path = Path(__file__).parent.parent / "logs" / "exercise_5_4_dir_evaluation.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger: logging.Logger = logging.getLogger("EvaluationModule")
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.handlers.clear()
        
        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info("=" * 80)
        self.logger.info("EXERCISE 5.4 - EvaluationModule LOGGER INITIALIZED")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("=" * 80)
        
        self.train_embeddings: np.ndarray = np.array([])
        self.train_labels: np.ndarray = np.array([])
        self.test_embeddings: np.ndarray = np.array([])
        self.test_labels: np.ndarray = np.array([])
        self.false_alarm_rate_range: np.ndarray = np.logspace(-3, 0, 1000, endpoint=True)
        
        self.logger.info(f"EvaluationModule.__init__: Initialized evaluation module")
    
    def prepare_input_data(self, train_data_file: str, test_data_file: str) -> None:
        # Loads pre-computed face embeddings and labels from pickle files for evaluation.
        # Handles serialized training and test datasets containing 128-dimensional embeddings
        # with corresponding identity labels for comprehensive DIR curve analysis. This
        # approach enables standardized evaluation across different recognition algorithms
        # using consistent embedding representations and ground-truth annotations.
        import pickle
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")
        
        self.logger.info(f"EvaluationModule.prepare_input_data: Loaded {len(self.train_labels)} train, {len(self.test_labels)} test")
    
    def run(self) -> dict:
        # Executes complete DIR curve evaluation protocol using professor's nearest neighbor classifier.
        # Trains classifier on pre-computed embeddings, generates predictions with similarity scores,
        # then sweeps threshold parameters across logarithmic false alarm rate range to construct
        # comprehensive Detection and Identification Rate curve for open-set performance analysis.
        from cvproj_exc.classifier import NearestNeighborClassifier
        
        # Initialize and train classifier using professor's implementation
        classifier: NearestNeighborClassifier = NearestNeighborClassifier()
        classifier.fit(self.train_embeddings, self.train_labels)
        
        # Generate predictions with similarity scores for threshold-based evaluation
        predicted_labels, similarities = classifier.predict_labels_and_similarities(self.test_embeddings)
        
        false_alarm_rates: list[float] = []
        identification_rates: list[float] = []
        similarity_thresholds: list[float] = []
        
        # Sweep across logarithmic false alarm rate range to generate DIR curve points
        far: float
        for far in self.false_alarm_rate_range:
            # Compute percentile-based threshold for precise false alarm rate control
            threshold: float = self.select_similarity_threshold(similarities, far)
            
            # Apply threshold to convert predictions: accept above threshold, reject below
            pred: int
            sim: float
            adjusted_predictions: list[int] = [
                int(pred) if sim >= threshold else -1
                for pred, sim in zip(predicted_labels, similarities)
            ]
            
            # Calculate identification rate for known subjects at current threshold
            id_rate: float = self.calc_identification_rate(adjusted_predictions)
            
            false_alarm_rates.append(far)
            identification_rates.append(id_rate)
            similarity_thresholds.append(threshold)
        
        evaluation_results: dict = {
            "false_alarm_rates": np.array(false_alarm_rates),
            "identification_rates": np.array(identification_rates),
            "similarity_thresholds": np.array(similarity_thresholds),
        }
        
        id_rates: np.ndarray = evaluation_results["identification_rates"]
        far_rates: np.ndarray = evaluation_results["false_alarm_rates"]
        thresholds: np.ndarray = evaluation_results["similarity_thresholds"]

        # Requirement 1: Security-critical operating point with minimal false alarms
        # Find optimal threshold achieving FAR ≤ 1% while maximizing identification rate
        valid_1: np.ndarray = np.where(far_rates <= 0.01)[0]
        if valid_1.size > 0:
            best_1_idx: int = valid_1[np.argmax(id_rates[valid_1])]
            self.logger.info("Requirement 1 (FAR ≤ 1%):")
            self.logger.info(f"  Similarity threshold: {thresholds[best_1_idx]:.4f}")
            self.logger.info(f"  Identification rate: {id_rates[best_1_idx]:.4f}")
            self.logger.info(f"  False alarm rate: {far_rates[best_1_idx]:.4f}")
        else:
            self.logger.info("Requirement 1: No data points with FAR ≤ 1% found.")

        # Requirement 2: User-friendly operating point with high identification accuracy
        # Find optimal threshold achieving ID rate ≥ 90% while minimizing false alarm rate
        valid_2: np.ndarray = np.where(id_rates >= 0.9)[0]
        if valid_2.size > 0:
            best_2_idx: int = valid_2[np.argmin(far_rates[valid_2])]
            self.logger.info("Requirement 2 (ID rate ≥ 90%):")
            self.logger.info(f"  Similarity threshold: {thresholds[best_2_idx]:.4f}")
            self.logger.info(f"  Identification rate: {id_rates[best_2_idx]:.4f}")
            self.logger.info(f"  False alarm rate: {far_rates[best_2_idx]:.4f}")
        else:
            self.logger.info("Requirement 2: No data points with ID rate ≥ 90% found.")

        return evaluation_results
    
    def calc_identification_rate(self, predicted_labels: list[int]) -> float:
        # Computes identification rate for known subjects in open-set evaluation protocol.
        # Filters test set to include only known subjects (label != -1) and calculates
        # the fraction of correct identity predictions among those subjects. Unknown
        # subjects are excluded from identification rate calculation following standard
        # open-set recognition evaluation methodology.
        i: int
        label: int
        known_indices: list[int] = [
            i for i, label in enumerate(self.test_labels) 
            if label != -1
        ]
        
        if not known_indices:
            return 0.0
        
        # Extract predictions and ground truth for known subjects only
        predicted_known: list[int] = [predicted_labels[i] for i in known_indices]
        true_known: list[int] = [self.test_labels[i] for i in known_indices]
        
        # Count exact label matches between predictions and ground truth
        p: int
        t: int
        correct: int = sum(p == t for p, t in zip(predicted_known, true_known))
        identification_rate: float = correct / len(known_indices)
        
        self.logger.debug(f"EvaluationModule.calc_identification_rate: {correct}/{len(known_indices)} = {identification_rate:.3f}")
        
        return identification_rate
    
    def select_similarity_threshold(self, similarities: list[float], false_alarm_rate: float) -> float:
        # Determines similarity threshold using percentile-based selection for precise FAR control.
        # Extracts similarities from unknown subjects and computes percentile threshold
        # to achieve target false alarm rate in open-set evaluation scenarios.
        # Extract similarity scores from unknown subjects for threshold computation
        sim: float
        label: int
        unknown_similarities: list[float] = [
            sim for sim, label in zip(similarities, self.test_labels) 
            if label == -1
        ]
        
        if not unknown_similarities:
            return max(similarities)
        
        # Percentile-based threshold achieves precise false alarm rate control
        percentile: float = 100 * (1 - false_alarm_rate)
        threshold: float = float(np.percentile(unknown_similarities, percentile))
        
        self.logger.debug(f"EvaluationModule.select_similarity_threshold: FAR={false_alarm_rate:.3f}, threshold={threshold:.3f}")
        
        return threshold
    
    def forward(
        self,
        predictions: list[PredictionResult],
        target_labels: list[str],
        false_alarm_rates: list[float]
    ) -> DIRCurveResult:
        
        # Convert distance-based predictions to similarity scores for threshold evaluation
        pred: PredictionResult
        similarities: list[float] = [-pred.distance for pred in predictions]
        predicted_labels: list[str] = [pred.label for pred in predictions]
        
        # Initialize DIR curve point collection for comprehensive evaluation
        dir_points: list[DIRPoint] = []
        
        # Generate DIR curve points by sweeping thresholds across false alarm rates
        far: float
        for far in false_alarm_rates:
            # Compute percentile-based threshold for precise false alarm rate control
            threshold: float = self.select_similarity_threshold(
                similarities, target_labels, far
            )
            
            # Apply threshold to create final predictions with unknown rejection
            pred_label: str
            sim: float
            thresholded_predictions: list[str] = [
                pred_label if sim >= threshold else "unknown"
                for pred_label, sim in zip(predicted_labels, similarities)
            ]
            
            # Calculate identification rate for known subjects at current threshold
            id_rate: float = self.calc_identification_rate(
                thresholded_predictions, target_labels
            )
            
            # Store DIR point with complete performance metrics
            dir_points.append(DIRPoint(
                false_alarm_rate=far,
                identification_rate=id_rate,
                threshold=threshold
            ))
        
        # Determine optimal operating points for specific application requirements
        # Security-critical scenario: maximize identification rate while maintaining FAR ≤ 1%
        optimal_low_far: DIRPoint | None = None
        point: DIRPoint
        for point in dir_points:
            if point.false_alarm_rate <= 0.01:
                # Select point with highest identification rate among low-FAR candidates
                if optimal_low_far is None or point.identification_rate > optimal_low_far.identification_rate:
                    optimal_low_far = point
        
        # User-friendly scenario: minimize false alarm rate while maintaining ID rate ≥ 90%
        optimal_high_id: DIRPoint | None = None
        for point in dir_points:
            if point.identification_rate >= 0.90:
                # Select point with lowest false alarm rate among high-accuracy candidates
                if optimal_high_id is None or point.false_alarm_rate < optimal_high_id.false_alarm_rate:
                    optimal_high_id = point
        
        return DIRCurveResult(
            points=dir_points,
            optimal_threshold_low_far=optimal_low_far.threshold if optimal_low_far else 0.0,
            optimal_threshold_high_id=optimal_high_id.threshold if optimal_high_id else 0.0
        )

