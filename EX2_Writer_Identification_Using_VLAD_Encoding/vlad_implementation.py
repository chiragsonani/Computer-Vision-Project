# VLAD + E-SVM Implementation for Writer Identification
# This file handles the main algorithms for Exercise 2:
# 1. Creates a "vocabulary" of handwriting patterns (dictionary learning)
# 2. Encodes each document using VLAD (basically a fancy histogram)
# 3. Makes encodings more discriminative with E-SVM (one SVM per test sample)
# 4. Finds which writer most likely wrote each test document

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import gzip
import _pickle as cPickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


# Base Module System
class Module:
    # Base class that all our processing classes inherit from. When you
    # call them, they run forward() method. It's like a contract saying
    # "hey, everything that processes data works the same way - call it
    # and it runs forward()".
    def __init__(self) -> None:
        pass
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


# Data Structures (these just hold our data, no processing)
@dataclass(frozen=True, slots=True)
class ClusterCenters:
    # Holds the K cluster centers (visual words) from k-means. These are
    # like the "alphabet" of handwriting patterns we found. Each center
    # is a 128-dim SIFT descriptor representing a common local pattern.
    centers: np.ndarray      # K x 128 matrix of cluster centers
    n_clusters: int          # K (usually 128)
    descriptor_dim: int      # Always 128 for SIFT
    
    def compute_distances(self, descriptors: np.ndarray) -> np.ndarray:
        # Calculates how far each descriptor is from each cluster center.
        # Used to figure out which cluster each descriptor belongs to.
        n_descriptors: int = descriptors.shape[0]
        distances: np.ndarray = np.zeros((n_descriptors, self.n_clusters), dtype=np.float32)
        
        for k in range(self.n_clusters):
            diff: np.ndarray = descriptors - self.centers[k]
            distances[:, k] = np.linalg.norm(diff, axis=1)
        
        return distances


@dataclass(frozen=True, slots=True)
class AssignmentMatrix:
    # Binary matrix showing which descriptors belong to which clusters.
    # It's TxK where T=number of descriptors, K=clusters. Each row has
    # exactly one 1 (hard assignment) showing which cluster that
    # descriptor is assigned to.
    assignments: np.ndarray  # TxK binary matrix
    n_descriptors: int       # T
    n_clusters: int          # K
    
    def get_assigned_indices(self, cluster_idx: int) -> np.ndarray:
        # Gets all descriptor indices that belong to a specific cluster.
        # Like asking "which descriptors were assigned to cluster 5?"
        indices: np.ndarray = np.where(self.assignments[:, cluster_idx] == 1)[0]
        return indices


@dataclass(frozen=True, slots=True)
class VLADEncoding:
    # The final VLAD vector for one document. It's basically K residual
    # vectors concatenated together. Each residual shows how the
    # descriptors in that cluster differ from the cluster center.
    encoding: np.ndarray     # K*D dimensional vector (16384 for us)
    n_clusters: int          # K
    descriptor_dim: int      # D (128)
    file_path: str           # Which document this came from
    
    @property
    def encoding_dim(self) -> int:
        return self.n_clusters * self.descriptor_dim  # 128 * 128 = 16384


@dataclass(frozen=True, slots=True)
class ESVMModel:
    # Stores an SVM model (we train one per test sample). Not actually
    # used in our implementation but kept for completeness.
    weights: np.ndarray      # SVM weight vector
    bias: float              # SVM bias term
    test_idx: int            # Which test sample this SVM is for


@dataclass(frozen=True, slots=True)
class SIFTDescriptors:
    # Holds SIFT descriptors loaded from a .pkl.gz file. Each document
    # has hundreds/thousands of these 128-dim descriptors extracted
    # from local patches.
    descriptors: np.ndarray  # TxD matrix (T varies per document)
    file_path: str           # Which file these came from
    n_descriptors: int       # T (varies, usually 100s-1000s)
    descriptor_dim: int      # D (always 128 for SIFT)


@dataclass(frozen=True, slots=True)
class DistanceMatrix:
    # Pairwise distances between all documents. Used to find which
    # documents are most similar (likely same writer).
    distances: np.ndarray    # NxN matrix of cosine distances
    n_samples: int           # N (number of documents)


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    # Performance metrics. Top-1 = how often we get the writer right.
    # mAP = average precision across all queries.
    top1_accuracy: float          # Percentage correct (0-1)
    mean_average_precision: float # mAP score (0-1)


# Processing Classes (these do the actual work)
class SIFTDataset(Module):
    # Loads pre-computed SIFT features from disk. The prof already
    # extracted SIFT descriptors and saved them as .pkl.gz files.
    # We just load them and wrap in our data structure.
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, file_path: str) -> SIFTDescriptors:
        # Open the compressed pickle file and load the descriptors
        with gzip.open(file_path, 'rb') as f:
            descriptors: np.ndarray = cPickle.load(f, encoding='latin1')
        
        n_descriptors: int = descriptors.shape[0]
        descriptor_dim: int = descriptors.shape[1]
        
        return SIFTDescriptors(
            descriptors=descriptors,
            file_path=file_path,
            n_descriptors=n_descriptors,
            descriptor_dim=descriptor_dim
        )


class DictionaryLearner(Module):
    # Learns the "visual vocabulary" using k-means clustering. Takes
    # ~500K random SIFT descriptors and finds K=128 cluster centers.
    # These centers represent common handwriting patterns across all
    # writers.
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, descriptors: np.ndarray, n_clusters: int) -> ClusterCenters:
        # MiniBatch k-means is faster than regular k-means for big data
        kmeans: MiniBatchKMeans = MiniBatchKMeans(
            n_clusters=n_clusters,    # K=128 visual words
            random_state=42,          # For reproducibility
            batch_size=1000,          # Process 1000 descriptors at a time
            n_init=3                  # Run k-means 3 times, pick best
        )
        
        kmeans.fit(descriptors)
        centers: np.ndarray = kmeans.cluster_centers_
        descriptor_dim: int = centers.shape[1]
        
        return ClusterCenters(
            centers=centers,
            n_clusters=n_clusters,
            descriptor_dim=descriptor_dim
        )


class AssignmentComputer(Module):
    # Assigns each SIFT descriptor to its nearest cluster center (hard
    # assignment). Creates a binary matrix where each descriptor gets
    # assigned to exactly one cluster - no fuzzy business here.
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, descriptors: np.ndarray, cluster_centers: ClusterCenters) -> AssignmentMatrix:
        # Calculate distance from each descriptor to each cluster
        distances: np.ndarray = cluster_centers.compute_distances(descriptors)
        
        # Create binary assignment matrix (all zeros initially)
        assignments: np.ndarray = np.zeros((descriptors.shape[0], cluster_centers.n_clusters), dtype=np.float32)
        # Find nearest cluster for each descriptor
        nearest_clusters: np.ndarray = np.argmin(distances, axis=1)
        
        # Set the assignment (put a 1 in the right spot for each descriptor)
        for t in range(descriptors.shape[0]):
            assignments[t, nearest_clusters[t]] = 1
        
        return AssignmentMatrix(
            assignments=assignments,
            n_descriptors=descriptors.shape[0],
            n_clusters=cluster_centers.n_clusters
        )


class VLADEncoder(Module):
    # The heart of VLAD. For each document, it:
    # 1. Assigns each descriptor to nearest cluster
    # 2. Computes residuals (how different from cluster center)
    # 3. Sums residuals per cluster
    # 4. Concatenates into one big vector
    # Think of it as a fancy histogram that also captures HOW different
    # the descriptors are from typical patterns.
    def __init__(self, powernorm: bool = False) -> None:
        super().__init__()
        self.powernorm = powernorm  # Whether to apply power normalization
    
    def forward(self, files: list[str], cluster_centers: ClusterCenters) -> list[np.ndarray]:
        dataset: SIFTDataset = SIFTDataset()
        assignment_computer: AssignmentComputer = AssignmentComputer()
        encodings: list[np.ndarray] = []
        
        # Process each document
        for file_path in tqdm(files):
            # Load SIFT descriptors for this document
            sift_data: SIFTDescriptors = dataset(file_path)
            descriptors: np.ndarray = sift_data.descriptors
            
            # Assign each descriptor to a cluster
            assignment_matrix: AssignmentMatrix = assignment_computer(descriptors, cluster_centers)
            
            # Initialize VLAD vector (K*D dimensions)
            vlad_dim: int = cluster_centers.n_clusters * cluster_centers.descriptor_dim
            vlad_encoding: np.ndarray = np.zeros(vlad_dim, dtype=np.float32)
            
            # For each cluster, aggregate residuals
            for k in range(cluster_centers.n_clusters):
                # Get all descriptors assigned to this cluster
                assigned_indices: np.ndarray = assignment_matrix.get_assigned_indices(k)
                
                if len(assigned_indices) > 0:
                    # Calculate residuals (descriptor - cluster_center)
                    assigned_descriptors: np.ndarray = descriptors[assigned_indices]
                    residuals: np.ndarray = assigned_descriptors - cluster_centers.centers[k]
                    # Sum all residuals for this cluster
                    aggregated_residual: np.ndarray = np.sum(residuals, axis=0)
                    
                    # Put it in the right spot in VLAD vector
                    start_idx: int = k * cluster_centers.descriptor_dim
                    end_idx: int = (k + 1) * cluster_centers.descriptor_dim
                    vlad_encoding[start_idx:end_idx] = aggregated_residual
            
            # Power normalization - reduces influence of frequent patterns
            if self.powernorm:
                vlad_encoding = np.sign(vlad_encoding) * np.sqrt(np.abs(vlad_encoding))
            
            # L2 normalize to unit length
            vlad_encoding = normalize(vlad_encoding.reshape(1, -1), norm='l2')[0]
            
            encodings.append(vlad_encoding)
        
        return encodings


class ESVMClassifier(Module):
    # E-SVM (Exemplar SVM) - the fancy part that makes each test encoding
    # more discriminative. For EACH test document, we train an SVM where:
    # - Positive example: just that one test document
    # - Negative examples: ALL training documents
    # Then we use the SVM weights AS the new encoding (not multiplication).
    # It's like asking "what makes THIS document special compared to all
    # training docs?" and representing it with those discriminative features.
    def __init__(self, C: float = 1000) -> None:
        super().__init__()
        self.C = C  # SVM regularization parameter
    
    def _compute_single_esvm(self, i: int, test_sample: np.ndarray, train_encodings: np.ndarray) -> tuple[int, np.ndarray]:
        # Trains one SVM for one test sample
        positive_sample: np.ndarray = test_sample.reshape(1, -1)
        negative_samples: np.ndarray = train_encodings
        n_train: int = train_encodings.shape[0]
        
        # Stack positive and all negatives
        X: np.ndarray = np.vstack([positive_sample, negative_samples])
        # Labels: 1 for test sample, -1 for all training samples
        y: np.ndarray = np.hstack([1, -np.ones(n_train, dtype=np.int32)])
        
        # Train linear SVM with balanced class weights
        svm: LinearSVC = LinearSVC(C=self.C, class_weight='balanced', random_state=42, max_iter=10000)
        svm.fit(X, y)
        
        # Use SVM weights directly as the new encoding (NOT multiplication!)
        weights: np.ndarray = svm.coef_[0]
        weights_normalized: np.ndarray = weights / np.linalg.norm(weights)
        
        # Return normalized weights as the new encoding
        return i, weights_normalized.reshape(1, -1)
    
    def forward(self, test_encodings: np.ndarray, train_encodings: np.ndarray) -> np.ndarray:
        # Process all test samples in parallel (3600 SVMs!)
        n_test: int = test_encodings.shape[0]
        n_train: int = train_encodings.shape[0]
        encoding_dim: int = test_encodings.shape[1]
        
        # Use all CPU cores for parallel processing
        n_cores: int = multiprocessing.cpu_count()
        print(f"\n> Using parallel processing with {n_cores} CPU cores")
        print(f"> Processing {n_test} test samples with E-SVM")
        
        # Train one SVM per test sample in parallel
        results: list[tuple[int, np.ndarray]] = Parallel(
            n_jobs=n_cores,
            backend='threading',  # Threading avoids pickle overhead
            verbose=0
        )(
            delayed(self._compute_single_esvm)(i, test_encodings[i], train_encodings)
            for i in tqdm(range(n_test), desc="E-SVM computation")
        )
        
        # Sort by index to maintain order
        results.sort(key=lambda x: x[0])
        new_encodings_list: list[np.ndarray] = [enc for _, enc in results]
        new_encodings: np.ndarray = np.concatenate(new_encodings_list, axis=0)
        
        return new_encodings


class DistanceComputer(Module):
    # Calculates cosine distance between all pairs of documents.
    # Cosine distance = 1 - cosine similarity. Small distance means
    # similar writing style (likely same writer).
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, encodings: np.ndarray) -> DistanceMatrix:
        # Normalize to unit vectors first
        encodings_normalized: np.ndarray = normalize(encodings, norm='l2')
        
        # Dot product of normalized vectors = cosine similarity
        dot_products: np.ndarray = np.dot(encodings_normalized, encodings_normalized.T)
        # Convert to distance (1 - similarity)
        distances: np.ndarray = 1 - dot_products
        
        # Set diagonal to max (document is not similar to itself for retrieval)
        np.fill_diagonal(distances, np.finfo(distances.dtype).max)
        
        return DistanceMatrix(
            distances=distances,
            n_samples=encodings.shape[0]
        )


class VLADProcessor(Module):
    # Main orchestrator that skeleton.py calls. Manages the whole
    # pipeline and provides simple methods that match what skeleton.py
    # expects. It's like the conductor of our VLAD orchestra.
    def __init__(self) -> None:
        super().__init__()
        self.dictionary_learner = DictionaryLearner()
        self.assignment_computer = AssignmentComputer()
        self.vlad_encoder: VLADEncoder | None = None
        self.esvm_classifier: ESVMClassifier | None = None
        self.distance_computer = DistanceComputer()
    
    def set_powernorm(self, powernorm: bool) -> None:
        # Create VLAD encoder with power normalization setting
        self.vlad_encoder = VLADEncoder(powernorm=powernorm)
    
    def set_svm_C(self, C: float) -> None:
        # Create E-SVM classifier with regularization parameter
        self.esvm_classifier = ESVMClassifier(C=C)
    
    def learn_dictionary(self, descriptors: np.ndarray, n_clusters: int) -> np.ndarray:
        # Wrapper for dictionary learning (skeleton.py expects just the centers)
        cluster_centers: ClusterCenters = self.dictionary_learner(descriptors, n_clusters)
        return cluster_centers.centers
    
    def compute_assignments(self, descriptors: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        # Wrapper for assignment computation (skeleton.py expects just the matrix)
        n_clusters: int = clusters.shape[0]
        descriptor_dim: int = clusters.shape[1]
        
        cluster_centers: ClusterCenters = ClusterCenters(
            centers=clusters,
            n_clusters=n_clusters,
            descriptor_dim=descriptor_dim
        )
        
        assignment_matrix: AssignmentMatrix = self.assignment_computer(descriptors, cluster_centers)
        return assignment_matrix.assignments
    
    def encode_vlad(self, files: list[str], clusters: np.ndarray, powernorm: bool = False) -> list[np.ndarray]:
        # Wrapper for VLAD encoding
        if self.vlad_encoder is None:
            self.set_powernorm(powernorm)
        
        n_clusters: int = clusters.shape[0]
        descriptor_dim: int = clusters.shape[1]
        
        cluster_centers: ClusterCenters = ClusterCenters(
            centers=clusters,
            n_clusters=n_clusters,
            descriptor_dim=descriptor_dim
        )
        
        encodings: list[np.ndarray] = self.vlad_encoder(files, cluster_centers)
        return encodings
    
    def apply_esvm(self, test_encodings: np.ndarray, train_encodings: np.ndarray, C: float = 1000) -> np.ndarray:
        # Wrapper for E-SVM transformation
        if self.esvm_classifier is None:
            self.set_svm_C(C)
        
        new_encodings: np.ndarray = self.esvm_classifier(test_encodings, train_encodings)
        return new_encodings
    
    def compute_distances(self, encodings: np.ndarray) -> np.ndarray:
        # Wrapper for distance computation
        distance_matrix: DistanceMatrix = self.distance_computer(encodings)
        return distance_matrix.distances