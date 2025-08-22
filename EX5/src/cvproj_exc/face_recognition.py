import os
import pickle

import cv2
import numpy as np

from cvproj_exc.config import Config
from face_recognition_implementation import FaceClusteringModule, FaceEmbedding, ClusterAssignment, ClusterState, FaceRecognizerModule


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.facenet = cv2.dnn.readNetFromONNX(str(Config.RESNET50))

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    @classmethod
    @property
    def get_embedding_dimensionality(cls):
        """Get dimensionality of the extracted embeddings."""
        return 128


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours: int = 5, max_distance: float = 0.7, min_prob: float = 0.8) -> None:
        self.recognizer_module: FaceRecognizerModule = FaceRecognizerModule(
            k=num_neighbours, 
            max_distance=max_distance, 
            min_prob=min_prob
        )

        # The underlying gallery: class labels and embeddings.
        self.labels: list[str] = []
        self.embeddings: np.ndarray = np.empty((0, FaceNet.get_embedding_dimensionality))

        # Load face recognizer from pickle file if available.
        if os.path.exists(Config.REC_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self) -> None:
        print("FaceRecognizer saving: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "wb") as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self) -> None:
        print("FaceRecognizer loading: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "rb") as f:
            (self.labels, self.embeddings) = pickle.load(f)
            self.recognizer_module._labels = list(self.labels) if isinstance(self.labels, list) else self.labels.tolist()
            self.recognizer_module._embeddings = np.array(self.embeddings, dtype=np.float32)
            print(f"Loaded {len(self.recognizer_module._labels)} labels and {len(self.recognizer_module._embeddings)} embeddings")
            self.recognizer_module.logger.info(f"Gallery loaded: {len(self.recognizer_module._labels)} labels, {self.recognizer_module._embeddings.shape} embeddings")

    # TODO: Train face identification with a new face with labeled identity.
    def partial_fit(self, face: np.ndarray, label: str) -> None:
        self.recognizer_module.partial_fit(face, label)
        self.labels = self.recognizer_module._labels
        self.embeddings = self.recognizer_module._embeddings

    # TODO: Predict the identity for a new face.
    def predict(self, face: np.ndarray) -> tuple[str, float]:
        predicted_label: str
        probability: float
        predicted_label, probability = self.recognizer_module.predict(face)
        return predicted_label, probability


# The FaceClustering class enables unsupervised clustering of face images according to their
# identity and re-identification.
class FaceClustering:

    def __init__(self, num_clusters: int = 2, max_iter: int = 25) -> None:
        self.facenet: FaceNet = FaceNet()
        self.clustering_module: FaceClusteringModule = FaceClusteringModule(k=num_clusters)
        
        self.embeddings: np.ndarray = np.empty((0, FaceNet.get_embedding_dimensionality))
        self.num_clusters: int = num_clusters
        self.cluster_center: np.ndarray = np.empty((num_clusters, FaceNet.get_embedding_dimensionality))
        self.cluster_membership: list[int] = []
        self.max_iter: int = max_iter

        if os.path.exists(Config.CLUSTER_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceClustering saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "wb") as f:
            pickle.dump(
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
                f,
            )

    # Load trained model from a pickle file.
    def load(self):
        print("FaceClustering loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "rb") as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = (
                pickle.load(f)
            )
        
        # Restore cluster state in our module after loading
        if self.cluster_center is not None and len(self.cluster_center) > 0:
            assignments: list[ClusterAssignment] = []
            for idx, cluster_id in enumerate(self.cluster_membership):
                if idx < len(self.embeddings):
                    embedding_vector: np.ndarray = self.embeddings[idx]
                    distances: list[float] = [
                        float(np.linalg.norm(embedding_vector - center)) 
                        for center in self.cluster_center
                    ]
                    assignments.append(ClusterAssignment(
                        cluster_id=cluster_id,
                        distance_to_center=distances[cluster_id],
                        all_distances=distances
                    ))
            
            self.clustering_module._cluster_state = ClusterState(
                centers=list(self.cluster_center),
                assignments=assignments,
                objective_values=[],
                iteration=0,
                converged=True
            )

    def partial_fit(self, face: np.ndarray) -> None:
        embedding: np.ndarray = self.facenet.predict(face)
        face_emb: FaceEmbedding = FaceEmbedding(vector=embedding, label=None)
        self.clustering_module.partial_fit(face_emb)
        self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])

    def fit(self) -> None:
        from pathlib import Path
        
        cluster_state: ClusterState = self.clustering_module.fit()
        self.cluster_center = np.array(cluster_state.centers)
        self.cluster_membership = [a.cluster_id for a in cluster_state.assignments]
        
        figures_dir: Path = Path(__file__).parent.parent.parent / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.clustering_module.save_convergence_analysis(str(figures_dir), num_runs=5)
        
        viz_path: Path = figures_dir / 'exercise_5_3_cluster_visualization.png'
        self.clustering_module.save_cluster_visualization(str(viz_path))

    def predict(self, face: np.ndarray) -> tuple[int, np.ndarray]:
        embedding: np.ndarray = self.facenet.predict(face)
        face_emb: FaceEmbedding = FaceEmbedding(vector=embedding, label=None)
        assignment: ClusterAssignment = self.clustering_module.predict(face_emb)
        return int(assignment.cluster_id), np.array(assignment.all_distances, dtype=np.float32)
