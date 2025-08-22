import cv2
import numpy as np
from mtcnn import MTCNN
from face_recognition_implementation import FaceTrackerModule, FaceDetectorModule


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=25, tm_threshold=0.2, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # Store template matching parameters
        self.tm_window_size = tm_window_size
        self.tm_threshold = tm_threshold
        
        # Initialize tracker module with parameters
        self.tracker = FaceTrackerModule(tm_window_size=tm_window_size, tm_threshold=tm_threshold)
        
        # Initialize our face detector module for Ex 5.1 figure generation
        self.face_detector_module = FaceDetectorModule()

    # Track a face in a new image using template matching.
    def track_face(self, image):
        # First try tracking with existing state
        bbox = self.tracker(image)
        
        # If tracking fails, try detecting
        if bbox is None:
            face_dict = self.detect_face(image)
            if face_dict is not None:
                # Initialize tracker with new detection
                x, y, w, h = face_dict["rect"]
                from face_recognition_implementation import BoundingBox
                detection = BoundingBox(x=x, y=y, width=w, height=h, confidence=face_dict.get("response", 0.9))
                bbox = self.tracker(image, detection=detection)
                if bbox:
                    # Update reference for template matching
                    self.reference = image[y:y+h, x:x+w].copy() if len(image.shape) == 2 else cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY).copy()
                    # Return in professor's expected format
                    return {
                        "rect": [bbox.x, bbox.y, bbox.width, bbox.height],
                        "image": image,
                        "aligned": self.align_face(image, [bbox.x, bbox.y, bbox.width, bbox.height]),
                        "response": bbox.confidence
                    }
            return None
        
        # Tracking succeeded, return in professor's expected format
        if bbox:
            return {
                "rect": [bbox.x, bbox.y, bbox.width, bbox.height],
                "image": image,
                "aligned": self.align_face(image, [bbox.x, bbox.y, bbox.width, bbox.height]),
                "response": bbox.confidence
            }
        return None

    # Face detection in a new image.
    def detect_face(self, image):
        # Use our FaceDetectorModule which handles figure generation
        bbox = self.face_detector_module(image)
        
        if bbox is None:
            self.reference = None
            return None
        
        # Convert BoundingBox to professor's expected format
        face_rect = [bbox.x, bbox.y, bbox.width, bbox.height]
        
        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": bbox.confidence}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(
            self.crop_face(image, face_rect),
            dsize=(self.aligned_image_size, self.aligned_image_size),
        )

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]
