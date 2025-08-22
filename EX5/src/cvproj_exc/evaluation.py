import pickle
import numpy as np
from cvproj_exc.classifier import NearestNeighborClassifier

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_recognition_implementation import EvaluationModule

UNKNOWN_LABEL = -1

class OpenSetEvaluation:
    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        self.evaluation_module = EvaluationModule()
        self.evaluation_module.false_alarm_rate_range = false_alarm_rate_range
        self.classifier = classifier

    def prepare_input_data(self, train_data_file, test_data_file):
        self.evaluation_module.prepare_input_data(train_data_file, test_data_file)

    def run(self):
        return self.evaluation_module.run()

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        return self.evaluation_module.select_similarity_threshold(similarity, false_alarm_rate)

    def calc_identification_rate(self, prediction_labels):
        return self.evaluation_module.calc_identification_rate(prediction_labels)
