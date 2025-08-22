import matplotlib.pyplot as plt
import numpy as np

from cvproj_exc.classifier import NearestNeighborClassifier
from cvproj_exc.config import Config
from cvproj_exc.evaluation import OpenSetEvaluation


def main():
    # The range of the false alarm rate in logarithmic space to draw DIR curves.
    false_alarm_rate_range = np.logspace(-3.0, 0, 1000, endpoint=False)

    # Pickle files containing embeddings and corresponding class labels for the
    # training and the test dataset.
    train_data_file = Config.EVAL_TRAIN_DATA
    test_data_file = Config.EVAL_TEST_DATA

    # We use a nearest neighbor classifier for this evaluation.
    classifier = NearestNeighborClassifier()

    # Prepare a new evaluation instance and feed training and test data into this evaluation.
    evaluation = OpenSetEvaluation(
        classifier=classifier, false_alarm_rate_range=false_alarm_rate_range
    )
    evaluation.prepare_input_data(train_data_file, test_data_file)

    # Run the evaluation and retrieve the performance measures (identification rates and
    # false alarm rates) on the test dataset.
    results = evaluation.run()

    # Plot the DIR curve (matching Chirag's style).
    plt.figure(figsize=(8, 6))
    plt.semilogx(results["false_alarm_rates"], results["identification_rates"], label='DIR Curve')
    plt.xlabel('False Alarm Rate (FAR)')
    plt.ylabel('Identification Rate (Rank-1)')
    plt.title('Detection and Identification Rate (DIR) Curve')
    plt.grid(True)
    plt.legend()
    
    # Save figure like Chirag
    figures_dir = Config.PROJECT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)
    save_path = figures_dir / "exercise_5_4_dir_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"DIR curve saved to: {save_path}")


if __name__ == "__main__":
    main()
