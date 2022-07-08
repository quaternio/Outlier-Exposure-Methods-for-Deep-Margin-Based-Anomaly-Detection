import sklearn.neighbors
import torch

class KNNRepresentationEvaluator:
    def score(
            self,
            training_features: torch.Tensor,
            training_targets: torch.Tensor,
            testing_features: torch.Tensor,
            testing_targets: torch.Tensor) -> None:
        best_accuracy = 0.0
        for n_neighbors in [1, 2, 4, 8, 16, 32, 64, 128]:
            knn_classifier = sklearn.neighbors.KNeighborsClassifier(
                n_neighbors = n_neighbors)
            knn_classifier.fit(
                training_features.cpu().numpy(),
                training_targets.cpu().numpy())
            knn_accuracy = knn_classifier.score(
                testing_features.cpu().numpy(),
                testing_targets.cpu().numpy())
            best_accuracy = knn_accuracy if knn_accuracy > best_accuracy \
                else best_accuracy
        return best_accuracy
