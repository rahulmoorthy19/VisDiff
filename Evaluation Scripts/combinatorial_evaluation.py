import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class Combinatorial_Evaluation:
    def visibility_visibility_measure(self, actual_visibility_batch, predicted_visibility_batch):
        batch_size = len(actual_visibility_batch)
        scores = np.zeros(batch_size)
        scores_one = np.zeros(batch_size)
        scores_zero = np.zeros(batch_size)
        act_num_edges = np.zeros(batch_size)
        pred_num_edges = np.zeros(batch_size)
        pred_correct_num_edges = np.zeros(batch_size)
        for i in range(batch_size):
            actual_visibility = actual_visibility_batch[i]
            predicted_visibility = predicted_visibility_batch[i]
            correct_count = np.sum(actual_visibility == predicted_visibility)
            total_count = actual_visibility.shape[0] * actual_visibility.shape[1]
            scores[i] = correct_count / total_count
            scores_one[i] = np.sum(np.logical_and(predicted_visibility == 1, actual_visibility == 1))/np.sum(actual_visibility == 1)
            scores_zero[i] = np.sum(np.logical_and(predicted_visibility == 0, actual_visibility == 0))/np.sum(actual_visibility == 0)
            act_num_edges[i] = np.sum(actual_visibility == 1)/(actual_visibility.shape[0] * actual_visibility.shape[1])
            pred_num_edges[i] = np.sum(predicted_visibility == 1)/(predicted_visibility.shape[0] * predicted_visibility.shape[1])
            pred_correct_num_edges[i] = np.sum(np.logical_and(predicted_visibility == 1, actual_visibility == 1))/(predicted_visibility.shape[0] * predicted_visibility.shape[1])
            if scores_zero[i] == 0:
                print(predicted_visibility)
        print("Visibility Edges Accuracy", np.round(np.mean(scores_one),3), "Non Visibility Edges Accuracy", np.round(np.mean(scores_zero),3))

    def calculate_quatitative_metrics_visibility(self, actual_visibility_batch, predicted_visibility_batch):
        batch_size = len(actual_visibility_batch)
        acc_scores = np.zeros(batch_size)
        prec_scores = np.zeros(batch_size)
        recall_scores = np.zeros(batch_size)
        f1_scores = np.zeros(batch_size)

        for i in range(batch_size):
            actual_visibility = actual_visibility_batch[i]
            predicted_visibility = predicted_visibility_batch[i]
            acc = accuracy_score(actual_visibility.flatten(), predicted_visibility.flatten())
            prec = precision_score(actual_visibility.flatten(), predicted_visibility.flatten(), average = "macro")
            rec = recall_score(actual_visibility.flatten(), predicted_visibility.flatten(), average = "macro")
            f1 = f1_score(actual_visibility.flatten(), predicted_visibility.flatten(), average = "macro")
            acc_scores[i] = acc
            prec_scores[i] = prec
            recall_scores[i] = rec
            f1_scores[i] = f1
        print("Accuracy", np.round(np.mean(acc_scores),3), "Precision", np.round(np.mean(prec_scores),3), "Recall", np.round(np.mean(recall_scores),3), "f1-Score", np.round(np.mean(f1_scores),3))

    def calculate_quatitative_metrics_visibility_single(self, actual_visibility, predicted_visibility, return_value = False):
        acc = accuracy_score(actual_visibility.flatten(), predicted_visibility.flatten())
        prec = precision_score(actual_visibility.flatten(), predicted_visibility.flatten(), average = "macro")
        rec = recall_score(actual_visibility.flatten(), predicted_visibility.flatten(), average = "macro")
        f1 = f1_score(actual_visibility.flatten(), predicted_visibility.flatten(), average = "macro")
        if not return_value:
            print("Accuracy", np.round(acc,3), "Precision", np.round(prec,3), "Recall", np.round(rec,3), "f1-Score", np.round(f1,3))
        else:
            return {"Accuracy": np.round(acc,3), "Precision":np.round(prec,3), "Recall":np.round(rec,3), "f1-Score":np.round(f1,3)}
    
    def rotate_polygon(self, polygon):
        # First vertex as origin (0, 0)
        origin = polygon[0]

        # Second vertex to define the x-axis
        x_axis_point = polygon[1]

        # Vector from the first vertex to the second
        x_vector = x_axis_point - origin

        # Calculate the angle to rotate the x_vector to align with the positive x-axis
        angle = np.arctan2(x_vector[1], x_vector[0])

        # Create a rotation matrix
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)]
        ])

        # Rotate all points using the rotation matrix
        rotated_polygon = np.dot(polygon - origin, rotation_matrix)

        return rotated_polygon
    
    def chamfer_distance(self,A, B):
        """
        Compute the Chamfer Distance between two sets of points A and B.

        Parameters:
        A (numpy.ndarray): Point set A of shape (N, D), where N is the number of points and D is the dimensionality.
        B (numpy.ndarray): Point set B of shape (M, D), where M is the number of points and D is the dimensionality.

        Returns:
        float: Chamfer Distance between A and B.
        """
        # Compute pairwise distances between points in A and points in B
        diff = np.expand_dims(A, axis=1) - np.expand_dims(B, axis=0)  # Shape: (N, M, D)
        distances = np.linalg.norm(diff, axis=-1)  # Shape: (N, M), Euclidean distances between A and B

        # For each point in A, find the minimum distance to a point in B
        min_dist_A_to_B = np.min(distances, axis=1)
        # For each point in B, find the minimum distance to a point in A
        min_dist_B_to_A = np.min(distances, axis=0)

        # Chamfer Distance is the average of the minimum distances
        chamfer_dist = np.mean(min_dist_A_to_B) + np.mean(min_dist_B_to_A)
        
        return chamfer_dist

    def all_orderings(self, polygon):
        """Generate all cyclic orderings and their reversals"""
        n = len(polygon)
        cyclics = [np.roll(polygon, -i, axis=0) for i in range(n)]
        reversed_cyclics = [c[::-1] for c in cyclics]
        return cyclics + reversed_cyclics

    def triangulation_evaluation(self, actual, predicted):
        distances = list()
        # euclidian_distance = list()
        # iou_distance = list()
        min_cd = float('inf')
        for i in range(len(predicted)):
            actual_pol = self.rotate_polygon(actual[i])
            generated_orderings = self.all_orderings(predicted[i])
            for B_variant in generated_orderings:
                pred_pol = self.rotate_polygon(B_variant)
                # cd = self.chamfer_distance(actual_pol, pred_pol)
                cd = np.linalg.norm(actual_pol - pred_pol, axis=1).mean()
                if cd < min_cd:
                    min_cd = cd

            # distances.append(self.chamfer_distance(actual_pol, pred_pol))
            distances.append(min_cd)
        distances = np.array(distances)
        # print("Chamfer Distance Difference", np.round(np.mean(distances),3))
        print("Euclidian Distance Difference", np.round(np.mean(distances),3))

    def quantitative_measure_validity(self, actual_validity, actual_visibility_batch, predicted_visibility_batch):
        batch_size = len(actual_visibility_batch)
        acc_scores = np.zeros(batch_size)
        prec_scores = np.zeros(batch_size)
        recall_scores = np.zeros(batch_size)
        f1_scores = np.zeros(batch_size)
        for i in range(batch_size):
            actual_visibility = actual_visibility_batch[i]
            predicted_visibility = predicted_visibility_batch[i]
            acc = accuracy_score(actual_visibility.flatten(), predicted_visibility.flatten())
            prec = precision_score(actual_visibility.flatten(), predicted_visibility.flatten())
            rec = recall_score(actual_visibility.flatten(), predicted_visibility.flatten())
            f1 = f1_score(actual_visibility.flatten(), predicted_visibility.flatten())

            acc_scores[i] = acc
            prec_scores[i] = prec
            recall_scores[i] = rec
            f1_scores[i] = f1
        f1_score_threshold = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        accuracy_list_classification = list()
        f1_list_classification = list()
        for each in f1_score_threshold:
            accuracy_list_classification_threshold = list()
            for i in range(batch_size):
                if f1_scores[i] > each:
                    accuracy_list_classification_threshold.append(1)
                else:
                    accuracy_list_classification_threshold.append(0)
            acc = np.round(accuracy_score(actual_validity, accuracy_list_classification_threshold)*100)
            f1 = np.round(f1_score(actual_validity, accuracy_list_classification_threshold),2)
            accuracy_list_classification.append(acc)
            f1_list_classification.append(f1)
            print(each)
            print(confusion_matrix(actual_validity, accuracy_list_classification_threshold))

        data = {
        'Threshold': f1_score_threshold,
        'Accuracy': accuracy_list_classification
        }

        # Convert the data to a pandas DataFrame
        df = pd.DataFrame(data)

        # Modify the axis labels to "Threshold" for x-axis and "Accuracy %" for y-axis

        plt.figure(figsize=(16, 6))

        # Create the plot with larger text sizes
        # ax = sns.lineplot(
        # x='Threshold',
        # y='Accuracy',
        # data=df,
        # marker='o',
        # markersize=20,
        # linewidth=10,
        # color="steelblue"
        # )
        ax = sns.barplot(
            x='Threshold',
            y='Accuracy',
            data=df,
            color="steelblue",
            width=0.4  # Adjust bar width as needed
        )
        ax.set_ylim(40, 90)
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Update labels and title with appropriate axes names and font sizes
        # plt.title('Model Accuracy over Thresholds', fontsize=20)
        # plt.xlabel('Threshold', fontsize=20, fontweight='bold')
        # plt.ylabel('Accuracy', fontsize=20, fontweight='bold')

        # Increase the size of tick labels
        plt.xticks(fontsize=40, fontweight='bold')
        plt.yticks(fontsize=40, fontweight='bold')

        # Remove top and right spines for a clean look
        sns.despine()

        # Show the updated plot
        plt.savefig("visibility_graph_classification_accuracy.pdf", format='pdf', bbox_inches="tight", transparent=True, dpi=300)

    #     plt.clf()

    #     data = {
    #     'Threshold': f1_score_threshold,
    #     'F1': f1_list_classification
    #     }

    #     # Convert the data to a pandas DataFrame
    #     df = pd.DataFrame(data)

    #     # Modify the axis labels to "Threshold" for x-axis and "Accuracy %" for y-axis

    #     plt.figure(figsize=(8, 6))

    #     # Create the plot with larger text sizes
    #     # sns.lineplot(x='Threshold', y='F1', data=df, marker='o')
    #     sns.lineplot(
    #     x='Threshold',
    #     y='F1',
    #     data=df,
    #     marker='o',
    #     markersize=20,
    #     linewidth=10,
    #     color="steelblue"
    # )

    #     # Update labels and title with appropriate axes names and font sizes
    #     # plt.title('Model Accuracy over Thresholds', fontsize=20)
    #     plt.xlabel('Threshold', fontsize=16)
    #     plt.ylabel('F1', fontsize=16)

    #     # Increase the size of tick labels
    #     plt.xticks(fontsize=14)
    #     plt.yticks(fontsize=14)

    #     # Remove top and right spines for a clean look
    #     sns.despine()

    #     # Show the updated plot
    #     plt.savefig("visibility_graph_classification_f1.pdf", format='pdf', bbox_inches="tight", transparent=True, dpi=300)