import pickle 
from combinatorial_evaluation import *

with open("/home/isleri/mahes092/Polygon_Dataset_Generator/Polygon_Visualization/Visdiff_Predictions_Visibility_Final_Ordering_Diffusion_Pixel_Aligned_Global_Dual.pkl", "rb") as f:
    visdiff_predictions = pickle.load(f)

actual_polygon = list()
predicted_polygon = list()

for each_key in visdiff_predictions.keys():
    current_prediction = visdiff_predictions[each_key]
    if current_prediction["predicted_polygon"] is not None:
        predicted_polygon.append(current_prediction["predicted_polygon"])
        actual_polygon.append(current_prediction["true_polygon"])
Combinatorial_Evaluation().triangulation_evaluation(actual_polygon, predicted_polygon)