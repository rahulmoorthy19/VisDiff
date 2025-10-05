import pickle 
from combinatorial_evaluation import *

with open("/home/isleri/mahes092/Polygon_Dataset_Generator/Vertex_Prediction_Architecture/Evaluation_Scripts/Visdiff_Predictions_Visibility_Final_Ordering_Diffusion_Pixel_Aligned_Global_Unconditional.pkl", "rb") as f:
    visdiff_predictions = pickle.load(f)
actual_visibility = list()
predicted_visibility = list()
for each_key in visdiff_predictions.keys():
    current_prediction = visdiff_predictions[each_key]
    predicted_visibility.append(current_prediction["predicted_visibility"])
    actual_visibility.append(current_prediction["true_visibility"])
    
Combinatorial_Evaluation().visibility_visibility_measure(actual_visibility, predicted_visibility)
Combinatorial_Evaluation().calculate_quatitative_metrics_visibility(actual_visibility, predicted_visibility)