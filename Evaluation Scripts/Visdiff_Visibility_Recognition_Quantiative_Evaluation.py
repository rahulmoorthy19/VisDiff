import pickle 
from combinatorial_evaluation import *

## Add Saved Pickle Here
with open("", "rb") as f:
    visdiff_predictions = pickle.load(f)
actual_visibility = list()
predicted_visibility = list()
validity = list()
test_dataset = "/home/isleri/mahes092/Polygon_Dataset_Generator/Minimim_Feature_25_Dataset_Non_Valid/"
for each_key in visdiff_predictions.keys():
    with open(test_dataset +  each_key, 'rb') as f:
        polygon_dict = pickle.load(f)
    current_prediction = visdiff_predictions[each_key]
    validity.append(current_prediction["visibility_validity"])
    predicted_visibility.append(current_prediction["predicted_visibility"])
    actual_visibility.append(current_prediction["true_visibility"])

Combinatorial_Evaluation().quantitative_measure_validity(validity, actual_visibility, predicted_visibility)