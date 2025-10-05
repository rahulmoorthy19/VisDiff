# VisDiff: SDF-Guided Polygon Generation for Visibility Reconstruction, Characterization and Recognition

This repository contains the code of the work "**VisDiff: SDF-Guided Polygon Generation for Visibility Reconstruction, Characterization and Recognition**" which is accepted at **Neural Information Processing Systems (NeurIPS) 2025**. Please find the final trained models and dataset [here](https://drive.google.com/drive/u/0/folders/1CulaZ5ZJniKtuIr2d32VLMfZ5y1v8XRp).

## Train

To train your own model of VisDiff:

* Unzip the training data from link provided [here](https://drive.google.com/drive/u/0/folders/1CulaZ5ZJniKtuIr2d32VLMfZ5y1v8XRp) and change the relevant paths for the dataset in the next steps.
* Create the SDF of the training data using the script provided in SDF_Generator folder with the following command:
```
python SDF_Polygon_Generator.py
```
* Train the model using the script in the Training Scripts folder with the following commands:
```
## Difusion Model Training

python VisDiff_Diffusion.py

## Vertex Prediction Model Training

python VisDiff_Vertex_Prediction.py
```
Please make sure to make the relevant path changes in the code for model saving and dataset.

## Evaluation

To evaluate the trained models:

* Unzip the testing data from link provided [here](https://drive.google.com/drive/u/0/folders/1CulaZ5ZJniKtuIr2d32VLMfZ5y1v8XRp) and change the relevant paths for the dataset in the next steps.
* Evaluate VisDiff using the scripts in the Evaluation Scripts folder with the following commands:
```
## Visibility Reconstruction Evaluation

python Visibility_Reconstruction_Evaluation.py
python Visdiff_Visibility_Reconstruction_Quantiative_Evaluation.py

## Visibility Recognition Evaluation

python Visibility_Recogniton_Evaluation.py
python Visdiff_Visibility_Recognition_Quantiative_Evaluation.py

## Visibility Charecterization Evaluation
python Visibility_Charecterization_Evaluation.py

## Triangulation Graph Evaluation
python Visdiff_Triangulation_Reconstruction_Quantiative_Evaluation.py
python Triangulation_Recontruction_Evaluation.py
```
Please make sure to make the relevant path changes in the code for model loading and dataset.

## Interpolation

To perform polygon-to-polygon and graph-to-graph interpolation with the trained models, use the scripts in the Interpolation Scripts folder :

```
## Polygon to Polygon Interpolation
python Polygon_to_Polygon_Interpolation.py

## Graph to Graph Interpolation
python Graph_to_Graph_Interpolation.py
```

## To Cite

```
@misc{moorthy2025visdiffsdfguidedpolygongeneration,
      title={VisDiff: SDF-Guided Polygon Generation for Visibility Reconstruction and Recognition}, 
      author={Rahul Moorthy and Jun-Jee Chao and Volkan Isler},
      year={2025},
      eprint={2410.05530},
      archivePrefix={arXiv},
      primaryClass={cs.CG},
      url={https://arxiv.org/abs/2410.05530}, 
}
```