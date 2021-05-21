## python implementation for the GIME method in IJCAI 2021 "Learning Groupwise Explanations for Black-Box Models"
## Packages
- python 3.6.9
- torch 1.2.0
- numpy 1.16.2
- scikit-learn 0.20.2
- scipy 1.2.1
## Structure
- dataset
- gime, the main implementation of our method
- utils.py, helper functions
- run.py, the pipeline to run our method
## Run
python run.py, which loads the dataset (e.g., Wine), trains the target model (e.g., SVR), generates groupwise explanations, and evaluates the generalized fidelity.

