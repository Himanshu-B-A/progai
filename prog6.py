import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the dataset
heartDisease = pd.read_csv(r"C:\Users\himub\OneDrive\Desktop\PROG\Labprog_AI AND ML\dataset.csv")

# Replace missing values with NaN
heartDisease = heartDisease.replace('?', np.nan)

# Print a few examples from the dataset
print('Few examples from the dataset:')
print(heartDisease.head())

# Define the structure of the Bayesian Network
model = BayesianModel([('age', 'heartdisease'), ('sex', 'heartdisease'), ('exang', 'heartdisease'), 
                       ('cp', 'heartdisease'), ('heartdisease', 'restecg'), ('heartdisease', 'chol')])

# Fit the model using Maximum Likelihood Estimator
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Perform inference using Variable Elimination
infer = VariableElimination(model)

# Query 1: Probability of heart disease given age = 28
q1 = infer.query(variables=['heartdisease'], evidence={'age': 28})
print('\n1. Probability of Heart Disease given age=28:')
print(q1)

# Query 2: Probability of heart disease given cholesterol level = 100
q2 = infer.query(variables=['heartdisease'], evidence={'chol': 100})
print('\n2. Probability of Heart Disease given chol=100:')
print(q2)
