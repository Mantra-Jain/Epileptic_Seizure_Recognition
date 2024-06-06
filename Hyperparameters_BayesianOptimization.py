#Import Packages
import numpy as np
import pandas as pd
import torch
from fastai.tabular.all import *
from bayes_opt import BayesianOptimization
# Data Preparation
## Load the Data
raw_data = pd.read_csv("Data/data.csv")
#print(raw_data.head())
# Bayesian Optimization
# Define the Functions

def fit_with(lr:float, wd:float, dp:float):
  # create a Learner
  config = tabular_config(embed_p=dp, ps=wd)
  learn = tabular_learner(raw_data, layers=[200,100], metrics=accuracy, config=config)
  # Train for x epochs
  with learn.no_bar():
    learn.fit_one_cycle(3, lr)
  # Save, print, and return the overall accuracy
  acc = float(learn.validate()[1])
  return acc

def fit_with(lr:float, wd:float, dp:float, n_layers:float, layer_1:float, layer_2:float, layer_3:float):
  print(lr, wd, dp)
  if int(n_layers) == 2:
    layers = [int(layer_1), int(layer_2)]
  elif int(n_layers) == 3:
    layers = [int(layer_1), int(layer_2), int(layer_3)]
  else:
    layers = [int(layer_1)]
  config = tabular_config(embed_p=float(dp),
                          ps=float(wd))
  learn = tabular_learner(dls, layers=layers, metrics=accuracy, config = config)
  with learn.no_bar() and learn.no_logging():
    learn.fit(5, lr=float(lr))
  acc = float(learn.validate()[1])
  return acc

#Set the variables

raw_data['y'].replace(2, 0, inplace=True)
raw_data['y'].replace(3, 0, inplace=True)
raw_data['y'].replace(4, 0, inplace=True)
raw_data['y'].replace(5, 0, inplace=True)
cont_names_data = list(raw_data.columns.values)
cont_names_data.remove("Unnamed: 0")
cont_names_data.remove('y')

procs = [Categorify, FillMissing, Normalize]
y_names = 'y'
y_block = CategoryBlock()
splits = RandomSplitter(valid_pct=0.2, seed=None)(range_of(raw_data))

to = TabularPandas(
    raw_data,
    procs=procs,
    cont_names=cont_names_data,
    y_names=y_names,
    y_block=y_block,
    splits=splits
)

dls = to.dataloaders(bs=256)

#Declare Hyperparameters
hps = {'lr': (1e-10, 1e-01),
      'wd': (4e-4, 0.4),
      'dp': (0.01, 0.5),
       'n_layers': (1,3),
       'layer_1': (50, 200),
       'layer_2': (100, 1000),
       'layer_3': (200, 2000)}

optim = BayesianOptimization(
    f = fit_with, # our fit function
    pbounds = hps, # our hyper parameters to tune
    verbose = 2, # 1 prints out when a maximum is observed, 0 for silent
    random_state=1
)
optim.maximize(n_iter=100)

print(optim.max)

#Darker/purple colors correspond to earlier samples and lighter/yellow colors correspond to later samples. 
# A red point shows the location of the minimum found by the optimization process.