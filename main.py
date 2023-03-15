import numpy as np
from sklearn.model_selection import train_test_split

from .my_model_selection import *
X = np.load("local_path_data") # local path to npy file
y = np.load("local_path_labels") # local path to npy file

# Split data into training and testing sets, Assuming X and y are your feature matrix and target variable, respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

#mode = 0, 1, 2
default_mode = 0
list_randomForest_mode = 1
hyper_parameters_mode =2
mode = default_mode


select_model(X_train, X_test, y_train, y_test, mode)

