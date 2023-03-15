import numpy as np
import time
from .feature_extraction import*
from .extraction_of_features import*

path_both_legs = "local_path_to_both_legs"
path_right = "local_path_to_right_leg"
path_left = "local_path_to_left_leg"
path_none = "local_path_to_none_legs"

print("------------------------------")
start = time.time()
features_both_legs = extract_feature_class(path_both_legs)
features_both_legs_reshaped = np.column_stack([features_both_legs[0], *features_both_legs[1:]])
minutes, remaining_seconds = divmod(time.time() - start, 60)
print("Extracting features from both legs finished and took {} minutes and {} seconds".format(minutes, remaining_seconds))

print("------------------------------")
start = time.time()
features_none_legs = extract_feature_class(path_none)
features_none_legs_reshaped = np.column_stack([features_none_legs[0], *features_none_legs[1:]])
minutes, remaining_seconds = divmod(time.time() - start, 60)
print("Extracting features from none legs finished and took {} minutes and {} seconds".format(minutes, remaining_seconds))

print("------------------------------")
start = time.time()
features_right_leg = extract_feature_class(path_right)
features_right_leg_reshaped = np.column_stack([features_right_leg[0], *features_right_leg[1:]])
minutes, remaining_seconds = divmod(time.time() - start, 60)
print("Extracting features from right leg finished and took {} minutes and {} seconds".format(minutes, remaining_seconds))

print("------------------------------")
start = time.time()
features_left_leg = extract_feature_class(path_left)
features_left_leg_reshaped = np.column_stack([features_left_leg[0], *features_left_leg[1:]])
minutes, remaining_seconds = divmod(time.time() - start, 60)
print("Extracting features from left leg finished and took {} minutes and {} seconds".format(minutes, remaining_seconds))

data = np.vstack((features_both_legs_reshaped, features_none_legs_reshaped, features_right_leg_reshaped, features_left_leg_reshaped))

labels = np.concatenate((np.zeros(features_both_legs_reshaped.shape[0]), np.ones(features_none_legs_reshaped.shape[0]), 2*np.ones(features_right_leg_reshaped.shape[0]), 3*np.ones(features_left_leg_reshaped.shape[0])))
print(labels.shape)
print(labels)

np.save("local_path_data", data)
np.save("local_path_labels", labels)