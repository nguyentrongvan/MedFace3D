import numpy as np

def min_max_scale(value_list):
    max_value = max(value_list)
    min_value = min(value_list)
    scaled_list = [(x - min_value) / (max_value - min_value) for x in value_list]
    return np.asarray(scaled_list)