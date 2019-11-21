import numpy as np

def calc_mae(a, b):
    """Compute Mean absolute error between the 2 numpy vectors"""
    return np.abs((a - b)).mean()

def print_metrics(data, counts):
    """Compute and print metrics"""

    # Mean Absolute Error
    mae = calc_mae(data['COUNTS'], np.array(counts))
    print("Mean Absolute Error: %.2f" % mae)