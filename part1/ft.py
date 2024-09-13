import numpy as np
from math import sqrt
import sys

def count(series):
    return len(ft_not_nan(series))

def mean(series):
    filtered_series = ft_not_nan(series)
    filtered_series_length = len(filtered_series)
    
    if filtered_series_length == 0:
        return np.nan
    
    return ft_sum(filtered_series) / filtered_series_length

def std(series):
    filtered_series = np.array(ft_not_nan(series))
    n = len(filtered_series)
    
    if n == 0:
        return np.nan
    
    mean_value = mean(series)
    sum = 0
    
    for num in filtered_series:
        sum += (num - mean_value) ** 2
    
    return sqrt(sum / (n - 1))

def min(series):
    filtered_series = np.array(ft_not_nan(series))
    min = sys.maxsize
    
    if len(filtered_series) == 0:
        return np.nan
    
    for num in filtered_series:
        if num < min:
            min = num
    
    return min

def percentille_25(series):
    return ft_percentille(series, 25)

def percentille_50(series):
    return ft_percentille(series, 50)

def percentille_75(series):
    return ft_percentille(series, 75)

def max(series):
    filtered_series = np.array(ft_not_nan(series))
    max = -sys.maxsize - 1
    
    if len(filtered_series) == 0:
        return np.nan
    
    for num in filtered_series:
        if num > max:
            max = num
    
    return max

def ft_not_nan(series):
    return series[series.notna()]

def ft_notna(elem):
    elem == elem

def ft_sum(series):
    filtered_series = np.array(ft_not_nan(series))
    sum = 0.0

    for num in filtered_series:
        sum += num
    
    return sum

def ft_sort(series):
    return np.sort(np.array(ft_not_nan(series)))

def ft_percentille(series, percent):
    filtered_series = np.array(ft_not_nan(series))
    n = len(filtered_series)
    
    if n == 0:
        return np.nan
    
    sorted_filtered_series = ft_sort(series)
    
    index = percent / 100 * (n - 1)
    floor_index = int(index)
    ceil_index = index - floor_index
    
    if (ceil_index == 0):
        return sorted_filtered_series[floor_index]
    
    return sorted_filtered_series[floor_index] + ceil_index * (sorted_filtered_series[floor_index + 1] - sorted_filtered_series[floor_index])