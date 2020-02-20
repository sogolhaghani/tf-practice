import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score).any() > threshold:
            outliers.append(y)
    return outliers

#  ./data/Data2Learn/Merged_B50Data20Sep2017140730.csv
label_col = 10
path_url = './data/Data2Learn/*.csv'
filenames = glob.glob(path_url)
dataframes = []
for f in filenames: 
    x = pd.read_csv(f, header=None, skiprows=1).to_numpy()
    ss = np.full((x.shape[0], 1), f[26:f.index("Data", 26)],dtype=np.int)
    x = np.append(ss,x, axis=1)
    dataframes.append(x)
arr = np.vstack(dataframes)

arr = np.delete(arr, 1, 1)
arr = np.delete(arr, 1, 1)
arr = np.delete(arr, 11, 1)
arr = np.delete(arr, 11, 1)
arr = np.delete(arr, 11, 1)
# arr = np.delete(arr, 8, 1)

# check for missing value. It was empty.
# print(np.argwhere(pd.isnull(arr)))

# check for outlier point. It was empty.
# outlier_datapoints = detect_outlier(arr[:,0:9])
# print(outlier_datapoints)


X = scale( arr, axis=0, with_mean=True, with_std=True )
X[:,label_col] = arr[:,label_col]

_labels = set( arr[:,label_col])
_labels_statistic = []
for id, val in enumerate(_labels): 
    _labels_statistic.append([id, val , len(X[X == val])])
    X[X == val] = id
    
print(_labels_statistic)
np.savetxt("./data/preprocessed/refine4.csv", X, delimiter=",")
_labels_statistic = np.array(_labels_statistic)
plt.bar(_labels_statistic[:,0], _labels_statistic[:,2])
plt.show()