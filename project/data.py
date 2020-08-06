import numpy as np
import random

def get_dataset(data, need_shuffling=None, train_count=None):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
    np_array_data = np.asarray(data)
    if need_shuffling == True:
        np.random.shuffle(np_array_data)
    if (not train_count) :
        return np_array_data
    else:
        train_data = np_array_data[:(int)(train_count)]
        test_data = np_array_data[(int)(train_count):]
        return train_data, test_data

def normalization(data, ignore_cols= None):
    data = np.asarray(data)
    cols_range = []
    if len(data.shape) == 2 :
        for i in range(data.shape[1]):
            if ignore_cols:
                if i not in ignore_cols:
                    col_range = (np.amin(data[:, i]), np.amax(data[:, i]))
                    data[:, i] = np.add(data[:, i], -col_range[0])
                    data[:, i] = np.multiply(data[:, i], 1.0 / (col_range[1] - col_range[0]))
                    cols_range.append(col_range)
                else:
                    cols_range.append((np.amin(data[:, i]), np.amax(data[:, i])))
            else:
                col_range = (np.amin(data[:, i]), np.amax(data[:, i]))
                data[:, i] = np.add(data[:, i], -col_range[0])
                data[:, i] = np.multiply(data[:, i], 1.0 / (col_range[1] - col_range[0]))
                cols_range.append(col_range)
    return data, cols_range

def denormalize(data, cols_range, ignore_cols=None):
    data = np.asarray(data)
    if len(data.shape) == 2:
        for i in range(data.shape[1]):
            if ignore_cols:
                if i not in ignore_cols:
                    data[:, i] = np.multiply(data[:, i], cols_range[i][1] - cols_range[i][0])
                    data[:, i] = np.add(data[:, i], cols_range[i][0])
            else:
                data[:, i] = np.multiply(data[:, i], cols_range[i][1] - cols_range[i][0])
                data[:, i] = np.add(data[:, i], cols_range[i][0])
    return data

def labeling(data, label):
    if not isinstance(label, list):
        res_data = []
        data = np.asarray(data)
        for i in range(len(data)):
            res_data.append(np.append(data[i, :], label).tolist())
        return np.asarray(res_data)
    else:
        res_data = []
        data = np.asarray(data)
        for i in range(len(data)):
            res_data.append(np.append(data[i, :], label[i]).tolist())
        return np.asarray(res_data)
def extract_labels(data) :
    labels = []
    for i in range(len(data)):
        if data[i, 2] not in labels:
            labels.append(data[i, 2])
    return labels


def replace_labels(data, labels):
    for i in range(len(data)):
        data[i, 2] = np.where(labels == data[i, 2])[0]
    return data

def cols_range(data):
    data = np.array(data)
    cols_range = []
    if len(data.shape) == 2:
        for i in range(data.shape[1]):
            col_range = (np.amin(data[:, i]), np.amax(data[:, i]))
            cols_range.append(col_range)
    return cols_range