import plot
import file
import numpy as np
import data
import FCM
import RBF

number_of_clusters = 10
clusters_radius = 0.9

def func1() :
    raw_data = file.readXLSX('data/5clstrain1500.xlsx')
    train_data, test_data = data.get_dataset(data=raw_data, need_shuffling=True, train_count=0.7 * len(raw_data))
    labels = data.extract_labels(train_data)
    train_data = data.replace_labels(train_data, labels)
    train_data, cols_range = data.normalization(train_data, ignore_cols=[2])
    u_train, V = FCM.FCM(train_data[:, 0:2], c=number_of_clusters, m= 3)
    train_clusters = []
    for i in range(len(train_data)):
        temp = [train_data[i][0], train_data[i][1], np.argmax(u_train[i])]
        train_clusters.append(temp)
    test_data, test_data_cols_range = data.normalization(test_data, ignore_cols=[2])
    test_data = data.replace_labels(test_data, labels)
    W = RBF.RBF(clusters_radius, u_train, V, train_data, 3.0, 5)
    u_test = FCM.calcUik(test_data[:, 0:2], V, 3)
    Yhat = RBF.test(clusters_radius, u_test, V, test_data[:, 0:2], 3.0, W)
    test_data_correct = []
    test_data_incorrect = []
    efficiency = 0
    for i in range(len(Yhat)):
        if Yhat[i] != test_data[i, 2]:
            test_data_incorrect.append(test_data[i, 0:2])
        else:
            test_data_correct.append(test_data[i, 0:2])
            efficiency += 1
    print('efficiency is ' + str((efficiency / len(test_data))))
    test_data_correct = data.labeling(test_data_correct, 0.0)
    test_data_incorrect = data.labeling(test_data_incorrect, 1.0)
    result_data = test_data_correct
    result_data = np.append(result_data, test_data_incorrect, axis=0)
    test_data = data.denormalize(result_data, test_data_cols_range, ignore_cols=[2])
    plot.plot_input_data(test_data, mode='result_mode')
    return None

# create files
def func2():
    raw_data = file.readXLSX('data/5clstrain1500.xlsx')
    train_data, test_data = data.get_dataset(data=raw_data, need_shuffling=True, train_count=0.7 * len(raw_data))
    labels = data.extract_labels(train_data)
    labels_indices = list(range(0, len(labels)))
    train_data = data.replace_labels(train_data, labels)
    test_data = data.replace_labels(test_data, labels)
    train_data_norm, cols_range = data.normalization(train_data, ignore_cols=[2])
    clusters = list(range(0, number_of_clusters))
    u_train, V = FCM.FCM(train_data_norm[:, 0:2], c=clusters, m=3)
    train_data = data.denormalize(train_data, cols_range, ignore_cols=[2])
    labels_csv = np.array([labels_indices, labels]).transpose()
    file.writeCSV('data/sample_label.csv', labels_csv)
    file.writeCSV('data/sample_train.csv', train_data)
    file.writeCSV('data/sample_test.csv', test_data)
    file.writeCSV('data/sample_V.csv', V)
    return V

# clusters bounds
def fun3():
    V = file.readCSV('data/sample_V.csv')
    V = data.get_dataset(data=V)
    all_data = file.readXLSX('data/5clstrain1500.xlsx')
    all_data = data.get_dataset(data=all_data)
    cols_range = data.cols_range(all_data)
    # generated_data = np.mgrid[cols_range[0][0]:cols_range[0][1]:0.1, cols_range[1][0]:cols_range[1][1]:0.1].reshape(2, -1).T
    generated_data = np.mgrid[0:1:0.01,0:1:0.01].reshape(2, -1).T
    u_generated = FCM.calcUik(generated_data, V, 3.0)
    u_generated = np.argmax(u_generated, axis=1)
    u = data.labeling(generated_data, u_generated.tolist())
    V_label = len(V)
    V = data.labeling(V, V_label)
    u = np.append(u, V, axis=0)
    u = data.denormalize(u, cols_range, ignore_cols=[2])
    plot.plot_input_data(u, 'not_result_mode')
    return u_generated

def fun4():
    V = file.readCSV('data/sample_V.csv')
    V = data.get_dataset(data=V)
    train_data = file.readCSV('data/sample_train.csv')
    train_data = data.get_dataset(data=train_data)
    test_data = file.readCSV('data/sample_test.csv')
    test_data = data.get_dataset(data=test_data)
    labels = file.readCSV('data/sample_label.csv')
    labels = data.get_dataset(data=labels)
    labels = labels[:, 1]
    train_data, cols_range = data.normalization(train_data, ignore_cols=[2])
    u_train = FCM.calcUik(train_data[:, 0:2], V, 3.0)
    # u_train, V = FCM.FCM(train_data[:, 0:2], c=labels, m=3)
    test_data, test_data_cols_range = data.normalization(test_data, ignore_cols=[2])
    # test_data = data.replace_labels(test_data, labels)
    W = RBF.RBF(clusters_radius, u_train, V, train_data, 3.0, 5)
    u_test = FCM.calcUik(test_data[:, 0:2], V, 3)
    Yhat = RBF.test(clusters_radius, u_test, V, test_data[:, 0:2], 3.0, W)
    test_data_correct = []
    test_data_incorrect = []
    efficiency = 0
    for i in range(len(Yhat)):
        if Yhat[i] != test_data[i, 2]:
            test_data_incorrect.append(test_data[i, 0:2])
        else:
            test_data_correct.append(test_data[i, 0:2])
            efficiency += 1
    print('efficiency is ' + str((efficiency / len(test_data))))
    test_data_correct = data.labeling(test_data_correct, 0.0)
    test_data_incorrect = data.labeling(test_data_incorrect, 1.0)
    result_data = test_data_correct
    result_data = np.append(result_data, test_data_incorrect, axis=0)
    test_data = data.denormalize(result_data, test_data_cols_range, ignore_cols=[2])
    plot.plot_input_data(test_data, mode='result_mode')
    return None

def fun5():
    V = file.readCSV('data/sample_V.csv')
    V = data.get_dataset(data=V)
    all_data = file.readXLSX('data/5clstrain1500.xlsx')
    all_data = data.get_dataset(data=all_data)
    cols_range = data.cols_range(all_data)
    train_data = file.readCSV('data/sample_train.csv')
    train_data = data.get_dataset(data=train_data)
    train_data = train_data[:, 0:2]
    train_data, train_data_cols_range = data.normalization(train_data)
    u_test = FCM.calcUik(train_data[:, 0:2], V, 3.0)
    u_test = np.argmax(u_test, axis=1)
    u = data.labeling(train_data, u_test.tolist())
    V_label = len(V)
    V = data.labeling(V, V_label)
    u = np.append(u, V, axis=0)
    u = data.denormalize(u, cols_range, ignore_cols=[2])
    plot.plot_input_data(u, 'not_result_mode')
    return u_test

fun4()