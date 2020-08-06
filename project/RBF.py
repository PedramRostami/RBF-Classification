import numpy as np

def RBF(cluster_radius, u, V, data, m, number_of_classes) :
    G = np.zeros((len(data), len(V)))
    for i in range(len(data)):
        for j in range(len(V)):
            Vi = np.ones((len(data), 2))
            Vi[:, 0] = np.multiply(Vi[:, 0], V[j][0])
            Vi[:, 1] = np.multiply(Vi[:, 1], V[j][1])
            XmV = np.subtract(data[:, 0:2], Vi)
            XmVT = np.transpose(XmV)
            XmVTotal = np.matmul(XmV, XmVT).diagonal()
            Ui = np.power(u[:, j], m)
            TOP = np.multiply(Ui, XmVTotal)
            Ci = np.sum(TOP) / np.sum(Ui)
            subtraction = np.subtract(data[i, 0:2], V[j, :])
            Gij = np.exp((-1) * cluster_radius * np.matmul(np.transpose(subtraction), np.multiply((1.0 / Ci), subtraction)))
            G[i, j] = Gij
    print('G matrix calculated for train data')
    Y = np.zeros((len(data), number_of_classes))
    for i in range(len(data)):
        Y[i,(int)(data[i][2])] = 1
    GT = np.transpose(G)
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(GT, G)), GT), Y)
    print('W matrix calculated for train data')
    return W

def test(cluster_radius, u, V, data, m, W):
    G = np.zeros((len(data), len(V)))
    for i in range(len(data)):
        for j in range(len(V)):
            Vi = np.ones((len(data), 2))
            Vi[:, 0] = np.multiply(Vi[:, 0], V[j][0])
            Vi[:, 1] = np.multiply(Vi[:, 1], V[j][1])
            XmV = np.subtract(data[:, 0:2], Vi)
            XmVT = np.transpose(XmV)
            XmVTotal = np.matmul(XmV, XmVT).diagonal()
            Ui = np.power(u[:, j], m)
            TOP = np.multiply(Ui, XmVTotal)
            Ci = np.sum(TOP) / np.sum(Ui)
            subtraction = np.subtract(data[i, 0:2], V[j, :])
            Gij = np.exp(
                (-1) * cluster_radius * np.matmul(np.transpose(subtraction), np.multiply((1.0 / Ci), subtraction)))
            G[i, j] = Gij
    Yhat = np.argmax(np.matmul(G, W), axis=1)
    print('Yhat matrix calculated for train data')
    return Yhat
