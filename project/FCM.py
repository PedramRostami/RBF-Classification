import numpy as np
import random

def FCM(data, m= None, c= None):
    # initialize u by random
    u = []
    for i in range(len(data)):
        rand_placeholders = []
        for j in range(len(c) - 1):
            rand_placeholders.append(random.random())
        rand_placeholders = np.sort(rand_placeholders)
        ui = []
        for j in range(len(c)):
            if j == 0:
                ui.append(rand_placeholders[0])
            elif j == len(c) - 1:
                ui.append(1 - rand_placeholders[j - 1])
            else:
                ui.append(rand_placeholders[j] - rand_placeholders[j - 1])
        u.append(ui)
    u = np.asarray(u)
    # initialize V
    V = np.zeros((len(c), 2))
    u_difference = 1000
    counter = 0
    while u_difference > 1:
        for i in range(len(V)):
            temp_u = np.power(u[:, i], m)
            V[i] = [np.sum(np.multiply(temp_u, data[:, 0])) / np.sum(temp_u), np.sum(np.multiply(temp_u, data[:, 1])) / np.sum(temp_u)]

        u_next_gen = np.zeros((len(data), len(V)))
        for i in range(len(u)):
            for j in range(len(V)):
                summation = 0
                for k in range(len(V)):
                    subtraction = np.subtract(data[i], V[k])
                    if not (subtraction[0] == 0.0 and subtraction[1] == 0.0):
                        summation += np.power(np.sum(np.power(np.subtract(data[i], V[j]), 2)) / np.sum(np.power(subtraction, 2)), 1.0 / ((float)(m - 1)))
                u_next_gen[i][j] = 1.0 / summation
        u_difference = np.sum(np.absolute(np.add(u, -u_next_gen)))
        u = u_next_gen
        counter += 1
        if counter % 10 == 0:
            print('counter is ' + str(counter) + ' --- u_defference is ' + str(u_difference))
    print('counter is ' + str(counter) + ' --- u_defference is ' + str(u_difference))
    return u, V



def calcUik(data, V, m):
    U = np.zeros((len(data), len(V)))
    for i in range(len(data)):
        for j in range(len(V)):
            summation = 0
            for k in range(len(V)):
                subtraction = np.subtract(data[i], V[k])
                if not (subtraction[0] == 0.0 and subtraction[1] == 0.0):
                    summation += np.power(np.sum(np.power(np.subtract(data[i], V[j]), 2)) / np.sum(np.power(subtraction, 2)),1.0 / ((float)(m - 1)))
            U[i][j] = 1.0 / summation
    return U