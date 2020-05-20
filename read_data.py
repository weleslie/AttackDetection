import numpy as np
import pandas as pd


def read_train_data(file):
    train1 = pd.read_csv(file)
    train1_np = np.array(train1)
    train1_np = train1_np[:, 1:-1]

    return train1_np


def read_detect_data(file):
    attack = ["16/01/17 00", 96]
    true_attack = ["16/01/17 09", 70]
    test = pd.read_csv(file)
    time = np.array(test["DATETIME"])
    test = np.array(test)
    test = test[:, 1:]

    length = np.zeros([time.shape[0], 1])

    for i in range(time.shape[0]):
        if time[i] == true_attack[0]:
            length[i:i + true_attack[1]] = 0.1
            break

    for i in range(time.shape[0]):
        if time[i] == attack[0]:
            data = test[i:i+attack[1], :]
            length = length[i:i+attack[1], :]
            break

    return data, length


def read_test_data(file):
    # if file == "BATADAL_trainingset2.csv":
    #     attack = [["13/09/16 23", 50], ["26/09/16 11", 24], ["09/10/16 09", 60],
    #               ["29/10/16 19", 94], ["26/11/16 17", 60], ["06/12/16 07", 94],
    #               ["14/12/16 15", 110]]
    #
    #     label_path = "label_train2.pkl"
    # else:
    #     attack = [["16/01/17 09", 70], ["30/01/17 08", 65], ["09/02/17 03", 31],
    #               ["12/02/17 01", 31], ["24/02/17 05", 100], ["10/03/17 14", 80],
    #               ["25/03/17 20", 30]]
    #
    #     label_path = "label_test.pkl"

    test = pd.read_csv(file)
    # time = np.array(test["DATETIME"])

    # length = np.zeros([time.shape[0], 1])
    # for i in range(time.shape[0]):
    #     for j in range(7):
    #         if time[i] == attack[j][0]:
    #             length[i:i+attack[j][1]] = 1

    # import pickle
    # pickle.dump(length, open(label_path, "wb"))

    test_np = np.array(test)

    if file == "BATADAL_trainingset2.csv":
        test_np = test_np[:, 1:-1]
    else:
        test_np = test_np[:, 1:]

    return test_np


if __name__ == "__main__":
    data = read_detect_data("BATADAL_test_dataset.csv")
    size = data.shape

