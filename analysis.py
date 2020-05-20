import numpy as np
import pickle
import matplotlib.pyplot as plt

ERROR = 1.984009  # save_7_lr2_cf2
# ERROR = 5.853815  # save_7_lr2_cf1

# ERROR = 20.712499  # save_5_lr2_cf1
# ERROR = 1.611237  # save_5_lr2_cf2

# ERROR = 2.766666  # save_3_lr2_cf1
# ERROR = 19.727727  # save_3_lr2_cf2


def getPredictWithSmoothWindow(test, percentile, n):
    threshold = percentile * ERROR
    pred = np.zeros([len(test), 1])
    for i in range(len(test)):
        if i < n:
            error = np.mean(test[:i+1])
        else:
            error = np.mean(test[i-n+1:i+1])
        if error > threshold:
            pred[i] = 1

    return pred


def getPredict(test, percentile):
    threshold = percentile * ERROR
    pred = np.zeros([len(test), 1])
    for i in range(len(test)):
        if test[i] > threshold:
            pred[i] = 1

    return pred


def computeMetrics(label, pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(label.shape[0]):
        if label[i] == pred[i] and label[i] == 1:
            TP += 1
        elif label[i] == pred[i] and label[i] == 0:
            TN += 1
        elif label[i] != pred[i] and label[i] == 1:
            FN += 1
        else:
            FP += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)

    return precision, recall, F1


def main(percentile):
    test = pickle.load(open("save_7_lr2_cf2/test.pkl", "rb"))
    label_test = pickle.load(open("label_test.pkl", "rb"))

    train2 = pickle.load(open("save_7_lr2_cf2/train2.pkl", "rb"))
    label_train2 = pickle.load(open("label_train2.pkl", "rb"))

    p_train = []
    r_train = []
    f_train = []

    p_test = []
    r_test = []
    f_test = []

    for i in range(percentile.shape[0]):
        # pred_test = getPredict(test, percentile)
        pred_test = getPredictWithSmoothWindow(test, percentile[i], 1)
        p1, r1, f1 = computeMetrics(label_test, pred_test)
        print("test percentile: %.6f, precision: %.6f, recall: %.6f, F1: %.6f" % (percentile[i], p1, r1, f1))
        p_test.append(p1)
        r_test.append(r1)
        f_test.append(f1)

        # pred_train2 = getPredict(train2, percentile)
        pred_train2 = getPredictWithSmoothWindow(train2, percentile[i], 1)
        p2, r2, f2 = computeMetrics(label_train2, pred_train2)
        print("train2 percentile: %.6f, precision: %.6f, recall: %.6f, F1: %.6f" % (percentile[i], p2, r2, f2))
        p_train.append(p2)
        r_train.append(r2)
        f_train.append(f2)

    # return p_test, r_test, f_test, p_train, r_train, f_train
    return pred_test, label_test, pred_train2, label_train2


if __name__ == "__main__":
    percentile = np.array([1.4])

    p1, l1, p2, l2 = main(percentile)

    plt.rcParams["figure.dpi"] = 200
    plt.figure(1)
    plt.plot(p1)
    plt.plot(l1)
    plt.xlabel("Time")
    plt.ylabel("Attack or not")
    plt.title("Window length: 1 hour (on Dataset #3)")
    plt.legend(["Detected status", "Observed status"])

    plt.figure(2)
    plt.plot(p2)
    plt.plot(l2)
    plt.xlabel("Time")
    plt.ylabel("Attack or not")
    plt.title("Window length: 1 hour (on Dataset #2)")
    plt.legend(["Detected status", "Observed status"])

    plt.show()

    # percentile = np.arange(0.5, 2, 0.1)
    #
    # p1, r1, f1, p2, r2, f2 = main(percentile)
    #
    # import h5py
    # with h5py.File("window_smooth_24.h5", 'w') as hf:
    #     hf.create_dataset("p_test", data=np.array(p1))
    #     hf.create_dataset("p_train", data=np.array(p2))
    #     hf.create_dataset("r_test", data=np.array(r1))
    #     hf.create_dataset("r_train", data=np.array(r2))
    #     hf.create_dataset("f_test", data=np.array(f1))
    #     hf.create_dataset("f_train", data=np.array(f2))

    # plt.rcParams['figure.dpi'] = 200
    # plt.figure(1)
    # plt.plot(percentile, p1, linestyle="-.", marker=".")
    # plt.xlabel("percentile")
    # plt.ylabel("precision")
    # plt.title("test precision")
    #
    # plt.figure(2)
    # plt.plot(percentile, r1, linestyle="-.", marker=".")
    # plt.xlabel("percentile")
    # plt.ylabel("recall")
    # plt.title("test recall")
    #
    # plt.figure(3)
    # plt.plot(percentile, f1, linestyle="-.", marker=".")
    # plt.xlabel("percentile")
    # plt.ylabel("F1")
    # plt.title("test F1")
    #
    # plt.figure(4)
    # plt.plot(percentile, p2, linestyle="-.", marker=".")
    # plt.xlabel("percentile")
    # plt.ylabel("precision")
    # plt.title("train precision")
    #
    # plt.figure(5)
    # plt.plot(percentile, r2, linestyle="-.", marker=".")
    # plt.xlabel("percentile")
    # plt.ylabel("recall")
    # plt.title("train recall")
    #
    # plt.figure(6)
    # plt.plot(percentile, f2, linestyle="-.", marker=".")
    # plt.xlabel("percentile")
    # plt.ylabel("F1")
    # plt.title("train F1")
    #
    #
    # plt.show()
