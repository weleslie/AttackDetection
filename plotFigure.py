import matplotlib.pyplot as plt
import h5py
import numpy as np

percentile = np.arange(0.5, 2, 0.1)


def readh5Data(filename):
    with h5py.File(filename, "r") as hf:
        p1 = hf["p_test"][()]
        r1 = hf["r_test"][()]
        f1 = hf["f_test"][()]

        p2 = hf["p_train"][()]
        r2 = hf["r_train"][()]
        f2 = hf["f_train"][()]

    return p1, r1, f1, p2, r2, f2


def plotFig4_5(filename):
    p1, r1, f1, p2, r2, f2 = readh5Data(filename)

    plt.rcParams["figure.dpi"] = 300
    plt.figure(1)
    plt.plot(percentile, p1, linestyle="-.", marker=".")
    plt.plot(percentile, r1, linestyle="--", marker="o")
    plt.plot(percentile, f1, linestyle="-.", marker="x")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Average score on Dataset #3")
    plt.grid()
    plt.legend(["Precision", "Recall", "F1 Score"])

    plt.figure(2)
    plt.plot(percentile, p2, linestyle="-.", marker=".")
    plt.plot(percentile, r2, linestyle="--", marker="o")
    plt.plot(percentile, f2, linestyle="-.", marker="x")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Average score on Dataset #2")
    plt.grid()
    plt.legend(["Precision", "Recall", "F1 Score"])

    plt.figure(3)
    plt.plot(percentile, f2, linestyle="-.", marker=".")
    plt.plot(percentile, f1, linestyle="--", marker="o")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Average F1 score")
    plt.grid()
    plt.legend(["Dataset #2", "Dataset #3"])

    plt.show()


def plotFig6(filename1, filename2, filename3):
    p11, r11, f11, p21, r21, f21 = readh5Data(filename1)
    p12, r12, f12, p22, r22, f22 = readh5Data(filename2)
    p13, r13, f13, p23, r23, f23 = readh5Data(filename3)

    plt.rcParams["figure.dpi"] = 300
    plt.figure(1)
    plt.plot(percentile, p11, linestyle="-.", marker=".")
    plt.plot(percentile, p12, linestyle="--", marker="o")
    plt.plot(percentile, p13, linestyle="-.", marker="x")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Precision score on Dataset #3")
    plt.grid()
    plt.legend(["nl=1", "nl=3", "nl=5"])

    plt.figure(2)
    plt.plot(percentile, p21, linestyle="-.", marker=".")
    plt.plot(percentile, p22, linestyle="--", marker="o")
    plt.plot(percentile, p23, linestyle="-.", marker="x")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Precision score on Dataset #2")
    plt.grid()
    plt.legend(["nl=1", "nl=3", "nl=5"])

    plt.figure(3)
    plt.plot(percentile, r11, linestyle="-.", marker=".")
    plt.plot(percentile, r12, linestyle="--", marker="o")
    plt.plot(percentile, r13, linestyle="-.", marker="x")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Recall score on Dataset #3")
    plt.grid()
    plt.legend(["nl=1", "nl=3", "nl=5"])

    plt.figure(4)
    plt.plot(percentile, r21, linestyle="-.", marker=".")
    plt.plot(percentile, r22, linestyle="--", marker="o")
    plt.plot(percentile, r23, linestyle="-.", marker="x")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Recall score on Dataset #2")
    plt.grid()
    plt.legend(["nl=1", "nl=3", "nl=5"])

    plt.figure(5)
    plt.plot(percentile, f11, linestyle="-.", marker=".")
    plt.plot(percentile, f12, linestyle="--", marker="o")
    plt.plot(percentile, f13, linestyle="-.", marker="x")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("F1 score on Dataset #3")
    plt.grid()
    plt.legend(["nl=1", "nl=3", "nl=5"])

    plt.figure(6)
    plt.plot(percentile, f21, linestyle="-.", marker=".")
    plt.plot(percentile, f22, linestyle="--", marker="o")
    plt.plot(percentile, f23, linestyle="-.", marker="x")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("F1 score on Dataset #2")
    plt.grid()
    plt.legend(["nl=1", "nl=3", "nl=5"])

    plt.show()


def plotFig7(filename1, filename2):
    p11, r11, f11, p21, r21, f21 = readh5Data(filename1)
    p12, r12, f12, p22, r22, f22 = readh5Data(filename2)

    plt.rcParams["figure.dpi"] = 300
    plt.figure(1)
    plt.plot(percentile, p11, linestyle="-.", marker=".")
    plt.plot(percentile, p12, linestyle="--", marker="o")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Precision score on Dataset #3")
    plt.grid()
    plt.legend(["cf=1", "cf=2"])

    plt.figure(2)
    plt.plot(percentile, p21, linestyle="-.", marker=".")
    plt.plot(percentile, p22, linestyle="--", marker="o")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Precision score on Dataset #2")
    plt.grid()
    plt.legend(["cf=1", "cf=2"])

    plt.figure(3)
    plt.plot(percentile, r11, linestyle="-.", marker=".")
    plt.plot(percentile, r12, linestyle="--", marker="o")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Recall score on Dataset #3")
    plt.grid()
    plt.legend(["cf=1", "cf=2"])

    plt.figure(4)
    plt.plot(percentile, r21, linestyle="-.", marker=".")
    plt.plot(percentile, r22, linestyle="--", marker="o")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Recall score on Dataset #2")
    plt.grid()
    plt.legend(["cf=1", "cf=2"])

    plt.figure(5)
    plt.plot(percentile, f11, linestyle="-.", marker=".")
    plt.plot(percentile, f12, linestyle="--", marker="o")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("F1 score on Dataset #3")
    plt.grid()
    plt.legend(["cf=1", "cf=2"])

    plt.figure(6)
    plt.plot(percentile, f21, linestyle="-.", marker=".")
    plt.plot(percentile, f22, linestyle="--", marker="o")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("F1 score on Dataset #2")
    plt.grid()
    plt.legend(["cf=1", "cf=2"])

    plt.show()


def plotFig10(filename1, filename2, filename3, filename4, filename5):
    p11, r11, f11, p21, r21, f21 = readh5Data(filename1)
    p12, r12, f12, p22, r22, f22 = readh5Data(filename2)
    p13, r13, f13, p23, r23, f23 = readh5Data(filename3)
    p14, r14, f14, p24, r24, f24 = readh5Data(filename4)
    p15, r15, f15, p25, r25, f25 = readh5Data(filename5)

    plt.rcParams["figure.dpi"] = 300
    plt.figure(1)
    plt.plot(percentile, p11, linestyle="-.", marker=".")
    plt.plot(percentile, p12, linestyle="--", marker="o")
    plt.plot(percentile, p13, linestyle="-.", marker="x")
    plt.plot(percentile, p14, linestyle="--", marker="*")
    plt.plot(percentile, p15, linestyle=":", marker="+")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Precision score on Dataset #3")
    plt.grid()
    plt.legend(["1 hour", "3 hours", "6 hours", "12 hours", "24 hours"])

    plt.figure(2)
    plt.plot(percentile, p21, linestyle="-.", marker=".")
    plt.plot(percentile, p22, linestyle="--", marker="o")
    plt.plot(percentile, p23, linestyle="-.", marker="x")
    plt.plot(percentile, p24, linestyle="--", marker="*")
    plt.plot(percentile, p25, linestyle=":", marker="+")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Precision score on Dataset #2")
    plt.grid()
    plt.legend(["1 hour", "3 hours", "6 hours", "12 hours", "24 hours"])

    plt.figure(3)
    plt.plot(percentile, r11, linestyle="-.", marker=".")
    plt.plot(percentile, r12, linestyle="--", marker="o")
    plt.plot(percentile, r13, linestyle="-.", marker="x")
    plt.plot(percentile, r14, linestyle="--", marker="*")
    plt.plot(percentile, r15, linestyle=":", marker="+")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Recall score on Dataset #3")
    plt.grid()
    plt.legend(["1 hour", "3 hours", "6 hours", "12 hours", "24 hours"])

    plt.figure(4)
    plt.plot(percentile, r21, linestyle="-.", marker=".")
    plt.plot(percentile, r22, linestyle="--", marker="o")
    plt.plot(percentile, r23, linestyle="-.", marker="x")
    plt.plot(percentile, r24, linestyle="--", marker="*")
    plt.plot(percentile, r25, linestyle=":", marker="+")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("Recall score on Dataset #2")
    plt.grid()
    plt.legend(["cf=1", "cf=2"])
    plt.legend(["1 hour", "3 hours", "6 hours", "12 hours", "24 hours"])

    plt.figure(5)
    plt.plot(percentile, f11, linestyle="-.", marker=".")
    plt.plot(percentile, f12, linestyle="--", marker="o")
    plt.plot(percentile, f13, linestyle="-.", marker="x")
    plt.plot(percentile, f14, linestyle="--", marker="*")
    plt.plot(percentile, f15, linestyle=":", marker="+")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("F1 score on Dataset #3")
    plt.grid()
    plt.legend(["1 hour", "3 hours", "6 hours", "12 hours", "24 hours"])

    plt.figure(6)
    plt.plot(percentile, f21, linestyle="-.", marker=".")
    plt.plot(percentile, f22, linestyle="--", marker="o")
    plt.plot(percentile, f23, linestyle="-.", marker="x")
    plt.plot(percentile, f24, linestyle="--", marker="*")
    plt.plot(percentile, f25, linestyle=":", marker="+")
    plt.xlabel("Percentile of reconstruction error on validation dataset")
    plt.ylabel("F1 score on Dataset #2")
    plt.grid()
    plt.legend(["1 hour", "3 hours", "6 hours", "12 hours", "24 hours"])

    plt.show()


if __name__ == "__main__":
    filename1 = "3_1.h5"
    filename2 = "3_2.h5"

    filename3 = "5_1.h5"
    filename4 = "5_2.h5"

    filename5 = "7_1.h5"
    filename6 = "7_2.h5"
    # plotFig4_5(filename6)

    # plotFig6(filename2, filename4, filename6)

    # plotFig7(filename5, filename6)

    w1 = "window_smooth_1.h5"
    w2 = "window_smooth_3.h5"
    w3 = "window_smooth_6.h5"
    w4 = "window_smooth_12.h5"
    w5 = "window_smooth_24.h5"
    plotFig10(w1, w2, w3, w4, w5)
