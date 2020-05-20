from AutoEncoder import AutoEncoder
import numpy as np
import matplotlib.pyplot as plt


filename = "BATADAL_test_dataset.csv"
autoEncoder = AutoEncoder(epochs=200, learning_rate=1e-2, feature_length=43, batch_size=300, savepath="save_7_lr2_cf2")
label, pred, length = autoEncoder.detect(filename)

index = [2, 13, 15, 14, 16, 34, 33]

label = label[:, index]
pred = pred[:, index]

error = np.square(label - pred)
nor_error = error / (np.sum(error, axis=0) + 1e-6)
ave_error = np.mean(nor_error, axis=1)

plt.figure(1)
plt.plot(length)
plt.plot(ave_error)
plt.plot(nor_error[:, 0])
plt.xlabel("Time")
plt.ylabel("Reconstruction error")
plt.title("L_T3")
plt.legend(["Observed status", "average error", "L_T3 error"], loc=1)

plt.figure(2)
plt.plot(length)
plt.plot(ave_error)
plt.plot(nor_error[:, 1])
plt.xlabel("Time")
plt.ylabel("Reconstruction error")
plt.title("F_PU4")
plt.legend(["Observed status", "average error", "F_PU4 error"], loc=1)

plt.figure(3)
plt.plot(length)
plt.plot(ave_error)
plt.plot(nor_error[:, 2])
plt.xlabel("Time")
plt.ylabel("Reconstruction error")
plt.title("F_PU5")
plt.legend(["Observed status", "average error", "F_PU5 error"], loc=1)

plt.figure(4)
plt.plot(length)
plt.plot(ave_error)
plt.plot(nor_error[:, 3])
plt.xlabel("Time")
plt.ylabel("Reconstruction error")
plt.title("S_PU4")
plt.legend(["Observed status", "average error", "S_PU4 error"], loc=1)

plt.figure(5)
plt.plot(length)
plt.plot(ave_error)
plt.plot(nor_error[:, 4])
plt.xlabel("Time")
plt.ylabel("Reconstruction error")
plt.title("S_PU5")
plt.legend(["Observed status", "average error", "S_PU5 error"], loc=1)

plt.figure(6)
plt.plot(length)
plt.plot(ave_error)
plt.plot(nor_error[:, 5])
plt.xlabel("Time")
plt.ylabel("Reconstruction error")
plt.title("P_J256")
plt.legend(["Observed status", "average error", "P_J256 error"], loc=1)

plt.figure(7)
plt.plot(length)
plt.plot(ave_error)
plt.plot(nor_error[:, 6])
plt.xlabel("Time")
plt.ylabel("Reconstruction error")
plt.title("P_J300")
plt.legend(["Observed status", "average error", "P_J300 error"], loc=1)

plt.show()
