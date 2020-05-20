import tensorflow as tf
import numpy as np
from read_data import *
import os


class AutoEncoder(object):
    def __init__(self, epochs, learning_rate, feature_length, batch_size, savepath):
        self.epochs = epochs
        self.lr = learning_rate
        self.fl = feature_length
        self.bs = batch_size
        self.savepath = savepath

        self.sess = tf.Session()
        self._init_setup()

    def _init_setup(self):
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, self.fl], name="label")
        self.pred = self.Network()

        self.loss = tf.losses.mean_squared_error(self.label, self.pred)

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.saver = tf.train.Saver()

    def Network(self):
        l1 = tf.keras.layers.Dense(units=32, activation="relu", kernel_initializer="he_normal")(self.label)
        l2 = tf.keras.layers.Dense(units=24, activation="relu", kernel_initializer="he_normal")(l1)
        l3 = tf.keras.layers.Dense(units=16, activation="relu", kernel_initializer="he_normal")(l2)
        l4 = tf.keras.layers.Dense(units=24, activation="relu", kernel_initializer="he_normal")(l3)
        l5 = tf.keras.layers.Dense(units=32, activation="relu", kernel_initializer="he_normal")(l4)
        output = tf.keras.layers.Dense(units=self.fl, activation="relu", kernel_initializer="he_normal")(l5)

        return output

    def train(self, file):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if self.load():
            print("[*] Load Successful")
        else:
            print("[*] Load Failed")

        data = read_train_data(file)
        size = data.shape

        # index = np.arange(size[0])
        # np.random.shuffle(index)
        # data = data[index, :]

        training_length = int(size[0] / 3 * 2)

        training_data = data[:training_length, :]
        validating_data = data[training_length:, :]

        nbatches = int(training_length / self.bs)

        ave_train_loss = []
        ave_valid_loss = []
        for i in range(self.epochs):
            train_losses = 0
            valid_losses = 0
            for j in range(nbatches):
                batch_data = training_data[j*self.bs: j*self.bs+self.bs, :]

                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict={self.label: batch_data})

                train_losses += train_loss

            ave_train_loss.append(train_losses / nbatches)

            for j in range(validating_data.shape[0]):
                batch_data = validating_data[j, :]
                batch_data = batch_data[np.newaxis, :]
                valid_loss = self.sess.run(self.loss, feed_dict={self.label: batch_data})

                valid_losses += valid_loss

            ave_valid_loss.append(valid_losses / validating_data.shape[0])

            print("Epochs: %d, Training loss: %.6f, Validating loss: %.6f" % (i, ave_train_loss[-1], ave_valid_loss[-1]))

            self.save(i)

        import h5py
        with h5py.File("result.h5", 'w') as hf:
            hf.create_dataset("train_loss", data=ave_train_loss)
            hf.create_dataset("valid_loss", data=ave_valid_loss)

        return ave_train_loss, ave_valid_loss

    def test(self, file):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if self.load():
            print("[*] Load Successful")
        else:
            print("[*] Load Failed")

        data = read_test_data(file)

        ave_test_loss = []

        for j in range(data.shape[0]):
            batch_data = data[j, :]
            batch_data = batch_data[np.newaxis, :]
            test_loss = self.sess.run(self.loss, feed_dict={self.label: batch_data})

            ave_test_loss.append(test_loss)

        # import h5py
        # with h5py.File("test_result.h5", 'w') as hf:
        #     hf.create_dataset("train2", ave_test_loss)

        import pickle
        if file == "BATADAL_test_dataset.csv":
            pickle.dump(ave_test_loss, open('test.pkl', 'wb'))
        else:
            pickle.dump(ave_test_loss, open('train2.pkl', 'wb'))

    def detect(self, file):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if self.load():
            print("[*] Load Successful")
        else:
            print("[*] Load Failed")

        data, length = read_detect_data(file)

        pred_series = np.zeros_like(data)

        for j in range(data.shape[0]):
            batch_data = data[j, :]
            batch_data = batch_data[np.newaxis, :]
            pred = self.sess.run(self.pred, feed_dict={self.label: batch_data})

            pred_series[j, :] = pred

        # import pickle
        # pickle.dump(data, open("detect_label.pkl", "wb"))
        # pickle.dump(pred_series, open("detect_pred.pkl", "wb"))

        return data, pred_series, length

    def load(self):
        print("[*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.savepath)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.savepath, ckpt_name))
            return True
        else:
            return False

    def save(self, step):
        model_name = "autoEncoder.model"

        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.saver.save(self.sess, os.path.join(self.savepath, model_name), global_step=step)


if __name__ == "__main__":
    filename = "BATADAL_trainingset1.csv"
    autoEncoder = AutoEncoder(epochs=200, learning_rate=1e-2, feature_length=43, batch_size=300, savepath="save_7_lr2_cf2")
    tl, vl = autoEncoder.train(filename)
    print("Done!")

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(tl)
    plt.plot(vl)
    plt.show()
