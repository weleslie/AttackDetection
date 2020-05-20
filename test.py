from AutoEncoder import AutoEncoder


filename = "BATADAL_test_dataset.csv"
autoEncoder = AutoEncoder(epochs=200, learning_rate=1e-2, feature_length=43, batch_size=300, savepath="save_7_lr2_cf2")
autoEncoder.test(filename)


