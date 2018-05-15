from dataproc.dataproc import getAEData
import dataproc.datadb as db
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model.keras.AE import LstmAE
import numpy as np

import keras
from keras.models import Model
from keras.layers import InputLayer, Input
from keras.optimizers import Adam

train_rate = 0.9
window_size = 5

x, y = getAEData(db.stockFilename[0], window_size)
# print(y.shape)
# fig, axes = plt.subplots(2, 2)
# axes[0][0].plot(x[6])
# axes[0][1].plot(x[36])
# axes[1][0].plot(x[66])
# axes[1][1].plot(x[96])
# plt.show()

train_num = int(len(y) * train_rate)
train_x, train_y, test_x, test_y = x[:train_num], y[:train_num], x[train_num:], y[train_num:]
ae_model, encoder_model, decoder_model = None, None, None

def lstm_ae(is_train=False):
    lstm_ae_weight_path = "./saved_model/lstm_ae_weight.h5"
    lstm_size = 60
    lstm_layer = 2
    global ae_model, encoder_model, decoder_model
    ae_model, encoder_model, decoder_model = LstmAE.getLstmAEModel((window_size, 1), lstm_size, lstm_layer)
    #ae_model, encoder_model = LstmAE.getFCNAEModel((window_size, 1), [], 4)

    ae_model.compile(optimizer="adam", loss="mse")
    # train
    if is_train:
        ae_model.fit(train_x, train_y, batch_size=100, epochs=150, shuffle=True, verbose=2, validation_data=(test_x, test_y))
        ae_model.save_weights(lstm_ae_weight_path)
    #load
    else:
        ae_model.load_weights(lstm_ae_weight_path)
        encoder_model.set_weights(ae_model.get_weights()[0:3 * (lstm_layer // 2)])
        ae_model.summary()
        encoder_model.summary()
        decoder_model.summary()
        for i in range(len(ae_model.get_weights())):
            print(ae_model.get_weights()[i].shape)
        decoder_model.set_weights(ae_model.get_weights()[3 * (lstm_layer // 2):])

def fcn_ae(is_train=False):
    fcn_ae_weight_path = "./saved_model/fcn_ae_weight.h5"
    layer_sizes = [128, window_size]
    fcn_layer_num = 2
    global ae_model, encoder_model, decoder_model
    ae_model, encoder_model, decoder_model = LstmAE.getFCNAEModel((window_size, ), layer_sizes, fcn_layer_num)

    optimizer = Adam(lr=0.0001)
    ae_model.compile(optimizer=optimizer, loss="mse")
    # train
    if is_train:
        ae_model.fit(train_x, train_y, batch_size=100, epochs=200, shuffle=True, verbose=2, validation_data=(test_x, test_y))
        ae_model.save_weights(fcn_ae_weight_path)
    # load
    else:
        ae_model.load_weights(fcn_ae_weight_path)
        encoder_model.set_weights(ae_model.get_weights()[0:2*(fcn_layer_num//2)])
        ae_model.summary()
        encoder_model.summary()
        decoder_model.summary()
        for i in range(len(ae_model.get_weights())):
            print(ae_model.get_weights()[i].shape)
        decoder_model.set_weights(ae_model.get_weights()[2*(fcn_layer_num//2):])

train_x, train_y, test_x, test_y = np.squeeze(train_x), np.squeeze(train_y), np.squeeze(test_x), np.squeeze(test_y)
fcn_ae()
# lstm_ae(False)

train_embedding = encoder_model.predict(train_x)
print(train_embedding.shape)
# fig, axes = plt.subplots(2, 2)
# axes[0][0].plot(train_x[6])
# axes[0][1].plot(train_embedding[6].T)
# axes[1][0].plot(train_x[66])
# axes[1][1].plot(train_embedding[66].T)
# print(train_embedding[6]-train_embedding[66])
# plt.show()

# visualizing embedding data
from sklearn.manifold import TSNE

train_tsne_embedding = TSNE(n_components=2, random_state=66).fit_transform(train_embedding)
# train_tsne_embedding = train_embedding

# cluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# train_embedding = PCA(n_components=10).fit_transform(train_embedding)
# print(train_embedding.shape)
num_per_class=[]
num_class = 3

# DBSCAN
# train_pred_class = DBSCAN(eps=0.1, min_samples=10).fit_predict(train_tsne_embedding)
# num_class = max(train_pred_class) + 2
# print(num_class)
# for i in range(-1, num_class-1):
#     print(i, sum(train_pred_class==i))
#     num_per_class.append(sum(train_pred_class==i))




cluster_model = KMeans(n_clusters=num_class, )
cluster_model.fit(train_tsne_embedding)
train_pred_class = cluster_model.predict(train_tsne_embedding)

for i in range(num_class):
    num_per_class.append(sum(train_pred_class==i))
    print(i, sum(train_pred_class==i))

# visualizing embedding points
plt.figure()
plt.scatter(train_tsne_embedding[:, 0], train_tsne_embedding[:, 1], c=train_pred_class)
# fig = plt.figure()
# ax=fig.add_subplot(111, projection="3d")
# ax.scatter(train_tsne_embedding[:, 0], train_tsne_embedding[:, 1],train_tsne_embedding[:, 2], c=train_pred_class)

# visualizing samples of different class
w, h = 10, 10
for k in range(num_class):
    fig = plt.figure("class_{}".format(k))
    axes = fig.subplots(w, h)
    samples = train_x[train_pred_class==k]
    for i in range(w):
        for j in range(h):
            if i*w+j >= num_per_class[k]: break
            # if len(samples)>3*num_samples: axes[i][k].plot(samples[np.random.randint(0, len(samples))])
            else: axes[i][j].plot(samples[(i*w+j)%len(samples)])

# visualizing cluster center
fig2, axes2 = plt.subplots(2, num_class)
for k in range(num_class):
    train_xcenter = np.mean(train_x[train_pred_class==k], 0)
    emcenter=np.mean(train_embedding[train_pred_class==k], 0)[np.newaxis,:]
    orcenter=decoder_model.predict(emcenter)[0]
    axes2[1][k].plot(orcenter)
    axes2[0][k].plot(train_xcenter)

# plt.show()
plt.show()