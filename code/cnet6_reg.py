from cnet6_imports import *
from lasagne import layers
from lasagne.updates import nesterov_momentum, sgd, momentum, adagrad
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import softmax
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import normalize, StandardScaler
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import theano
import cPickle as pickle
import matplotlib.pyplot as plt
import socket


class AdjustVariable(object):

    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        epoch = train_history[-1]['epoch']
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class HaverSineDist(object):

    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        epoch = train_history[-1]['epoch']
        if epoch % 4 == 0:
            f = getattr(nn, self.name)
            y_pred = inverse_customize(f(X_test), SW, NE)
            d = np.zeros_like(y_test[:, 1])
            for i in range(len(y_test)):
                d[i] = haversine(y_pred[i, 1], y_pred[i, 0],
                                 y_test[i, 1], y_test[i, 0])
            dft = pd.DataFrame(
                {epoch: d, 'lat': y_pred[:, 0], 'lng': y_pred[:, 1]})
            dft.to_csv(outpath + 'latlng%d.csv' % epoch)
            dd.append(dft)
            fig = plt.figure()
            plt.scatter(dft[dft[epoch] < 1.].lng, dft[
                        dft[epoch] < 1.].lat, s=10, color='b')
            plt.scatter(dft[dft[epoch] > 1.].lng, dft[
                        dft[epoch] > 1.].lat, s=3, color='r')
            plt.scatter(y_test[:, 1], y_test[:, 0], s=2)
            plt.savefig(outpath + 'latlng_plot%05d.png' % epoch)
            plt.close(fig)
            histogramsave(d, outpath + 'distance_hist%05d.png' % epoch)

            train_loss = np.array([i["train_loss"]
                                   for i in net3.train_history_]) / 100.
            valid_loss = np.array([i["valid_loss"]
                                   for i in net3.train_history_]) / 100.
            fig = plt.figure()
            plt.plot(train_loss, linewidth=3, label="train")
            plt.plot(valid_loss, linewidth=3, label="valid")
            plt.grid()
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.ylim(1e-1, 1e1)
            plt.yscale("log")
            plt.savefig(outpath + 'train_val_loss.png')
            plt.close(fig)


if __name__ == '__main__':

    picsize = [120, 120]
    ip = socket.getfqdn()
    outpath = './model_outputs%s_%s_60/' % (picsize[0], ip)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # photodb_MainST600_40m03r/'#photodb_MainST100_25m1r#photodb_MainST_NE
    pathname = './data/photodb_MainST100_25m1r/'
    csv2read = 'folderdata.csv'

    # box including everithing
    SW_sf = np.array([37.707875, -122.518624])
    NE_sf = np.array([37.815086, -122.378205])
    SW = SW_sf
    NE = NE_sf
    csv2save = 'folderdata_SW%d_%dNE%d_%d.csv' % (SW[0], SW[1], NE[0], NE[1])

    csv2read, SW_rdata, NE_rdata = inabox_select(
        SW, NE, pathname, csv2read, csv2save)

    X_train, y_train, X_test, y_test = load2d(
        pathname, csv2read)  # load 2-d data

    y_train_trfm = customize(y_train, SW, NE)
    y_test_trfm = customize(y_test, SW, NE)
    df_ytest = pd.DataFrame(y_test)
    df_ytest.to_csv(outpath + 'y_test_df')
    dd = []
    net3 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),  # !
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),  # !
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),  # !
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),  # !
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        input_shape=(None, 3, picsize[0], picsize[0]),
        conv1_num_filters=32, conv1_filter_size=(5, 5), pool1_pool_size=(3, 3),
        dropout1_p=0.001,  # !
        conv2_num_filters=64, conv2_filter_size=(5, 5), pool2_pool_size=(3, 3),
        dropout2_p=0.001,  # !
        conv3_num_filters=128, conv3_filter_size=(4, 4), pool3_pool_size=(3, 3),
        dropout3_p=0.001,  # !
        # conv4_num_filters=256, conv4_filter_size=(8, 8), pool4_pool_size=(4, 4),
        # dropout4_p=0.0006,  # !
        hidden4_num_units=500,
        dropout4_p=0.012,  # !
        hidden5_num_units=500,
        output_num_units=2, output_nonlinearity=None,
        # update = adagrad,
        update_learning_rate=theano.shared(np.float32(0.00001), borrow=True),
        update_momentum=theano.shared(np.float32(0.01), borrow=True),
        objective_loss_function=None,
        regression=True,
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=5e-5, stop=5e-6),
            AdjustVariable('update_momentum', start=0.0001, stop=0.3),
            HaverSineDist('predict'),
        ],
        max_epochs=2000,
        verbose=1,
    )

    print("X_train.shape == {}; X_train.min == {:.3f}; X_train.max == {:.3f}".format(
        X_train.shape, X_train.min(), X_train.max()))

    net3.fit(X_train, y_train_trfm)

    with open('net64_1_SF_%d_25_1r_5.pickle' % picsize[0], 'wb') as f:
        pickle.dump(net3, f, -1)

    y_pred_trfm = net3.predict(X_test)
    y_pred = inverse_customize(y_pred_trfm, SW, NE)

    print 'size of the lat-long box for the raw data (SW%f-%f,NE%f,%f): L%f x W%f' \
    %(SW_rdata[0], SW_rdata[1], NE_rdata[0], NE_rdata[1], haversine(SW_rdata[1],\
        SW_rdata[0], NE_rdata[1], SW_rdata[0]),
            haversine(SW_rdata[1], SW_rdata[0], SW_rdata[1], NE_rdata[0]))
    df_dd = pd.DataFrame({'dist_epochs': dd})
    df_dd.to_csv(outpath + 'distance_epochs.csv')
