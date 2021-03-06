from uwnet import *

def softmax_model():
    l = [make_connected_layer(3072, 10, SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(3072, 32, LRELU),
            make_connected_layer(32, 10, SOFTMAX)]
    return make_net(l)

def conv_net():
    # The network below achieves 64.2% train and 60.5% test.
    # How many operations are needed for a forard pass through this network?
    # Your answer: 1,108,480
    # 32 * 32 * 3 * 8 * 3 * 3 / 1
    # + 16 * 16 * 8 * 16 * 3 * 3 / 1
    # + 8 * 8 * 16 * 32 * 3 / 1
    # + 4 * 4 * 32 * 64 * 3 * 3 / 1
    # + 256 * 10
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)

def your_net():
    # Define your network architecture here. It should have 5 layers. How many operations does it need for a forward pass?
    # It doesn't have to be exactly the same as conv_net but it should be close.
    # Networks I tried:
    # 55.1% train, 50.0% test
    # l = [   make_connected_layer(3072, 3072, LRELU),
    #         make_connected_layer(3072, 10, SOFTMAX)]
    #
    # 55.0% train, 51.3% test
    # l = [   make_connected_layer(3072, 256, LRELU),
    #         make_connected_layer(256, 512, LRELU),
    #         make_connected_layer(512, 10, LRELU),
    #
    # The network defined below achieves 56.3% train and 51.3% test accuracies.
    # This is in comparison to the 64.2% train and 60.5% test accuracies that
    # conv_net achieves.
    # From these experiments, it seems that convolutional neural networks make
    # better use of their parameters than fully connected layers. This is
    # because convolutions allow for weight sharing---we can use the same
    # parameters across multiple inputs. Also, convolutions are built with
    # spatial bias which is crucial when processing images.
    l = [   make_connected_layer(3072, 256, LRELU),
            make_connected_layer(256, 256, LRELU),
            make_connected_layer(256, 256, LRELU),
            make_connected_layer(256, 512, LRELU),
            make_connected_layer(512, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test", "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = your_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#

