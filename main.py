import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
from glob import glob
import re
import scipy.misc
import numpy as np
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tags = [vgg_tag]
    loaded = tf.saved_model.loader.load(sess, tags, vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)



def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output (original dimension = 1000)
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output (original dimension = 512)
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output (original dimension = 256)
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Implement function
    #   1x1 convolution output channels:  performance improves with higher number of channels
    #   min prior layer dimension is 256
    #   the Long paper uses 21 channels (20 classes + background)
    fc32 = tf.contrib.layers.conv2d(vgg_layer7_out, 256, 1, 1)
    fc32 = tf.contrib.layers.conv2d_transpose(fc32, num_classes, 32, 32)
    fc16 = tf.contrib.layers.conv2d(vgg_layer4_out, 256, 1, 1)
    fc16 = tf.contrib.layers.conv2d_transpose(fc16, num_classes, 16, 16)
    fc16 = fc16 + fc32
    fc8 = tf.contrib.layers.conv2d(vgg_layer3_out, 256, 1, 1)
    fc8 = tf.contrib.layers.conv2d_transpose(fc8, num_classes, 8, 8)
    fc8 = fc8 + fc16
    return fc8

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1).minimize(cost)
    
    return logits, optimizer, cost
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # Implement function
    for epoch in range(epochs):
        batch = 0
        for value in get_batches_fn(batch_size):
            print("epoch {} batch: {}".format(epoch,batch))
            batch_images = value[0]
            batch_labels = value[1]
            if batch_images.shape[0] == batch_size:
                nop, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: batch_images, correct_label: batch_labels,
                                                                                keep_prob: 0.5, learning_rate: 0.005})
                batch = batch + 1
                print("loss: {}".format(loss))
    pass
tests.test_train_nn(train_nn)



def augment_training(data_folder):
    # from helper.py; use the same file and path names
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
    for image_file in image_paths:
        gt_image_file = label_paths[os.path.basename(image_file)]
        image = scipy.misc.imread(image_file)
        gt_image = scipy.misc.imread(gt_image_file)
        image_name = os.path.basename(image_file)
        gt_image_name = os.path.basename(gt_image_file)

        print('image_name: {}, gt_image_name: {}'.format(image_name, gt_image_name))

        if True:
            # apply idential operations on both primary image and ground truth image
        
            # mirror
            image_modified = np.fliplr(image)
            gt_image_modified = np.fliplr(gt_image)
            scipy.misc.imsave(os.path.join(data_folder, 'image_2a', 'flip_' + image_name), image_modified)
            scipy.misc.imsave(os.path.join(data_folder, 'gt_image_2a', 'flip_' + gt_image_name), gt_image_modified)

            # rotate left
            theta = 3.0
            image_modified = scipy.misc.imrotate(image, theta)
            gt_image_modified = scipy.misc.imrotate(gt_image, theta)
            scipy.misc.imsave(os.path.join(data_folder, 'image_2b', 'rotl_' + image_name), image_modified)
            scipy.misc.imsave(os.path.join(data_folder, 'gt_image_2b', 'rotl_' + gt_image_name), gt_image_modified)

            # rotate right
            theta = -3.0
            image_modified = scipy.misc.imrotate(image, theta)
            gt_image_modified = scipy.misc.imrotate(gt_image, theta)
            scipy.misc.imsave(os.path.join(data_folder, 'image_2c', 'rotr_' + image_name), image_modified)
            scipy.misc.imsave(os.path.join(data_folder, 'gt_image_2c', 'rotr_' + gt_image_name), gt_image_modified)


def run():
    num_classes = 2
    image_shape = (160, 576)
    channels = 3
    data_dir = './data'
    runs_dir = './runs'
    
    # This test passed prior to augmentation
    #   I now have 4x the number of images so the test would assert
    if False:
        tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # only do augmentation once...
    #   after generating the modified images I placed them in the standard data directory
    if False:        
        augment_training(os.path.join(data_dir, 'data_road/training'))
    
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        batch_size = 25
        epochs = 250
        keep_prob = tf.placeholder(tf.float32)
        learning_rate = tf.placeholder(tf.float32)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        
        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        
        final_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        
        correct_label = tf.Variable(tf.zeros(shape=(batch_size, image_shape[0], image_shape[1], num_classes)))
        input_image = tf.Variable(tf.zeros(shape=(batch_size, image_shape[0], image_shape[1], channels)), dtype=tf.float32)
        
        logits, train_op, cross_entropy_loss = optimize(final_layer, correct_label, learning_rate, num_classes)
                
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                 image_input, correct_label, keep_prob, learning_rate)

        saver.save(sess, 'trained_model', global_step=epochs)
        
        # Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        helper.save_inference_samples(os.path.join(data_dir, 'data_road_test_results'),
                                      data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
