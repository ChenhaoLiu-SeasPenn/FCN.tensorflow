from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

import TensorflowUtils as utils
import read_in_data as scene_parsing
import datetime
import TFReader as dataset
from six.moves import xrange
from contrib import dense_crf, vgg_net_singlechannel

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_integer("class_num", "2", "number of classes")
tf.flags.DEFINE_integer("gpu", "0", "specify which GPU to use")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/dataset/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_bool('image_augmentation', "False", "Image augmentation: True/ False")
tf.flags.DEFINE_float('dropout', "0.5", "Probably of keeping value in dropout (valid values (0.0,1.0]")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize/ predict") #test not implemented
tf.flags.DEFINE_float('pos_weight', '1', 'Weight for FNs, higher increases recall')

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(5e4+1)
NUM_OF_CLASSESS = FLAGS.class_num
# IMAGE_SIZE = 224
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net

def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    # processed_image = utils.process_image(image, mean_pixel)
    # single channel version
    processed_image = image

    with tf.variable_scope("inference"):
        # image_net = vgg_net(weights, processed_image)
        # single channel version
        image_net = vgg_net_singlechannel(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        # A strided convolution downsamples 7x7 feature map to 4x4
        W6 = utils.weight_variable([3, 3, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_strided(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

        annotation_pred = tf.argmax(conv8, axis=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv8


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    with tf.device('/device:GPU:0'):
        keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        # image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="input_image")
        # try using mere segmentation as input for this task
        image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="input_image")
        annotation = tf.placeholder(tf.int32, shape=[None, 4, 4, 1], name="annotation")

        pred_annotation, logits = inference(image, keep_probability)
        tf.summary.image("input_image", image, max_outputs=FLAGS.batch_size)
        tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=FLAGS.batch_size)
        tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=FLAGS.batch_size)
        weights = np.ones((FLAGS.class_num), dtype=np.int32)
        weights[-1] = FLAGS.pos_weight
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        if FLAGS.pos_weight == 1:
            loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                              labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                              name="entropy")))
        else:
            loss = tf.reduce_mean((tf.nn.weighted_cross_entropy_with_logits(targets = tf.one_hot(tf.squeeze(annotation, squeeze_dims=[3]), FLAGS.class_num, axis=-1),
                                                                              logits=logits,
                                                                              pos_weight=weights,
                                                                              name="entropy")))
        # tf.nn.weighted_cross_entropy_with_logits
        
        loss_summary = tf.summary.scalar("entropy", loss)
        iou_error, update_op = tf.metrics.mean_iou(pred_annotation, annotation, NUM_OF_CLASSESS)

        trainable_var = tf.trainable_variables()
        if FLAGS.debug:
            for var in trainable_var:
                utils.add_to_regularization_and_summary(var)
        train_op = train(loss, trainable_var)

        print("Setting up summary op...")
        summary_op = tf.summary.merge_all()

        print("Setting up image reader...")
    print('Here')
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir, pwc=True)
#     print(valid_records)
    print("No. train records: ", len(train_records))
    print("No. validation records: ", len(valid_records))

    print("Setting up dataset reader")
    image_options_train = {'resize': True, 'resize_width': IMAGE_WIDTH, 'resize_height': IMAGE_HEIGHT, 'image_augmentation':FLAGS.image_augmentation}
    image_options_val = {'resize': True, 'resize_width': IMAGE_WIDTH, 'resize_height': IMAGE_HEIGHT}
    if FLAGS.mode == 'train':
        train_val_dataset = dataset.TrainVal.from_records(
            train_records, valid_records, image_options_train, image_options_val, FLAGS.batch_size, FLAGS.batch_size, pwc=True)
    #validation_dataset_reader = dataset.BatchDatset(valid_records, image_options_val)

    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config= config)

        print("Setting up Saver...")
        saver = tf.train.Saver()

        # create two summary writers to show training loss and validation loss in the same graph
        # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
        train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        if FLAGS.mode == "train":
            it_train, it_val = train_val_dataset.get_iterators()
            # get_next = iterator.get_next()
            #training_init_op, val_init_op = train_val_dataset.get_ops()
            if FLAGS.dropout <=0 or FLAGS.dropout > 1:
                raise ValueError("Dropout value not in range (0,1]")
            #sess.run(training_init_op)

            #Ignore filename from reader
            next_train_images, next_train_annotations, next_train_name = it_train.get_next()
            next_val_images, next_val_annotations, next_val_name = it_val.get_next()
            for i in xrange(MAX_ITERATION):
    #             print(sess.run(next_train_name))
                train_images, train_annotations, train_name = sess.run([next_train_images, next_train_annotations, next_train_name])
#                 print(train_annotations)
    #             print(train_images)
    #             print(train_annotations)
                feed_dict = {image: train_images, annotation: train_annotations, keep_probability: (1 - FLAGS.dropout)}

                sess.run(train_op, feed_dict=feed_dict)

                if i % 10 == 0:
                    train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                    print("Step: %d, Train_loss:%g" % (i, train_loss))
                    train_writer.add_summary(summary_str, i)

                if i % 500 == 0:
                    #sess.run(val_init_op)

                    valid_images, valid_annotations = sess.run([next_val_images, next_val_annotations])
                    valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations,
                                                           keep_probability: 1.0})
                    print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                    # add validation loss to TensorBoard
                    validation_writer.add_summary(summary_sva, i)
                    saver.save(sess, FLAGS.logs_dir + "model.ckpt", i)
                    #sess.run(training_init_op)


        elif FLAGS.mode == "visualize":
            iterator = train_val_dataset.get_iterator()
            get_next = iterator.get_next()
            training_init_op, val_init_op = train_val_dataset.get_ops()
            sess.run(val_init_op)
            valid_images, valid_annotations = sess.run(get_next)
            pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                        keep_probability: 1.0})
            valid_annotations = np.squeeze(valid_annotations, axis=3)
            pred = np.squeeze(pred, axis=3)

            for itr in range(FLAGS.batch_size):
                utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
                utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
                utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
                print("Saved image: %d" % itr)

        elif FLAGS.mode == "predict":
            predict_records = scene_parsing.read_prediction_set(FLAGS.data_dir)
            no_predict_images = len(predict_records)
            print ("No. of predict records {}".format(no_predict_images))
            predict_image_options = {'resize': True, 'resize_width': IMAGE_WIDTH, 'resize_height': IMAGE_HEIGHT, 'predict_dataset': True}
            test_dataset_reader = dataset.SingleDataset.from_records(predict_records, predict_image_options)
            next_test_image = test_dataset_reader.get_iterator().get_next()
            if not os.path.exists(os.path.join(FLAGS.logs_dir, "predictions")):
                os.makedirs(os.path.join(FLAGS.logs_dir, "predictions"))
            for i in range(no_predict_images):
                if (i % 10 == 0):
                    print("Predicted {}/{} images".format(i, no_predict_images))
                predict_images, predict_names = sess.run(next_test_image)
                pred = sess.run(pred_annotation, feed_dict={image: predict_images,
                                                            keep_probability: 1.0})
                pred = np.squeeze(pred, axis=3)
                utils.save_image((pred[0]).astype(np.uint8), os.path.join(FLAGS.logs_dir, "predictions"),
                                 name="predict_" + str(predict_names))
        
#         predict_records = scene_parsing.read_prediction_set(FLAGS.data_dir)
#         no_predict_images = len(predict_records)
#         print ("No. of predict records {}".format(no_predict_images))
#         predict_image_options = {'resize': True, 'resize_size': IMAGE_SIZE, 'predict_dataset': True}
#         test_dataset_reader = dataset.SingleDataset.from_records(predict_records, predict_image_options)
#         next_test_image = test_dataset_reader.get_iterator().get_next()
#         print("Predicting {} images".format(no_predict_images))

#         if not os.path.exists(os.path.join(FLAGS.logs_dir, "predictions")):
#             os.makedirs(os.path.join(FLAGS.logs_dir, "predictions"))
#         for i in range(no_predict_images):
#             if (i % 10 == 0):
#                 print("Predicted {}/{} images".format(i, no_predict_images))
#             predict_images, predict_names = sess.run(next_test_image)
#             pred = sess.run(pred_annotation, feed_dict={image: predict_images,
#                                                         keep_probability: 1.0})
#             pred = np.squeeze(pred, axis=3)
#             utils.save_image(pred[0].astype(np.uint8), os.path.join(FLAGS.logs_dir, "predictions"),
#                              name="predict_" + str(predict_names))

        elif FLAGS.mode == "test":

          test_records, _ = scene_parsing.read_dataset(FLAGS.data_dir, pwc=True, test=True)
          #             print(test_records)
          image_options_train = {'resize': True, 'resize_width': IMAGE_WIDTH, 'resize_height': IMAGE_HEIGHT,
                                 'image_augmentation': False}
          image_options_val = {'resize': True, 'resize_width': IMAGE_WIDTH, 'resize_height': IMAGE_HEIGHT}

          test_dataset = dataset.TrainVal.from_records(
            test_records, test_records, image_options_train, image_options_val, FLAGS.batch_size, FLAGS.batch_size, pwc=True)

          iou_result = np.zeros([int(np.ceil(len(test_records) / FLAGS.batch_size))])
          it_test, _ = test_dataset.get_iterators()
          next_test_images, next_test_annotations, next_test_name = it_test.get_next()
          print('All images:{}, batch size: {}'.format(len(test_records), FLAGS.batch_size))
          #             print(int(np.ceil(len(test_records) / FLAGS.batch_size)))
          for i in range(int(np.ceil(len(test_records) / FLAGS.batch_size))):
            test_images, test_annotations = sess.run([next_test_images, next_test_annotations])
            feed_dict = {image: test_images, annotation: test_annotations, keep_probability: 1.0}

            confusion = sess.run(update_op, feed_dict=feed_dict)
            mean_iou = sess.run(iou_error)
            print("Test batch {}, mean iou {}".format(i, mean_iou))
            print(confusion)


if __name__ == "__main__":
    tf.app.run()