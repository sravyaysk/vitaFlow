# Copyright 2018 The vitaFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Conditional GAN
"""
import os
import math
import collections

import tensorflow as tf
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.contrib.learn import ModeKeys

from vitaflow.core.features import ImageFeature
from vitaflow.helpers.print_helper import print_info, print_error
from vitaflow.core.models import ModelBase
from vitaflow.utils.image_utils import images_square_grid


class RunTrainOpsHook(session_run_hook.SessionRunHook):
    """A hook to run train ops a fixed number of times."""

    def __init__(self, train_op, train_steps):

        self._train_op = train_op
        self._train_steps = train_steps

    def before_run(self, run_context):
        for _ in range(self._train_steps):
            run_context.session.run(self._train_op)

class LogShapeHook(session_run_hook.SessionRunHook):
    """A hook to run train ops a fixed number of times."""

    def __init__(self, tensors):
        self._tensors = tensors

    def before_run(self, run_context):
        for tensor in self._tensors:
            # print_error(tensor)
            # print_error(tensor.get_shape())
            pass

class UserLogHook(session_run_hook.SessionRunHook):
    def __init__(self, z_image, d_loss, g_loss, global_Step):
        self._z_image = z_image
        self._d_loss = d_loss
        self._g_loss = g_loss
        self._global_Step = global_Step


    def before_run(self, run_context):
        global_step = run_context.session.run(self._global_Step)

        print_info("global_step {}".format(global_step))

        if global_step % 2 == 0 or global_step % 3 == 0 :
            samples = run_context.session.run(self._z_image)
            channel = self._z_image.get_shape()[-1]
            if channel == 1:
                images_grid= images_square_grid(samples, "L")
            else:
                images_grid= images_square_grid(samples, "RGB")

            if not os.path.exists(os.path.join(os.path.expanduser("~"), "vitaFlow/runtime/CGAN")): os.makedirs(EXPERIMENT_DATA_ROOT_DIR+'/vanilla_gan/' )

            images_grid.save(os.path.join(os.path.expanduser("~"), "vitaFlow/runtime/CGAN", '/step_{}.png'.format(global_step)))

        if global_step % 2 == 0:
            dloss, gloss = run_context.session.run([self._d_loss, self._g_loss])
            print_info("\nDiscriminator Loss: {:.4f}... Generator Loss: {:.4f}".format(dloss, gloss))

class GANTrainSteps(
    collections.namedtuple('GANTrainSteps', (
            'generator_train_steps',
            'discriminator_train_steps'
    ))):
    """Contains configuration for the GAN Training.

    Args:
      generator_train_steps: Number of generator steps to take in each GAN step.
      discriminator_train_steps: Number of discriminator steps to take in each GAN
        step.
    """


class CGAN(ModelBase, ImageFeature):
    """
        https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/CGAN.py
    """

    def __init__(self, hparams=None, data_iterator=None):
        ModelBase.__init__(self, hparams=hparams)
        ImageFeature.__init__(self)

        self._data_iterator = data_iterator

        self.num_image_channels = 3
        self.image_size = 32

        self.gen_filter_size = 1024
        self.learning_rate = 0.001
        self.alpha = 0.15
        self.beta1 = 0.4
        self.z_dim = 30

    def get_sequential_train_hooks(self, generator_train_op,
                                   discriminator_train_op,
                                   train_steps=GANTrainSteps(1, 1)):
        """Returns a hooks function for sequential GAN training.

        Args:
          train_steps: A `GANTrainSteps` tuple that determines how many generator
            and discriminator training steps to take.

        Returns:
          A function that takes a GANTrainOps tuple and returns a list of hooks.
        """
        # print_info(generator_train_op)
        # print_info(discriminator_train_op)

        generator_hook = RunTrainOpsHook(generator_train_op,
                                         train_steps.generator_train_steps)
        discriminator_hook = RunTrainOpsHook(discriminator_train_op,
                                             train_steps.discriminator_train_steps)
        return [discriminator_hook, generator_hook]

    def discriminator(self, images, input_z, reuse=False):
        """
        Create the discriminator network
        :param image: Tensor of input image(s)
        :param reuse: Boolean if the weights should be reused
        :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
        """



        hook = LogShapeHook([images, input_z])
        _,width,height,channel = images.get_shape().as_list()

        print_error("{}".format([width,height,channel]))

        with tf.variable_scope('discriminator', reuse=reuse):

            # y = tf.reshape(input_z, [-1, 1, 1, 740], name="y_reshape") #2*2*185=>740
            # input_z = tf.layers.batch_normalization(input_z)
            # input_z = tf.layers.dense(input_z, width*height*channel)
            # y = tf.reshape(input_z, [-1, width,height,channel], name="y_reshape")
            # print_error(y)


            # x1 = conv_cond_concat(y,images)
            '''
            # c_code = tf.expand_dims(tf.expand_dims(input_z, 1), 1)
            # c_code = tf.tile(c_code, [1, 1, 1, 740])
            # x1 = tf.concat([images, c_code], 3)
            # print_error(x1)
            '''
            # x1 = tf.layers.batch_normalization(images)

            # Input layer consider ?x32x32x3
            x1 = tf.layers.conv2d(images, 64, 5, strides=2, padding='same',
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            x1= tf.maximum(self.alpha * x1, x1)

            x1 = tf.layers.batch_normalization(x1, training=True)
            # relu1 = tf.layers.dropout(relu1, rate=0.5)
            # 16x16x64
            #         print(x1)
            x2 = tf.layers.conv2d(x1, 128, 5, strides=2, padding='same',
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            x2 = tf.maximum(self.alpha * x2, x2)


            x2 = tf.layers.batch_normalization(x2, training=True)
            # relu2 = tf.layers.dropout(relu2, rate=0.5)
            # 8x8x128
            #         print(x2)
            x3 = tf.layers.conv2d(x2, 256, 5, strides=2, padding='same',
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            x3 = tf.maximum(self.alpha * x3, x3)

            x3 = tf.layers.batch_normalization(x3, training=True)
            # relu3 = tf.layers.dropout(relu3, rate=0.5)
            # 4x4x256
            #         print(x3)
            # Flatten it
            flat = tf.reshape(x3, (-1, 4 * 4 * 256))

            flat = tf.concat([flat,input_z], -1)

            # conditioned_fully_connected_layer = tf.concat([flat], axis=-1)
            #
            # flat = tf.layers.dense(flat, 512)
            # flat = tf.layers.dense(flat, 1024)
            # flat = tf.layers.dense(flat, 512)
            # flat = tf.layers.dense(flat, 256)
            # flat = tf.layers.dense(flat, 128)


            logits = tf.layers.dense(flat, 512)
            logits = tf.maximum(self.alpha * logits, logits)
            logits = tf.layers.dense(logits, 1)

            #         print(logits)
            out = tf.sigmoid(logits)
            #         print('discriminator out: ', out)

            print_info("======>out: {}".format(out))


            return out, logits, hook

    def generator(self, z, out_channel_dim, is_train=True):
        """
        Create the generator network
        :param z: Input z on dimension Z
        :param out_channel_dim: The number of channels in the output image
        :param is_train: Boolean if generator is being used for training
        :return: The tensor output of the generator
        """

        with tf.variable_scope('generator', reuse=False ):
            gen_filter_size = self.gen_filter_size

            # x = tf.layers.batch_normalization(z)
            # First fully connected layer
            x = tf.layers.dense(z, 8 * 8 * gen_filter_size)
            # Reshape it to start the convolutional stack
            x = tf.reshape(x, (-1, 8, 8, gen_filter_size))
            # x = tf.layers.batch_normalization(x, training=is_train)
            x = tf.maximum(self.alpha * x, x)

            x = tf.layers.conv2d_transpose(x, gen_filter_size // 2, 5, strides=1, padding='same')
            x = tf.maximum(self.alpha * x, x)

            x = tf.layers.batch_normalization(x, training=is_train)

            gen_filter_size = gen_filter_size // 4
            # 32 //  8 = srt(4)  => 2 => (8) -> 16 -> 32
            # 64 //  8 = srt(8)  => 3 => (8) -> 16 -> 32 -> 64
            # 128 // 8 = srt(16) => 4 => (8) -> 16 -> 32 -> 64 -> 128

            # Based on image size adds Conv layer with appropriate filter size
            for i in range(int(math.sqrt(self.image_size // 8))):
                gen_filter_size = gen_filter_size // 2
                x = tf.layers.conv2d_transpose(x, gen_filter_size, 5, strides=2, padding='same')
                x = tf.maximum(self.alpha * x, x)
                x = tf.layers.batch_normalization(x, training=is_train)

                print_info("======>x at conv layer {} is {}".format(i, x))

            # Output layer
            logits = tf.layers.conv2d_transpose(x, out_channel_dim, 5, strides=1, padding='same')
            # HxWxNUM_CHANNELS now
            out = tf.tanh(logits)

            print_info("======>out: {}".format(out))

            hook = LogShapeHook([out])
            if is_train:
                return out, hook
            else:
                return out


    def model_loss(self, input_real, input_z, out_channel_dim, global_step):
        """
        Get the loss for the discriminator and generator
        :param input_real: Images from the real dataset
        :param input_z: Z input
        :param out_channel_dim: The number of channels in the output image
        :return: A tuple of (discriminator loss, generator loss)
        """
        #     print('Generator for fake images...')
        g_model, gen_hook = self.generator(input_z, out_channel_dim)
        #     print('Passing discriminator with real images...')
        d_model_real, d_logits_real, hook1 = self.discriminator(input_real, input_z)
        #     print('Passing discriminator with fake images...')
        d_model_fake, d_logits_fake, hook2 = self.discriminator(g_model, input_z, reuse=True)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

        d_loss = d_loss_real + d_loss_fake

        print_hooks = UserLogHook(g_model, d_loss, g_loss, global_step)

        return d_loss, g_loss, [print_hooks, hook1, hook2, gen_hook]

    def model_opt(self, d_loss, g_loss, learning_rate, beta1, global_step):
        """
        Get optimization operations
        :param d_loss: Discriminator loss Tensor
        :param g_loss: Generator loss Tensor
        :param learning_rate: Learning Rate Placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: A tuple of (discriminator training operation, generator training operation)
        """

        # Get weights and bias to update
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Optimize
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name="d_train_opt"). \
            minimize(d_loss, var_list=d_vars, global_step=global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        g_updates = [opt for opt in update_ops if opt.name.startswith('generator')]

        with tf.control_dependencies(g_updates):
            g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name="g_train_opt"). \
                minimize(g_loss, var_list=g_vars, global_step=global_step)

        # tf.logging.info("=========> {}".format(d_train_opt))
        # tf.logging.info("=========> {}".format(g_train_opt))

        return d_train_opt, g_train_opt


    def _get_loss(self, labels, logits):
        raise NotImplementedError

    def _build_layers(self, features, mode):
        raise NotImplementedError

    def _get_predicted_classes(self, logits):
        raise NotImplementedError

    def _get_class_probabilities(self, logits):
        raise NotImplementedError

    def _get_optimizer(self, loss):
        raise NotImplementedError

    def _get_eval_metrics(self, predictions, labels):
        raise NotImplementedError

    def _build(self, features, labels, params, mode, config=None):
        """Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        """

        :param features: 
        :param labels: 
        :param mode: 
        :param params: 
        :return: 
        """

        sample_image = None
        training_hooks = None

        # Create global step increment op.
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_inc = self.global_step.assign_add(0)

        z_placeholder = tf.random_normal(shape=[self._data_iterator.batch_size, 30])#features[self.FEATURE_NAME]  # Audio/Noise Placeholder to the discriminator
        tf.logging.info("=========> {}".format(z_placeholder))

        z_placeholder = tf.cast(z_placeholder, tf.float32)

        tf.logging.info("=========> {}".format(z_placeholder))

        if mode != ModeKeys.INFER:

            x_placeholder = features[self.FEATURE_NAME]  # Placeholder for input image vectors to the generator
            tf.logging.info("=========> {}".format(x_placeholder))

            x_placeholder = tf.cast(x_placeholder, tf.float32)
            tf.logging.info("=========> {}".format(x_placeholder))

            channel = x_placeholder.get_shape()[-1]
            d_loss, g_loss, print_hooks = self.model_loss(x_placeholder, z_placeholder, channel, self.global_step)

            d_train_opt, g_train_opt = self.model_opt(d_loss, g_loss,
                                                      self.learning_rate,
                                                      self.beta1,
                                                      self.global_step)
        else:
            sample_image = self.generator(z_placeholder, self.num_image_channels,is_train=False)
            #changes are made to take image channels from data iterator just for prediction


        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            loss = g_loss + d_loss
            tf.summary.scalar(name="g_loss", tensor=g_loss)
            tf.summary.scalar(name="d_loss", tensor=d_loss)

            training_hooks = self.get_sequential_train_hooks(d_train_opt, g_train_opt)
            training_hooks.extend(print_hooks)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=sample_image,
            loss=loss,
            train_op=self.global_step_inc,
            eval_metric_ops=eval_metric_ops,
            training_hooks=training_hooks
        )