import tensorflow as tf
import glob
import os
import gin


@gin.configurable
class CIDARIterator:
    """
    """

    def __init__(self,
                 data_dir=gin.REQUIRED,
                 num_threads=4,
                 batch_size=32,
                 prefetch_size=32):
        """
        ......
        """
    
        self._data_dir = data_dir
        self._num_threads = num_threads
        self._batch_size = batch_size
        self._prefetch_size = prefetch_size

    def train_input_fn(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_train_input_fn()

    def val_input_fn(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_val_input_fn()

    def test_input_fn(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_test_input_fn()

    def decode(self, serialized_example):
        # 1. define a parser
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image': tf.FixedLenFeature([512 * 512 * 3], tf.float32),
                'score_map': tf.FixedLenFeature([128 * 128 * 1], tf.float32),
                'geo_map': tf.FixedLenFeature([128 * 128 * 5], tf.float32),
                'training_masks': tf.FixedLenFeature([128 * 128 * 1], tf.float32),
            })

        image = tf.reshape(
            tf.cast(features['image'], tf.float32), shape=[512, 512, 3])
        score_map = tf.reshape(
            tf.cast(features['score_map'], tf.float32), shape=[128, 128, 1])
        geo_map = tf.reshape(
            tf.cast(features['geo_map'], tf.float32), shape=[128, 128, 5])
        training_masks = tf.reshape(
            tf.cast(features['training_masks'], tf.float32), shape=[128, 128, 1])

        return {"image": image, "score_map": score_map, "geo_map": geo_map, "training_masks": training_masks}, training_masks

    def _get_train_input_fn(self):
        """
        Inheriting class must implement this
        :return: dataset
        """
        # TF dataset APIs
        dataset = tf.data.TFRecordDataset(glob.glob(os.path.join(self._data_dir, "train/*.tfrecord")),
                                          num_parallel_reads=self._num_threads)
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(self.decode)

        dataset = dataset.batch(
            batch_size=self._batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._prefetch_size)
        # dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "train_data_cache"))
        print("Dataset output sizes are: ")
        print(dataset.output_shapes)

        return dataset

    #

    def _get_val_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        # TF dataset APIs
        dataset = tf.data.TFRecordDataset(glob.glob(os.path.join(self._data_dir, "test/*.tfrecord")),
                                          num_parallel_reads=self._num_threads)
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(self.decode)

        dataset = dataset.batch(
            batch_size=self._batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._prefetch_size)
        print("Dataset output sizes are: ")
        print(dataset.output_shapes)
        return dataset

    # def _get_test_input_function(self):
        # 	"""
        # 	Inheriting class must implement this
        # 	:return: callable
        # 	"""
        # 	dataset = tf.data.TFRecordDataset(glob.glob(os.path.join(self._dataset.TEST_OUT_PATH, "tfrecords/*.tfrecord")),
        # 																		num_parallel_reads=self._hparams.num_threads)
        # 	# Map the generator output as features as a dict and labels
        # 	dataset = dataset.map(self.decode)

        # 	dataset = dataset.batch(
        # 			batch_size=self._hparams.batch_size, drop_remainder=True)
        # 	dataset = dataset.prefetch(self._hparams.prefetch_size)
        # 	print("Dataset output sizes are: ")
        # 	print(dataset.output_shapes)
        # 	return dataset
