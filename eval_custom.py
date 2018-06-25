import glob
import json
import os
import time

import eval_util
import losses
import frame_level_models
import video_level_models
import readers
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import utils
import numpy as np
import average_precision_calculator as ap_calculator
from eval_util import top_k_by_class, flatten

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("model_file", "",
                      "The file to load the model files from.")
  flags.DEFINE_string(
      "eval_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")

  # Other flags.
  flags.DEFINE_integer("batch_size", 1024,
                       "How many examples to process per batch.")
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")
  flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")


class GapScore(object):
    def __init__(self, num_class, top_k):
        self.top_k = top_k
        self.global_ap_calculator = ap_calculator.AveragePrecisionCalculator()

    def accumulate(self, predictions, labels):
        sparse_predictions, sparse_labels, num_positives = top_k_by_class(predictions, labels, self.top_k)
        self.global_ap_calculator.accumulate(flatten(sparse_predictions), flatten(sparse_labels), sum(num_positives))

    def get(self):
        gap = self.global_ap_calculator.peek_ap_at_n()
        return gap


def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=1024,
                                 num_readers=1):
    """Creates the section of the graph which reads the evaluation data.

    Args:
      reader: A class which parses the training data.
      data_pattern: A 'glob' style path to the data files.
      batch_size: How many examples to process at a time.
      num_readers: How many I/O threads to use.

    Returns:
      A tuple containing the features tensor, labels tensor, and optionally a
      tensor containing the number of frames per video. The exact dimensions
      depend on the reader being used.

    Raises:
      IOError: If no files matching the given pattern were found.
    """
    logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
    with tf.name_scope("eval_input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find the evaluation files.")
        logging.info("number of evaluation files: " + str(len(files)))
        filename_queue = tf.train.string_input_producer(
            files, shuffle=False, num_epochs=1)
        eval_data = [
            reader.prepare_reader(filename_queue) for _ in range(num_readers)
            ]
        return tf.train.batch_join(
            eval_data,
            batch_size=batch_size,
            capacity=3 * batch_size,
            allow_smaller_final_batch=True,
            enqueue_many=True)

def fetc_inputs(reader,
                eval_data_pattern,
                batch_size=1024,
                num_readers=1):
    video_id_batch, model_input_raw, labels_batch, num_frames = get_input_evaluation_tensors(reader,
                                                                                             eval_data_pattern,
                                                                                             batch_size=batch_size,
                                                                                             num_readers=num_readers)
    return video_id_batch, model_input_raw, labels_batch, num_frames


def evaluate():
    model_path = FLAGS.model_file
    assert os.path.isfile(model_path + ".meta"), "Specified model does not exist."
    model_flags_path = os.path.join(os.path.dirname(model_path), "model_flags.json")

    if not os.path.exists(model_flags_path):
        raise IOError(("Cannot find file %s. Did you run train.py on the same "
                       "--train_dir?") % model_flags_path)
    flags_dict = json.loads(open(model_flags_path).read())


    # convert feature_names and feature_sizes to lists of values
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(flags_dict["feature_names"],
                                                                       flags_dict["feature_sizes"])

    if flags_dict["frame_features"]:
            reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                                    feature_sizes=feature_sizes)
    else:
        raise NotImplementedError

    video_id_batch, model_input_raw, labels_batch, num_frames = fetc_inputs(reader,
                                                                            FLAGS.eval_data_pattern,
                                                                            FLAGS.batch_size,
                                                                            FLAGS.num_readers)
    evl_metrics = GapScore(reader.num_classes, FLAGS.top_k)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        meta_graph_location = model_path + ".meta"
        logging.info("loading meta-graph: " + meta_graph_location)

        saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
        saver.restore(sess, model_path)

        input_tensor = tf.get_collection("input_batch_raw")[0]
        num_frames_tensor = tf.get_collection("num_frames")[0]
        predictions_tensor = tf.get_collection("predictions")[0]
        # loss = tf.get_collection("loss")[0]

        # Workaround for num_epochs issue.
        def set_up_init_ops(variables):
            init_op_list = []
            for variable in list(variables):
                if "train_input" in variable.name:
                    init_op_list.append(tf.assign(variable, 1))
                    variables.remove(variable)
            init_op_list.append(tf.variables_initializer(variables))
            return init_op_list

        sess.run(set_up_init_ops(tf.get_collection_ref(
            tf.GraphKeys.LOCAL_VARIABLES)))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        num_examples_processed = 0
        start_time = time.time()

        try:
            while not coord.should_stop():
                xvideo_id_batch, xmodel_input_raw, xlabels_batch, xnum_frames = sess.run(
                    [video_id_batch, model_input_raw, labels_batch, num_frames])
                predictions_val = sess.run([predictions_tensor], feed_dict={input_tensor: xmodel_input_raw,
                                                                                 num_frames_tensor: xnum_frames})[0]
                now = time.time()
                num_examples_processed += len(predictions_val)
                # num_classes = predictions_val.shape[1]
                logging.info(
                    "num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(
                        now - start_time))

                # Collect metrics
                evl_metrics.accumulate(predictions_val, xlabels_batch)

        except tf.errors.OutOfRangeError:
            logging.info('Done with inference.')
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

        logging.info('Calculating statistics.')
        epoch_info_dict = evl_metrics.get()
        logging.info('GAP Score: {}'.format(epoch_info_dict))


def main(unused_argv):
    logging.set_verbosity(tf.logging.INFO)
    evaluate()


if __name__ == "__main__":
    app.run()
