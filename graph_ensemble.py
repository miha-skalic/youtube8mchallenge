import argparse
import tensorflow as tf
import os
from tensorflow import logging
import shutil

parser = argparse.ArgumentParser(description='TF graph model ensembler.')

parser.add_argument('--save_folder', action="store", type=str, required=True)
parser.add_argument('--models', nargs='+', action="store", type=str, required=True,
                    help="models must be type '/path/to/model/inference_model'")
parser.add_argument('--weights', nargs='+', action="store", type=float, required=True,
                    help="list of weights, must sum up to 1.'")
params = parser.parse_args()

in_models = params.models
weights = params.weights
save_folder = params.save_folder

# Make sure you do not overwrite
assert not os.path.isdir(save_folder), "Point to non-exisiting Directory!"

# Make sure you do now have weights not summing to 1
assert abs(sum(weights) - 1) < 0.0001, "Weights do not sum up to 1"

# Make sure you have correct files
for in_model in in_models:
    assert os.path.isfile(in_model + ".meta")

assert len(in_models) == len(weights)
assert len(weights) > 1

common_input = tf.placeholder(tf.float32, shape=(None, 300, 1152), name="CommonIn")
common_frames = tf.placeholder(tf.int32, shape=(None,))

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    for i, in_model in enumerate(in_models):
        with tf.variable_scope("model_{}".format(i)), tf.device("/cpu:0"):
            saver = tf.train.import_meta_graph(in_model + ".meta", clear_devices=True)
        logging.info("restoring variables from " + in_model)
        saver.restore(sess, in_model)

    input_tensor = tf.get_collection("input_batch_raw")
    num_frames_tensor = tf.get_collection("num_frames")
    predictions_tensor = tf.get_collection("predictions")

    # Commect nodes of the graph
    for g_inpt in input_tensor:
        tf.contrib.graph_editor.connect(common_input, g_inpt)
    for g_inpt in num_frames_tensor:
        tf.contrib.graph_editor.connect(common_frames, g_inpt)

    final_out = predictions_tensor[0] * weights[0]
    # MapReduce the output
    for sumodel_out, wfactor in zip(predictions_tensor[1:], weights[1:]):
        final_out = final_out + sumodel_out * wfactor

        # Reset I/O collections
    tf.get_default_graph().clear_collection("input_batch_raw")
    tf.add_to_collection("input_batch_raw", common_input)

    tf.get_default_graph().clear_collection("num_frames")
    tf.add_to_collection("num_frames", common_frames)

    tf.get_default_graph().clear_collection("predictions")
    tf.add_to_collection("predictions", final_out)

    # Save the model
    os.makedirs(save_folder)

    # By quantizing we might end up with uninitialized variables.
    uninit = set(sess.run(tf.report_uninitialized_variables()))
    used_vars = [v for v in tf.global_variables() if v.name.split(':')[0] not in uninit]

    saver = tf.train.Saver(used_vars)
    saver.save(sess, os.path.join(save_folder, "inference_model"))

    # copy flags
    ref_falgs = os.path.join(os.path.dirname(in_models[-1]), "model_flags.json")
    shutil.copy(ref_falgs, save_folder)
    logging.info("We are done!")