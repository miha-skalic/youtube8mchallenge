import argparse
import numpy as np
import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow
import shutil

parser = argparse.ArgumentParser(description='TF graph weights-averager.')

parser.add_argument('--save_folder', action="store", type=str, required=True)
parser.add_argument('--models', nargs='+', action="store", type=str, required=True,
                    help="models must be type '/path/to/model/inference_model'")

params = parser.parse_args()
in_models = params.models
save_folder = params.save_folder

# Make sure you do not overwrite
assert not os.path.isdir(save_folder), "Point to non-exisiting Directory!"

# Make sure you have correct files
for in_model in in_models:
    assert os.path.isfile(in_model + ".meta")

in_model = in_models[0]

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    readers = [pywrap_tensorflow.NewCheckpointReader(xmodel) for xmodel in in_models]
    saver = tf.train.import_meta_graph(in_model + ".meta", clear_devices=True)
    global_vars = tf.global_variables()

    for xtensor in global_vars:
        final_t = np.mean([xreader.get_tensor(xtensor.name.split(":")[0]) for xreader in readers], axis=0)
        xtensor.load(final_t, session=sess)

    saver = tf.train.Saver(global_vars)
    saver.save(sess, os.path.join(save_folder, "inference_model"))

    # copy flags
    ref_falgs = os.path.join(os.path.dirname(in_models[-1]), "model_flags.json")
    shutil.copy(ref_falgs, save_folder)
    print("We are done!")
