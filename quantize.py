"""
Quatize or convert to float16 your model weights.
"""

import argparse
import numpy as np
import tensorflow as tf
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
import os
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser(description='Quantization of graphs.')
parser.add_argument('--min_elements', action="store", type=int, default=17000,
                    help="Only variables with more than this many elements will be converted.")
parser.add_argument('--transform_type', action="store", type=str, required=True, choices=["float16",
                                                                                          "quant_uniform",
                                                                                          "quant_kmeans",
                                                                                          "quant_quantile"])
parser.add_argument('--savefile', action="store", type=str, required=True)
parser.add_argument('--model', action="store", type=str, required=True,
                    help="Input_model. models must be type '/path/to/model/inference_model'")

params = parser.parse_args()
transform_type = params.transform_type
min_elements = params.min_elements
model = params.model
output_model = params.savefile
assert transform_type in ["float16", "quant_uniform", "quant_kmeans", "quant_quantile"], "Unknown transformation!"
assert os.path.isfile(model + ".meta"), "Input model does not exist!"
assert not os.path.isfile(output_model + ".meta"), "Model output exsist will not overwrite!"

if not os.path.isdir(os.path.dirname(output_model)):
    os.makedirs(os.path.dirname(output_model))


# The code
cast_variables = {}  # variables that will be converted to float16
load_variables = {}  # variables that will be loaded as they are
name_to_tensor_map = {}

# Load graph
xsaver = tf.train.import_meta_graph(model + ".meta", clear_devices=True)
fetch_vars = tf.global_variables()

# prepare the load
for c_var in fetch_vars:
    name_to_tensor_map[c_var.op.name] = c_var
    try:
        n_elements = reduce(lambda x, y: x * y, c_var.shape.as_list())
    except:
        n_elements = 1
    if (n_elements > min_elements) and (c_var.dtype.as_numpy_dtype == np.float32):
        cast_variables[c_var.op.name] = tf.contrib.framework.load_variable(model, c_var.op.name)
    else:
        load_variables[c_var.op.name] = tf.contrib.framework.load_variable(model, c_var.op.name)

# replace editor
ge = tf.contrib.graph_editor

# create mappings
loadings = []

# Make loading and casting of big matricies
print("Quantizing variables.")
for var_name, c_np in tqdm(cast_variables.items()):
    tf_var = name_to_tensor_map[var_name]
    if transform_type == "float16":  # Float16 transformation
        new_var = tf.Variable(tf.zeros(c_np.shape, tf.float16),
                                      dtype=tf.float16,
                                      name=var_name + "float16")
        ge.swap_inputs(tf_var.value(), tf.cast(new_var, tf.float32))
        loadings.append([new_var, c_np.astype(np.float16)])

    elif transform_type == "quant_uniform":
        quant_var = tf.Variable(tf.zeros(c_np.shape, tf.uint8),
                                dtype=tf.uint8,
                                name=var_name + "_quants")
        space_var = tf.Variable(tf.zeros(256, tf.float32),
                                dtype=tf.float32,
                                name=var_name + "_space")

        output = tf.gather(space_var, tf.cast(quant_var, tf.int32))

        ge.swap_inputs(tf_var.value(), output)

        mat_shape = c_np.shape
        c_np = c_np.flatten()
        space = np.linspace(c_np.min(), c_np.max(), num=256)
        quants = vq(c_np, space)[0].reshape(mat_shape)

        loadings.append([quant_var, quants.astype(np.uint8)])
        loadings.append([space_var, space.astype(np.float32)])

    elif transform_type == "quant_kmeans":

        quant_var = tf.Variable(tf.zeros(c_np.shape, tf.uint8),
                                dtype=tf.uint8,
                                name=var_name + "_quants")
        space_var = tf.Variable(tf.zeros(256, tf.float32),
                                dtype=tf.float32,
                                name=var_name + "_space")

        output = tf.gather(space_var, tf.cast(quant_var, tf.int32))

        ge.swap_inputs(tf_var.value(), output)

        mat_shape = c_np.shape
        c_np = c_np.flatten()

        # K-means difference
        km_pred = KMeans(n_clusters=256, n_jobs=-1)
        # Note: Sklearn has a bug with high-dimensional float32's.
        km_pred.fit(c_np[:, np.newaxis].astype(np.float64))

        space = sorted(km_pred.cluster_centers_.flatten())
        quants = vq(c_np, space)[0].reshape(mat_shape)

        loadings.append([quant_var, quants.astype(np.uint8)])
        loadings.append([space_var, space.astype(np.float32)])

    elif transform_type == "quant_quantile":
        quant_var = tf.Variable(tf.zeros(c_np.shape, tf.uint8),
                                dtype=tf.uint8,
                                name=var_name + "_quants")
        space_var = tf.Variable(tf.zeros(256, tf.float32),
                                dtype=tf.float32,
                                name=var_name + "_space")
        output = tf.gather(space_var, tf.cast(quant_var, tf.int32))
        ge.swap_inputs(tf_var.value(), output)

        mat_shape = c_np.shape
        c_np = c_np.flatten()

        ranges = np.linspace(0., 1., 256, endpoint=True)
        centers = pd.Series(c_np).quantile(ranges)

        space = np.array(centers)
        quants = vq(c_np, space)[0].reshape(mat_shape)

        loadings.append([quant_var, quants.astype(np.uint8)])
        loadings.append([space_var, space.astype(np.float32)])

    else:
        raise

# We load those as they are
for var_name, c_np in load_variables.items():
    tf_var = name_to_tensor_map[var_name]
    loadings.append([tf_var, c_np])

# Load all Variables into session.
sess = tf.Session()
for load_dest, np_val in loadings:
    load_dest.load(np_val, session=sess)


saver = tf.train.Saver(var_list=[x[0] for x in loadings])
saver.save(sess, output_model)
