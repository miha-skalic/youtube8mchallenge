import numpy as np
import tensorflow as tf


# parameters
min_elements = 6000
model = "../trained_models/attention_frames_v0/inference_model"
output_model = "./samplemodel/inference_model"


# The code
cast_variables = {}  # variables that will be converted to float16
load_variables = {}  # variables that will be loaded as they are
name_to_tensor_map = {}

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
# We will load those as float16
for var_name, c_np in cast_variables.items():
    tf_var = name_to_tensor_map[var_name]
    new_var = tf.Variable(tf.zeros(c_np.shape, tf.float16),
                                  dtype=tf.float16,
                                  name=var_name + "float16")
    ge.swap_inputs(tf_var.value(), tf.cast(new_var, tf.float32))
    loadings.append([new_var, c_np.astype(np.float16)])

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