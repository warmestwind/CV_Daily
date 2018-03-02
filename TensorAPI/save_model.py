import tensorflow as tf
import shutil
import os 
# Store-------------------
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)
v3 = tf.get_variable("v3", shape=[6], initializer = tf.zeros_initializer)
#error: must specify variable name
#v4 = tf.get_variable(shape=[6], initializer = tf.zeros_initializer)
v5 = tf.get_variable("v5", shape=[2], initializer = tf.ones_initializer)

unknown_var ={'unknown' : tf.saved_model.utils.build_tensor_info(v5)}

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)
dec_v3 = v3.assign(v3-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Another func to save a model
saved_model_dir = './savertmp/modelfile'
os.path.exists(saved_model_dir)
shutil.rmtree(saved_model_dir , ignore_errors=True)
builder = tf.saved_model.builder.SavedModelBuilder('./savertmp/modelfile')

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  dec_v3.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "./savertmp/model.ckpt")
  print("Model saved in path: %s" % save_path)

  # signature_def_map, inputs & outputs are TensorInfo 
  signature_map = tf.saved_model.signature_def_utils.build_signature_def(inputs =unknown_var,
                              outputs = None, method_name = 'sig_map')

  # Adds the current meta graph to the SavedModel and saves variables.
  builder.add_meta_graph_and_variables(sess, ['tag_string'], 
    signature_def_map={'test_signature':signature_map}) #for decoupling
builder.save()
  #then in modelfile/ will generate 
  # variables file : equals model.ckpt.data
  # saved_model.pb : equals model.ckpt.meta

# Restore-------------------
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])
v4 = tf.get_variable("v3", shape=[6])
# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver({"v2": v2, "v3":v4})
                        #saved key : variable name，v3 in check->v4 variable
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  '''
  # Non-restored variables need initializer
  v1.initializer.run()
  # Restore variables from disk.
  saver.restore(sess, "./savertmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
  print("v3 : %s" % v4.eval())
  print('keys = ', sess.graph.get_all_collection_keys()) #['variables', 'trainable_variables']
  print('value = ', sess.graph.get_collection('variables'))
  print('v1 =', sess.graph.get_tensor_by_name('v1:0').eval())
  #Tensor names must be of the form "<op_name>:<output_index>".  e.g. 'input_x:0'
  '''
  # Another func: Load savedmodel
  saved_model_dir = './savertmp/modelfile'
  meta_graph_def = tf.saved_model.loader.load(sess, ['tag_string'], saved_model_dir)

  # Use signature_def_map to load a input/out tensor
  # get signature instance for get tensor name from it
  signature = meta_graph_def.signature_def
  # get tensor name  
  unkown_var = signature['test_signature'].inputs['unknown'].name
  print('unkown_var_name =', sess.graph.get_tensor_by_name(unkown_var))
  # must initialize
  un_var = sess.graph.get_tensor_by_name(unkown_var)
  sess.run(tf.global_variables_initializer())
  print('unkown_var_value =', sess.run(un_var))

# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("./savertmp/model.ckpt", 
                tensor_name='', all_tensors=True, all_tensor_names= True)



