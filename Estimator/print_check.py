#https://tensorflow.google.cn/programmers_guide/saved_model
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow

#342 is global train step
chkp.print_tensors_in_checkpoint_file('trained_models/reg-model-04/model.ckpt-342',
                                        tensor_name='',all_tensors =True,all_tensor_names=True)

reader = pywrap_tensorflow.NewCheckpointReader('trained_models/reg-model-04/model.ckpt-342')
var_to_shape_map = reader.get_variable_to_shape_map()
for key in sorted(var_to_shape_map):
    #print all tensor name
    print("tensor_name: ", key)
    #print all tensor or specified tensor
    print(reader.get_tensor(key))
    print(reader.get_tensor('global_step'))
