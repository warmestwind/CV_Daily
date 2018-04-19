import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow import data
from tensorflow.python.feature_column import feature_column

from meta_data_proc import parse_csv_row , process_features
from meta_data_proc import get_feature_columns


#Estimator input function
def csv_input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL, 
                 skip_header_lines=0, 
                 num_epochs=None, 
                 batch_size=1, kwargs={'flag_proc':True}):
    '''get a itertor of dataset
    '''
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    
    print("")
    print("* data input_fn:")
    print("================")
    print("Input file(s): {}".format(files_name_pattern))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")

    #read csv file
    file_names = tf.matching_files(files_name_pattern)
    #ot = file_names.eval()
    dataset = data.TextLineDataset(filenames=file_names)
    dataset = dataset.skip(skip_header_lines)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)

    #process csv row
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda csv_row: parse_csv_row(csv_row)) #通过dataset的map函数对每行数据进行解析
    #[features, target] = dataset

    PROCESS_FEATURES = kwargs['flag_proc']

    if PROCESS_FEATURES:
        dataset = dataset.map(lambda features, target: (process_features(features), target))
    
    dataset = dataset.repeat(num_epochs) #for num epochs

    #iterator = tfe.Iterator(dataset)
    #features, target = iterator.next()
    iterator = dataset.make_one_shot_iterator()
    features, target = iterator.get_next()

    return features, target



#Estimator model function
def regression_model_fn(features, labels, mode, params):

    hidden_units = params.hidden_units
    output_layer_size = 1

    feature_columns = list(get_feature_columns(flag_proc= True, \
                                flag_extend= True ).values())

    dense_columns = list(
        filter(lambda column: isinstance(column, feature_column._NumericColumn),
               feature_columns
        )
    )

    categorical_columns = list(
        filter(lambda column: isinstance(column, feature_column._VocabularyListCategoricalColumn) |
                              isinstance(column, feature_column._BucketizedColumn),
                   feature_columns)
    )
    # indicator <-- categorical  : dense_tensor == [[2, 0, 0]]  # If "name" bytes_list is ["bob", "bob"]
    # multi-hot representation of given categorical column
    indicator_columns = list(
            map(lambda column: tf.feature_column.indicator_column(column),
                categorical_columns)
    )


    # Create the input layers from the features
    # features : dict , feature_columns will find the key of features then creat input layer tensor
    #input_layer args: 
    #feauture:dict {key:tensor},
    #feature_columns:[columns]
    #feauture_clumns look up feauture's key to construct input layer tensor
    input_layer = tf.feature_column.input_layer(features= features, 
                                                feature_columns= dense_columns+indicator_columns)


#     # Create only 1 hidden layer based on the first element of the hidden_units in the params
#     hidden_layer_size = hidden_units[0]
#     hidden_layer = tf.layers.dense(inputs= input_layer, 
#                                    units=hidden_layer_size, 
#                                    activation=tf.nn.relu)

    # Create a fully-connected layer-stack based on the hidden_units in the params
    hidden_layers = tf.contrib.layers.stack(inputs= input_layer,
                                            layer= tf.contrib.layers.fully_connected,
                                            stack_args= hidden_units) # num output

    # Connect the output layer (logits) to the hidden layer (no activation fn)
    logits = tf.layers.dense(inputs=hidden_layers, 
                             units=output_layer_size)

    # Reshape output layer to 1-dim Tensor to return predictions
    output = tf.squeeze(logits) #for batch input

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:

        predictions = {
            'scores': output
            # if model is cnn return as follow:
            #'classes': tf.argmax(logits, axis=1),
            #'probabilities': tf.nn.softmax(logits),
        }

        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(
            mode=mode, #specify the mode to be run in estimator
            predictions=predictions,
            export_outputs=export_outputs)

    # train and eval
    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, output)

    # Create Optimiser
    optimizer = tf.train.AdamOptimizer()

    # Create training operation
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            labels, output)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    estimator_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                loss=loss,
                                                train_op=train_op, 
                                                # train_op will be ignored in eval and pedict modes
                                                eval_metric_ops=eval_metric_ops#,
                                                # here we add a train hook to early stop which is a iterable object
                                                #training_hooks = [EarlyStoppingHook()]
                                                )
    return estimator_spec


#Estimator constuctor
def create_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=regression_model_fn, 
                                  params=hparams, 
                                  config=run_config)
    
    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")

    return estimator


class EarlyStoppingHook(tf.train.SessionRunHook):
    
    def __init__(self, early_stopping_rounds=1):
        self._best_loss = None
        self._early_stopping_rounds = early_stopping_rounds
        self._counter = 0
        
        print("")
        print("*** Early Stopping Hook: - Created")
        print("*** Early Stopping Hook:: Early Stopping Rounds: {}".format(self._early_stopping_rounds))
        print("")

    def before_run(self, run_context):
        
        graph = run_context.session.graph
        
#         tensor_name = "dnn/head/weighted_loss/Sum:0" #works!!
#         loss_tensor = graph.get_tensor_by_name(tensor_name)

        loss_tensor = graph.get_collection(tf.GraphKeys.LOSSES)[0]
        return tf.train.SessionRunArgs(loss_tensor)

    def after_run(self, run_context, run_values):
        
        last_loss = run_values.results
        
        print("")
        print("************************")
        print("** Evaluation Monitor - Early Stopping **")
        print("-----------------------------------------")
        print("Early Stopping Hook: Current loss: {}".format(str(last_loss)))
        print("Early Stopping Hook: Best loss: {}".format(str(self._best_loss)))

        if self._best_loss is None:
            self._best_loss = last_loss
            
        elif last_loss > self._best_loss:
            
            self._counter += 1
            print("Early Stopping Hook: No improvment! Counter: {}".format(self._counter))
            
            if self._counter == self._early_stopping_rounds:
                
                run_context.request_stop()
                print("Early Stopping Hook: Stop Requested: {}".format(run_context.stop_requested))
        else:
            
            self._best_loss = last_loss
            self._counter = 0
            
        print("************************")
        print("") 
