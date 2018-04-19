import tensorflow as tf
from custom_estimator import csv_input_fn, create_estimator
from meta_data_proc import process_features

TRAIN_DATA_FILES_PATTERN = './data/train-*.csv'
VALID_DATA_FILES_PATTERN = './data/valid-*.csv'
TEST_DATA_FILES_PATTERN = './data/test-*.csv'

#Define Experiment Function for lean_runner:

def generate_experiment_fn(**experiment_args):

    def _experiment_fn(run_config, hparams):

        train_input_fn = lambda: csv_input_fn(
            TRAIN_DATA_FILES_PATTERN,
            mode = tf.estimator.ModeKeys.TRAIN,
            num_epochs=hparams.num_epochs,#多轮训练
            batch_size=hparams.batch_size
        )

        eval_input_fn = lambda: csv_input_fn(
            VALID_DATA_FILES_PATTERN,
            mode=tf.estimator.ModeKeys.EVAL,
            num_epochs=1, #一轮评价
            batch_size=hparams.batch_size
        )

        estimator = create_estimator(run_config, hparams)

        return tf.contrib.learn.Experiment(
            estimator,  # Estimator
            train_input_fn=train_input_fn,  #estimator input_fn for training and evaluation
            eval_input_fn=eval_input_fn,
            eval_steps=None,
            **experiment_args
        )

    return _experiment_fn


#Define Serving Function for lean_runner:

def csv_serving_input_fn():
    
    SERVING_HEADER = ['x','y','alpha','beta']
    SERVING_HEADER_DEFAULTS = [[0.0], [0.0], ['NA'], ['NA']]

    rows_string_tensor = tf.placeholder(dtype=tf.string,
                                         shape=[None],
                                         name='csv_rows')
    
    receiver_tensor = {'csv_rows': rows_string_tensor}

    row_columns = tf.expand_dims(rows_string_tensor, -1) #add last dim =1
    columns = tf.decode_csv(row_columns, record_defaults=SERVING_HEADER_DEFAULTS) #每列一个tensor
    features = dict(zip(SERVING_HEADER, columns))

    return tf.estimator.export.ServingInputReceiver(
        process_features(features), receiver_tensor)
        #features to model         #placeholder
