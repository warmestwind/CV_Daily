# to get model.pb
import tensorflow as tf
from custom_estimator import csv_input_fn, create_estimator
import itertools
from meta_data_proc import process_features

# Args:
MODEL_NAME = 'reg-model-04'
TRAIN_SIZE = 12000
NUM_EPOCHS = 1
BATCH_SIZE = 200
NUM_EVAL = 10
CHECKPOINT_STEPS = int((TRAIN_SIZE/BATCH_SIZE) * (NUM_EPOCHS/NUM_EVAL))
CHECKPOINT_STEPS =int (TRAIN_SIZE/BATCH_SIZE)
hparams  = tf.contrib.training.HParams(
    num_epochs = NUM_EPOCHS,
    batch_size = BATCH_SIZE,
    hidden_units=[8, 4], 
    dropout_prob = 0.0)

model_dir = 'trained_models/{}'.format(MODEL_NAME)

run_config = tf.contrib.learn.RunConfig(
    save_checkpoints_steps=CHECKPOINT_STEPS,
    tf_random_seed=19830610,
    model_dir=model_dir
)

# Method 2: Predict by estimator 
estimator = create_estimator(run_config, hparams)


TRAIN_DATA_FILES_PATTERN = './data/train-*.csv'
def train_input_fn(): return csv_input_fn(files_name_pattern=TRAIN_DATA_FILES_PATTERN,
                                            mode=tf.estimator.ModeKeys.TRAIN,
                                            batch_size=200, num_epochs=1)

estimator.train(input_fn=train_input_fn)

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


estimator.export_savedmodel('F:\SourceCode\Python\Estimator\savedmodel',csv_serving_input_fn)

