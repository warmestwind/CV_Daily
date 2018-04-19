import tensorflow as tf
from custom_estimator import csv_input_fn, create_estimator

#Args:

MODEL_NAME = 'reg-model-04'

TRAIN_DATA_FILES_PATTERN = './data/train-*.csv'
VALID_DATA_FILES_PATTERN = './data/valid-*.csv'
TEST_DATA_FILES_PATTERN = './data/test-*.csv'

RESUME_TRAINING = False
PROCESS_FEATURES = True
EXTEND_FEATURE_COLUMNS = True
MULTI_THREADING = True

TRAIN_SIZE = 12000
NUM_EPOCHS = 1000
BATCH_SIZE = 500
NUM_EVAL = 10
CHECKPOINT_STEPS = int((TRAIN_SIZE/BATCH_SIZE) * (NUM_EPOCHS/NUM_EVAL))

TRAIN_SIZE = 12000
VALID_SIZE = 3000
TEST_SIZE = 5000

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

def train_input_fn(): return csv_input_fn(files_name_pattern=TRAIN_DATA_FILES_PATTERN,
                                          mode=tf.estimator.ModeKeys.EVAL,
                                          batch_size=TRAIN_SIZE)


def valid_input_fn(): return csv_input_fn(files_name_pattern=VALID_DATA_FILES_PATTERN,
                                          mode=tf.estimator.ModeKeys.EVAL,
                                          batch_size=VALID_SIZE)


def test_input_fn(): return csv_input_fn(files_name_pattern=TEST_DATA_FILES_PATTERN,
                                         mode=tf.estimator.ModeKeys.EVAL,
                                         batch_size=TEST_SIZE)


estimator = create_estimator(run_config, hparams)

train_results = estimator.evaluate(input_fn=train_input_fn, steps=1)
train_rmse = str(train_results["rmse"])
print()
print("######################################################################################")
print("# Train RMSE: {} - {}".format(train_rmse, train_results))
print("######################################################################################")

valid_results = estimator.evaluate(input_fn=valid_input_fn, steps=1)
valid_rmse = str(valid_results["rmse"])
print()
print("######################################################################################")
print("# Valid RMSE: {} - {}".format(valid_rmse, valid_results))
print("######################################################################################")

test_results = estimator.evaluate(input_fn=test_input_fn, steps=1)
test_rmse = str(test_results["rmse"])
print()
print("######################################################################################")
print("# Test RMSE: {} - {}".format(test_rmse, test_results))
print("######################################################################################")
