import tensorflow as tf
from custom_estimator import csv_input_fn, create_estimator
import itertools
# Args:
MODEL_NAME = 'reg-model-04'
TRAIN_SIZE = 12000
NUM_EPOCHS = 1000
BATCH_SIZE = 500
NUM_EVAL = 10
CHECKPOINT_STEPS = int((TRAIN_SIZE/BATCH_SIZE) * (NUM_EPOCHS/NUM_EVAL))

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


TEST_DATA_FILES_PATTERN = './data/test-*.csv'
def predict_input_fn(): return csv_input_fn(files_name_pattern=TEST_DATA_FILES_PATTERN,
                                            mode=tf.estimator.ModeKeys.PREDICT,
                                            batch_size=5)

# predictions = estimator.predict(input_fn=predict_input_fn)
# print("")
# print(list(itertools.islice(predictions, 5)))

predictions = estimator.predict(input_fn=predict_input_fn)
print('type is ',  next(predictions).keys()) # define in model_fn's  predict EstimatorSpec 
values = list(map(lambda item: item["scores"], list(
    itertools.islice(predictions,5))))
print()
print("Predicted Values: {}".format(values))


# Method 2 Serving via the Saved Model

import os

export_dir = "savedmodel"

saved_model_dir = export_dir + "/" + os.listdir(path=export_dir)[-1]

print(saved_model_dir)


predictor_fn = tf.contrib.predictor.from_saved_model(
    export_dir=saved_model_dir,
    signature_def_key="predictions"
)

output = predictor_fn({'csv_rows': ["0.5,1,ax01,bx02", "-0.5,-1,ax02,bx02"]})
print(output)






