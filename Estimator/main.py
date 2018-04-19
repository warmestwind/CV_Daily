from datetime import datetime
import shutil

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.learn import make_export_strategy

from meta_data_proc import get_feature_columns
from custom_estimator import csv_input_fn, create_estimator, EarlyStoppingHook
from experiment import generate_experiment_fn, csv_serving_input_fn

#tf model select
#tfe.enable_eager_execution()
#sess= tf.InteractiveSession()

#Args:

MODEL_NAME = 'reg-model-04'

TRAIN_DATA_FILES_PATTERN = './data/train-*.csv'
VALID_DATA_FILES_PATTERN = './data/valid-*.csv'
TEST_DATA_FILES_PATTERN = './data/test-*.csv'

RESUME_TRAINING = False
PROCESS_FEATURES = True
EXTEND_FEATURE_COLUMNS = True
MULTI_THREADING = True

# Input pipe line:

features, target = csv_input_fn(files_name_pattern= TEST_DATA_FILES_PATTERN ,kwargs={'flag_proc':PROCESS_FEATURES})
print("Feature read from CSV: {}".format(list(features.keys())))
print("Target read from CSV: {}".format(target))
#print(features)

#Feature column:
#feature_columns in this is a dict{feature name: feature_column}

feature_columns = get_feature_columns(flag_proc=PROCESS_FEATURES, flag_extend=EXTEND_FEATURE_COLUMNS )
print("Feature Columns: {}".format(feature_columns))

#Hparams and Configs :

TRAIN_SIZE = 12000
NUM_EPOCHS = 2
BATCH_SIZE = 500
NUM_EVAL = 10
CHECKPOINT_STEPS = int((TRAIN_SIZE/BATCH_SIZE) * (NUM_EPOCHS/NUM_EVAL))
EVAL_AFTER_SEC = 5
TOTAL_STEPS = (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS+1#49

hparams  = tf.contrib.training.HParams(
    num_epochs = NUM_EPOCHS,
    batch_size = BATCH_SIZE,
    hidden_units=[8, 4],
    max_steps = TOTAL_STEPS, 
    dropout_prob = 0.0)

model_dir = 'trained_models/{}'.format(MODEL_NAME)

run_config = tf.contrib.learn.RunConfig(
    save_checkpoints_steps=CHECKPOINT_STEPS,
    tf_random_seed=19830610,
    model_dir=model_dir
)

#Print training config 
print(hparams)
print("Model Directory:", run_config.model_dir)
print("")
print("Dataset Size:", TRAIN_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Steps per Epoch:",TRAIN_SIZE/BATCH_SIZE)
print("Total Steps:", (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS)
print("Required Evaluation Steps:", NUM_EVAL) 
print("That is 1 evaluation step after each",NUM_EPOCHS/NUM_EVAL," epochs")
print("Save Checkpoint After",CHECKPOINT_STEPS,"steps")


#Run Experiment via learn_runner:

if not RESUME_TRAINING:
    print("Removing previous artifacts...")
    shutil.rmtree(model_dir, ignore_errors=True)
else:
    print("Resuming training...") 

tf.logging.set_verbosity(tf.logging.INFO)

#time_start = datetime.utcnow() 
time_start = datetime.now() 
print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................") 

# learn_runner.run(
#     experiment_fn=generate_experiment_fn(
#         #导出配置
#         export_strategies=[make_export_strategy(
#             csv_serving_input_fn, # InputFnOp
#             #as_text
#             exports_to_keep=1
#         )] # retrun  ExportStrategy for Experiment
#     ), # return Experiment
#     run_config=run_config,
#     schedule="train_and_evaluate", #experiment method: train,test,train_and_evaluate
#     hparams=hparams
# )

estimator = create_estimator(run_config, hparams) #estimator 定义model_fn, run config, hparam

train_spec = tf.estimator.TrainSpec(
    input_fn = lambda: csv_input_fn(
        TRAIN_DATA_FILES_PATTERN,
        mode = tf.estimator.ModeKeys.TRAIN,
        num_epochs=hparams.num_epochs,
        batch_size=hparams.batch_size
    ),
    max_steps=hparams.max_steps, # total train step
    hooks=None
)

eval_spec = tf.estimator.EvalSpec(
    input_fn = lambda: csv_input_fn(
        VALID_DATA_FILES_PATTERN,
        mode=tf.estimator.ModeKeys.EVAL,
        num_epochs=1, 
        batch_size=hparams.batch_size
            
    ),
    hooks=[EarlyStoppingHook(20)], #early stop in validation set
    throttle_secs = 50,#EVAL_AFTER_SEC, #after training EVAL_AFTER_SEC seconds then do evaluation, but must have new checkpoint
    steps=None #total valid step , computed same as train, 
               #here valid data size is 3000, batch size is 500, so step is 6 to valid all valid dataset
)

tf.estimator.train_and_evaluate(
    estimator=estimator,
    train_spec=train_spec, 
    eval_spec=eval_spec
)


time_end = datetime.now() 
print(".......................................")
print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
print("")
time_elapsed = time_end - time_start
print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))













