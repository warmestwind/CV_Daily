import tensorflow as tf 

# ## 1. Define Dataset Metadata
# * CSV file header and defaults
# * Numeric and categorical feature names
# * Target feature name
# * Unused columns

HEADER = ['key','x','y','alpha','beta','target']
HEADER_DEFAULTS = [[0], [0.0], [0.0], ['NA'], ['NA'], [0.0]]

NUMERIC_FEATURE_NAMES = ['x', 'y']  

CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'alpha':['ax01', 'ax02'], 'beta':['bx01', 'bx02']}
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) #[alapa,beta]

FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

TARGET_NAME = 'target'

UNUSED_FEATURE_NAMES = list(set(HEADER) - set(FEATURE_NAMES) - {TARGET_NAME}) #key

print("Header: {}".format(HEADER))
print("Numeric Features: {}".format(NUMERIC_FEATURE_NAMES))
print("Categorical Features: {}".format(CATEGORICAL_FEATURE_NAMES))
print("Target: {}".format(TARGET_NAME))
print("Unused Features: {}".format(UNUSED_FEATURE_NAMES))
print("FEATURE_NAMES: {}".format(FEATURE_NAMES))


# ## 2. Define Data Input Function
# * Input csv files name pattern
# * Use TF Dataset APIs to read and process the data
# * Parse CSV lines to feature tensors
# * Apply feature processing
# * Return (features, target) tensors

# ### a. parsing and preprocessing logic

def parse_csv_row(csv_row):
    '''covert one csv line to a dict contain needed feature tensor
    '''
    columns = tf.decode_csv(csv_row, record_defaults=HEADER_DEFAULTS)
    features = dict(zip(HEADER, columns)) #dict(header: column)
    
    for column in UNUSED_FEATURE_NAMES:
        features.pop(column)
    
    target = features.pop(TARGET_NAME) #pop return TARGET_NAME

    return features, target #feature not contain 'key' and 'target'

def process_features(features):
    '''extend feature
    '''
    features["x_2"] = tf.square(features['x'])
    features["y_2"] = tf.square(features['y'])
    features["xy"] = tf.multiply(features['x'], features['y']) # features['x'] * features['y']
    features['dist_xy'] =  tf.sqrt(tf.squared_difference(features['x'],features['y']))
    
    return features



#Bridge(feature_column) between Estimator and raw data  
# ## 3. Define Feature Columns
# The input numeric columns are assumed to be normalized (or have the same scale). 
# Otherwise, a normlizer_fn, along with the normlisation params (mean, stdv or min, max) 
# should be passed to tf.feature_column.numeric_column() constructor.

def extend_feature_columns(feature_columns):
    
    # crossing, bucketizing, and embedding can be applied here
    
    feature_columns['alpha_X_beta'] = tf.feature_column.crossed_column(
        [feature_columns['alpha'], feature_columns['beta']], 4)
    
    return feature_columns

def get_feature_columns(**kwargs):
    '''construct feature column
    '''
    CONSTRUCTED_NUMERIC_FEATURES_NAMES = ['x_2', 'y_2', 'xy', 'dist_xy']
    all_numeric_feature_names = NUMERIC_FEATURE_NAMES.copy() 
    
    PROCESS_FEATURES = kwargs['flag_proc']
    if PROCESS_FEATURES:
        all_numeric_feature_names += CONSTRUCTED_NUMERIC_FEATURES_NAMES

    #numeric_column can add normalizer_fn
    numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name)
                       for feature_name in all_numeric_feature_names}

    categorical_column_with_vocabulary = {item[0]: tf.feature_column.categorical_column_with_vocabulary_list \
        (item[0], item[1]) for item in CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}
        # item[0]=key= 'alpha', item[1] = vocabulary_list = ['ax01', 'ax02']
    feature_columns = {}

    if numeric_columns is not None:
        feature_columns.update(numeric_columns)

    if categorical_column_with_vocabulary is not None:
        feature_columns.update(categorical_column_with_vocabulary)

    EXTEND_FEATURE_COLUMNS=kwargs['flag_extend']
    if EXTEND_FEATURE_COLUMNS:
        feature_columns = extend_feature_columns(feature_columns)
        
    return feature_columns
