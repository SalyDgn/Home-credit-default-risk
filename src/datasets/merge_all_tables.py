import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re

from loguru import logger

from utils import reduce_memory_usage


def merge_all_tables(
    application_train,
    application_test,
    bureau_aggregated,
    previous_aggregated,
    installments_aggregated,
    pos_aggregated,
    cc_aggregated,
):
    '''
    Function to merge all the tables together with the application_train and application_test tables
    on SK_ID_CURR.

    Inputs:
        All the previously pre-processed Tables.

    Returns:
        Single merged tables, one for training data and one for test data
    '''

    logger.info("Merging application_train and application_test with aggregated tables.")

    # merging application_train and application_test with Aggregated bureau table
    app_train_merged = application_train.merge(bureau_aggregated, on='SK_ID_CURR', how='left')
    app_test_merged = application_test.merge(bureau_aggregated, on='SK_ID_CURR', how='left')
    logger.info("Merged with bureau_aggregated.")

    # merging with aggregated previous_applications
    app_train_merged = app_train_merged.merge(previous_aggregated, on='SK_ID_CURR', how='left')
    app_test_merged = app_test_merged.merge(previous_aggregated, on='SK_ID_CURR', how='left')
    logger.info("Merged with previous_aggregated.")

    # merging with aggregated installments tables
    app_train_merged = app_train_merged.merge(installments_aggregated, on='SK_ID_CURR', how='left')
    app_test_merged = app_test_merged.merge(installments_aggregated, on='SK_ID_CURR', how='left')
    logger.info("Merged with installments_aggregated.")

    # merging with aggregated POS_Cash balance table
    app_train_merged = app_train_merged.merge(pos_aggregated, on='SK_ID_CURR', how='left')
    app_test_merged = app_test_merged.merge(pos_aggregated, on='SK_ID_CURR', how='left')
    logger.info("Merged with pos_aggregated.")

    # merging with aggregated credit card table
    app_train_merged = app_train_merged.merge(cc_aggregated, on='SK_ID_CURR', how='left')
    app_test_merged = app_test_merged.merge(cc_aggregated, on='SK_ID_CURR', how='left')

    # Filling missing values with 0
    app_train_merged = app_train_merged.fillna(0)
    app_test_merged = app_test_merged.fillna(0)
    logger.info("Filled missing values with 0.")

    # Clean column names to remove non-alphanumeric characters
    app_train_merged = app_train_merged.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    app_test_merged = app_test_merged.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    logger.info("Cleaned column names.")

    # Ensure the columns are the same for train and test data
    train_cols = set(app_train_merged.columns)
    test_cols = set(app_test_merged.columns)

    diff_train_cols = train_cols.difference(test_cols)
    diff_test_cols = test_cols.difference(train_cols)

    if 'TARGET' in diff_train_cols:
        diff_train_cols.remove('TARGET')

    app_train_merged.drop(diff_train_cols, axis=1, inplace=True)
    app_test_merged.drop(diff_test_cols, axis=1, inplace=True)
    logger.info("Ensured train and test data have the same columns.")

    # removing the SK_ID_CURR from training and test data
    app_train_merged = app_train_merged.drop(['SK_ID_CURR'], axis=1)
    app_test_merged = app_test_merged.drop(['SK_ID_CURR'], axis=1)
    logger.info("Removed SK_ID_CURR from the data.")

    # Reduce memory usage
    app_train_merged = reduce_memory_usage(app_train_merged)
    app_test_merged = reduce_memory_usage(app_test_merged)
    logger.info("Reduced memory usage of the merged tables.")

    return app_train_merged, app_test_merged
