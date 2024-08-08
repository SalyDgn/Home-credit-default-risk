import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from utils import reduce_memory_usage


class preprocess_previous_application:
    '''
    Preprocess the previous_application table.

    This class contains methods to load, clean, preprocess, and aggregate the `previous_application` table.

    Attributes:
        file_directory (str): Path to the directory containing the data files.
        verbose (bool): Whether to enable verbose logging.
        dump_to_pickle (bool): Whether to pickle the final preprocessed table.
    '''

    def __init__(self, file_directory: str = '', verbose: bool = True, dump_to_pickle: bool = False):
        '''
        Initializes the preprocess_previous_application class.

        Args:
            file_directory (str): Path to the directory where the files are located.
            verbose (bool): Whether to enable verbose logging.
            dump_to_pickle (bool): Whether to pickle the final preprocessed table.
        '''
        self.file_directory = file_directory
        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle

        self.start = datetime.now()
        logger.info('Preprocessing class initialized.')

    def load_dataframe(self):
        '''
        Loads the `previous_application.csv` DataFrame into memory.

        Returns:
            None
        '''
        if self.verbose:
            logger.info('########################################################')
            logger.info('#        Pre-processing previous_application.csv        #')
            logger.info('########################################################')
            logger.info("Loading the DataFrame, previous_application.csv, into memory...")

        # Loading the DataFrame into memory
        self.previous_application = reduce_memory_usage(pd.read_csv(self.file_directory + 'previous_application.csv'))
        self.initial_shape = self.previous_application.shape

        if self.verbose:
            logger.info("Loaded previous_application.csv")
            logger.info('Time Taken to load: {}', datetime.now() - self.start)

    def data_cleaning(self):
        '''
        Cleans the data by removing erroneous points and filling categorical NaNs with 'XNA'.

        Returns:
            None
        '''
        if self.verbose:
            start = datetime.now()
            logger.info('Starting Data Cleaning...')

        # Example of data cleaning (customize as needed)
        # self.previous_application['column_name'].fillna('XNA', inplace=True)

        if self.verbose:
            logger.info("Data Cleaning Done.")
            logger.info('Time taken: {}', datetime.now() - start)

    def preprocessing_feature_engineering(self):
        '''
        Performs preprocessing such as categorical encoding and feature engineering.

        Returns:
            None
        '''
        if self.verbose:
            start = datetime.now()
            logger.info("Performing Preprocessing and Feature Engineering...")

        # Example of feature engineering (customize as needed)
        # self.previous_application['new_feature'] = self.previous_application['feature_1'] / self.previous_application['feature_2']

        if self.verbose:
            logger.info("Preprocessing and Feature Engineering Done.")
            logger.info('Time taken: {}', datetime.now() - start)

    def aggregations(self) -> pd.DataFrame:
        '''
        Aggregates the previous applications over `SK_ID_CURR` and merges with application_bureau.

        Returns:
            pd.DataFrame: Final DataFrame after merging and aggregations.
        '''
        if self.verbose:
            logger.info("Aggregating previous applications over SK_ID_CURR...")

        # Number of previous applications per customer
        grp = (
            self.previous_application[['SK_ID_CURR', 'SK_ID_PREV']]
            .groupby(by=['SK_ID_CURR'])['SK_ID_PREV']
            .count()
            .reset_index()
            .rename(columns={'SK_ID_PREV': 'PREV_APP_COUNT'})
        )
        self.previous_application = self.previous_application.merge(grp, on=['SK_ID_CURR'], how='right')

        # Combining numerical features
        previous_numerical_aggregated = (
            self.previous_application.select_dtypes(include=[np.number])
            .drop('SK_ID_PREV', axis=1)
            .groupby(by=['SK_ID_CURR'])
            .mean()
            .reset_index()
        )

        # Combining categorical features
        previous_categorical = pd.get_dummies(self.previous_application.select_dtypes('object'))
        previous_categorical['SK_ID_CURR'] = self.previous_application['SK_ID_CURR']
        previous_categorical_aggregated = previous_categorical.groupby('SK_ID_CURR').mean().reset_index()

        # Merge numerical and categorical features
        previous_aggregated = previous_numerical_aggregated.merge(previous_categorical_aggregated, on='SK_ID_CURR')
        previous_aggregated.columns = [
            'PREV_' + column if column != 'SK_ID_CURR' else column for column in previous_aggregated.columns
        ]
        previous_aggregated.fillna(0, inplace=True)

        if self.verbose:
            logger.info('Aggregations Done.')
            logger.info('Size after merging, preprocessing, and aggregation: {}', previous_aggregated.shape)

        return previous_aggregated

    def main(self) -> pd.DataFrame:
        '''
        Performs complete preprocessing and aggregation of the `previous_application` table.

        Returns:
            pd.DataFrame: Final preprocessed and aggregated `previous_application` table.
        '''
        # Loading the DataFrame
        self.load_dataframe()

        # Cleaning the data
        self.data_cleaning()

        # Preprocessing the categorical features and creating new features
        self.preprocessing_feature_engineering()

        # Aggregating data over SK_ID_CURR and merging with application_bureau
        previous_aggregated = self.aggregations()

        if self.verbose:
            logger.info('Done aggregations.')
            logger.info('Initial Size of previous_application: {}', self.initial_shape)
            logger.info(
                'Size of previous_application after Pre-Processing, Feature Engineering and Aggregation: {}',
                previous_aggregated.shape,
            )
            logger.info('Total Time Taken: {}', datetime.now() - self.start)

        if self.dump_to_pickle:
            if self.verbose:
                logger.info('Pickling pre-processed previous_application to previous_application_preprocessed.pkl')
            with open(self.file_directory + 'previous_application_preprocessed.pkl', 'wb') as f:
                pickle.dump(previous_aggregated, f)
            if self.verbose:
                logger.info('Pickling completed.')

        return previous_aggregated
