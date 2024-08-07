import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import reduce_memory_usage
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger


class preprocess_credit_card_balance:
    '''
    Preprocess the credit_card_balance table.

    This class contains methods to load, preprocess, and aggregate the `credit_card_balance` table.

    '''

    def __init__(self, file_directory: str = '', verbose: bool = True, dump_to_pickle: bool = False):
        '''
        Initializes the preprocess_credit_card_balance class.

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
        Loads the `credit_card_balance.csv` DataFrame into memory.

        Returns:
            None
        '''
        if self.verbose:
            logger.info('#########################################################')
            logger.info('#        Pre-processing credit_card_balance.csv         #')
            logger.info('#########################################################')
            logger.info("Loading the DataFrame, credit_card_balance.csv, into memory...")

        self.cc_balance = reduce_memory_usage(pd.read_csv(self.file_directory + 'credit_card_balance.csv'))
        self.initial_size = self.cc_balance.shape

        if self.verbose:
            logger.info("Loaded credit_card_balance.csv")
            logger.info('Time Taken to load: {}', datetime.now() - self.start)

    def data_preprocessing_and_feature_engineering(self):
        '''
        Performs preprocessing and feature engineering on the DataFrame.

        Returns:
            None
        '''
        if self.verbose:
            start = datetime.now()
            logger.info("Starting Data Pre-processing and Feature Engineering...")

        # Example of preprocessing and feature engineering
        # self.cc_balance['new_feature'] = self.cc_balance['feature_1'] / self.cc_balance['feature_2']

        if self.verbose:
            logger.info("Data Pre-processing and Feature Engineering Done.")
            logger.info('Time Taken: {}', datetime.now() - start)

    def aggregations(self) -> pd.DataFrame:
        '''
        Aggregates the `credit_card_balance` table first over `SK_ID_PREV`, and then over `SK_ID_CURR`.

        Returns:
            pd.DataFrame: Aggregated `credit_card_balance` table.
        '''
        if self.verbose:
            logger.info("Aggregating the DataFrame, first over SK_ID_PREV, then over SK_ID_CURR")

        # Combining numerical features
        cc_numerical_aggregated = (
            self.cc_balance.select_dtypes(include=[np.number])
            .drop('SK_ID_PREV', axis=1)
            .groupby(by=['SK_ID_CURR'])
            .mean()
            .reset_index()
        )

        # Combining categorical features
        cc_categorical = pd.get_dummies(self.cc_balance.select_dtypes('object'))
        cc_categorical['SK_ID_CURR'] = self.cc_balance['SK_ID_CURR']
        cc_categorical_aggregated = cc_categorical.groupby('SK_ID_CURR').mean().reset_index()

        # Merge numerical and categorical features
        cc_aggregated = cc_numerical_aggregated.merge(cc_categorical_aggregated, on='SK_ID_CURR')
        cc_aggregated.columns = [
            'CC_' + column if column != 'SK_ID_CURR' else column for column in cc_aggregated.columns
        ]
        cc_aggregated.fillna(0, inplace=True)

        if self.verbose:
            logger.info('Aggregation Done.')
            logger.info('Size after aggregation: {}', cc_aggregated.shape)

        return cc_aggregated

    def main(self) -> pd.DataFrame:
        '''
        Performs complete preprocessing and aggregation of the `credit_card_balance` table.

        Returns:
            pd.DataFrame: Final preprocessed and aggregated `credit_card_balance` table.
        '''
        # Loading the DataFrame
        self.load_dataframe()

        # Performing preprocessing and feature engineering
        self.data_preprocessing_and_feature_engineering()

        # Aggregating the `credit_card_balance` over SK_ID_PREV and SK_ID_CURR
        cc_aggregated = self.aggregations()

        if self.verbose:
            logger.info('Done preprocessing credit_card_balance.')
            logger.info('Initial Size of credit_card_balance: {}', self.initial_size)
            logger.info(
                'Size of credit_card_balance after Pre-Processing, Feature Engineering and Aggregation: {}',
                cc_aggregated.shape,
            )
            logger.info('Total Time Taken: {}', datetime.now() - self.start)

        if self.dump_to_pickle:
            if self.verbose:
                logger.info('Pickling pre-processed credit_card_balance to credit_card_balance_preprocessed.pkl')
            with open(self.file_directory + 'credit_card_balance_preprocessed.pkl', 'wb') as f:
                pickle.dump(cc_aggregated, f)
            if self.verbose:
                logger.info('Pickling completed.')

        return cc_aggregated
