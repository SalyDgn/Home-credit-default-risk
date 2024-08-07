import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from utils import reduce_memory_usage

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class preprocess_installments_payments:
    '''
    Preprocess the installments_payments table.

    This class contains methods to load, preprocess, and aggregate the `installments_payments` table.

    '''

    def __init__(self, file_directory: str = '', verbose: bool = True, dump_to_pickle: bool = False):
        '''
        Initializes the preprocess_installments_payments class.

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
        Loads the `installments_payments.csv` DataFrame into memory.

        Returns:
            None
        '''
        if self.verbose:
            logger.info('##########################################################')
            logger.info('#        Pre-processing installments_payments.csv        #')
            logger.info('##########################################################')
            logger.info("Loading the DataFrame, installments_payments.csv, into memory...")

        self.installments_payments = reduce_memory_usage(pd.read_csv(self.file_directory + 'installments_payments.csv'))
        self.initial_shape = self.installments_payments.shape

        if self.verbose:
            logger.info("Loaded installments_payments.csv")
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

        # Example of preprocessing and feature engineering (customize as needed)
        # self.installments_payments['new_feature'] = self.installments_payments['feature_1'] / self.installments_payments['feature_2']

        if self.verbose:
            logger.info("Data Pre-processing and Feature Engineering Done.")
            logger.info('Time Taken: {}', datetime.now() - start)

    def aggregations_sk_id_curr(self) -> pd.DataFrame:
        '''
        Aggregates the installments payments on previous loans over SK_ID_CURR.

        Returns:
            pd.DataFrame: Installments payments aggregated over SK_ID_CURR.
        '''
        if self.verbose:
            logger.info("Aggregating installments payments over SK_ID_CURR...")

        # Combining numerical features (only numerical features)
        installments_payments_aggregated = (
            self.installments_payments.select_dtypes(include=[np.number])
            .drop('SK_ID_PREV', axis=1)
            .groupby(by=['SK_ID_CURR'])
            .mean()
            .reset_index()
        )
        installments_payments_aggregated.columns = [
            'INSTA_' + column if column != 'SK_ID_CURR' else column
            for column in installments_payments_aggregated.columns
        ]
        installments_payments_aggregated.fillna(0, inplace=True)

        if self.verbose:
            logger.info('Aggregation Done.')
            logger.info('Size after aggregation: {}', installments_payments_aggregated.shape)

        return installments_payments_aggregated

    def main(self) -> pd.DataFrame:
        '''
        Performs complete preprocessing and aggregation of the `installments_payments` table.

        Returns:
            pd.DataFrame: Final preprocessed and aggregated `installments_payments` table.
        '''
        # Loading the DataFrame
        self.load_dataframe()

        # Performing preprocessing and feature engineering
        self.data_preprocessing_and_feature_engineering()

        # Aggregating the installments payments over SK_ID_CURR
        installments_payments_aggregated = self.aggregations_sk_id_curr()

        if self.verbose:
            logger.info('Done preprocessing installments_payments.')
            logger.info('Initial Size of installments_payments: {}', self.initial_shape)
            logger.info(
                'Size of installments_payments after Pre-Processing, Feature Engineering and Aggregation: {}',
                installments_payments_aggregated.shape,
            )
            logger.info('Total Time Taken: {}', datetime.now() - self.start)

        if self.dump_to_pickle:
            if self.verbose:
                logger.info('Pickling pre-processed installments_payments to installments_payments_preprocessed.pkl')
            with open(self.file_directory + 'installments_payments_preprocessed.pkl', 'wb') as f:
                pickle.dump(installments_payments_aggregated, f)
            if self.verbose:
                logger.info('Pickling completed.')

        return installments_payments_aggregated
