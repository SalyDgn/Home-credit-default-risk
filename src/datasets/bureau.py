import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import reduce_memory_usage


class preprocess_bureau_balance_and_bureau:
    '''
    Preprocess the tables bureau_balance and bureau.
    Contains 4 member functions:
        1. init method
        2. preprocess_bureau_balance method
        3. preprocess_bureau method
        4. main method
    '''

    def __init__(self, file_directory: str = '', verbose: bool = True, dump_to_pickle: bool = False):
        '''
        This function is used to initialize the class members

        Inputs:
            self
            file_directory: Path, str, default = ''
                The path where the file exists. Include a '/' at the end of the path in input
            verbose: bool, default = True
                Whether to enable verbosity or not
            dump_to_pickle: bool, default = False
                Whether to pickle the final preprocessed table or not

        Returns:
            None
        '''

        self.file_directory = file_directory
        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.start = datetime.now()
        logger.info('Preprocessing class initialized.')

    def preprocess_bureau_balance(self):
        '''
        Function to preprocess bureau_balance table.
        This function first loads the table into memory, does some feature engineering, and finally
        aggregates the data over SK_ID_BUREAU

        Inputs:
            self

        Returns:
            preprocessed and aggregated bureau_balance table.
        '''

        if self.verbose:
            logger.info('#######################################################')
            logger.info('#          Pre-processing bureau_balance.csv          #')
            logger.info('#######################################################')
            logger.info("\nLoading the DataFrame, bureau_balance.csv, into memory...")

        bureau_balance = reduce_memory_usage(pd.read_csv(self.file_directory + 'bureau_balance.csv'))

        if self.verbose:
            logger.info("Loaded bureau_balance.csv")
            logger.info(f"Time Taken to load = {datetime.now() - self.start}")
            logger.info("\nStarting Data Cleaning and Feature Engineering...")

        # # Encode STATUS labels
        # dict_for_status = {'C': 0, '0': 1, '1': 2, '2': 3, 'X': 4, '3': 5, '4': 6, '5': 7}
        # bureau_balance['STATUS'] = bureau_balance['STATUS'].map(dict_for_status)

        # if self.verbose:
        #     logger.info("Halfway through. A little bit more patience...")
        #     logger.info(f"Total Time Elapsed = {datetime.now() - self.start}")

        # # Aggregating over whole dataset
        # aggregated_bureau_balance = bureau_balance.groupby(['SK_ID_BUREAU']).mean().reset_index()

        if self.verbose:
            logger.info('Done preprocessing bureau_balance.')
            logger.info(f"\nInitial Size of bureau_balance: {bureau_balance.shape}")
            logger.info(
                f'Size of bureau_balance after Pre-Processing, Feature Engineering and Aggregation: {bureau_balance.shape}'
            )
            logger.info(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                logger.info('\nPickling pre-processed bureau_balance to bureau_balance_preprocessed.pkl')
            with open(self.file_directory + 'bureau_balance_preprocessed.pkl', 'wb') as f:
                pickle.dump(aggregated_bureau_balance, f)
            if self.verbose:
                logger.info('Done.')

        return bureau_balance

    def preprocess_bureau(self, aggregated_bureau_balance):
        '''
        Function to preprocess the bureau table and merge it with the aggregated bureau_balance table.

        Inputs:
            self
            aggregated_bureau_balance: DataFrame of aggregated bureau_balance table

        Returns:
            Final preprocessed, merged and aggregated bureau table
        '''

        if self.verbose:
            start2 = datetime.now()
            logger.info('\n##############################################')
            logger.info('#          Pre-processing bureau.csv         #')
            logger.info('##############################################')
            if self.verbose:
                start2 = datetime.now()
                logger.info('Starting preprocessing of bureau.csv')
            logger.info("\nLoading the DataFrame, bureau.csv, into memory...")

        bureau = reduce_memory_usage(pd.read_csv(self.file_directory + 'bureau.csv'))

        if self.verbose:
            logger.info("Loaded bureau.csv")
            logger.info(f"Time Taken to load = {datetime.now() - start2}")
            logger.info("\nStarting Data Cleaning and Feature Engineering...")
        # Merge with aggregated_bureau_balance
        bureau_merged = bureau.merge(aggregated_bureau_balance, on=['SK_ID_BUREAU'], how='right').drop(
            'SK_ID_BUREAU', axis=1
        )
        # Combine numerical features
        bureau_numerical_aggregated = (
            bureau_merged.select_dtypes(include=[np.number]).groupby(['SK_ID_CURR']).mean().reset_index()
        )
        bureau_numerical_aggregated.columns = [
            'BUREAU_' + column if column != 'SK_ID_CURR' else column for column in bureau_numerical_aggregated.columns
        ]

        # Combine categorical features
        bureau_categorical = pd.get_dummies(bureau_merged.select_dtypes('object'))
        bureau_categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
        bureau_categorical_aggregated = bureau_categorical.groupby(['SK_ID_CURR']).mean().reset_index()

        # Merge numerical and categorical features
        bureau_merged_aggregated = bureau_numerical_aggregated.merge(bureau_categorical_aggregated, on='SK_ID_CURR')
        bureau_merged_aggregated.update(bureau_merged_aggregated.fillna(0))
        bureau_merged_aggregated.columns = [
            'BUREAU_' + column if column != 'SK_ID_CURR' else column for column in bureau_merged_aggregated.columns
        ]

        bureau_merged_aggregated.fillna(0, inplace=True)

        if self.verbose:
            logger.info('Preprocessing of bureau completed.')
            logger.info('Initial Size of bureau: {}', bureau.shape)
            logger.info('Size after merging, preprocessing, and aggregation: {}', bureau_merged_aggregated.shape)
            logger.info('Total Time Taken: {}', datetime.now() - self.start)

        if self.dump_to_pickle:
            if self.verbose:
                logger.info('Pickling pre-processed bureau and bureau_balance to bureau_merged_preprocessed.pkl')
            with open(self.file_directory + 'bureau_merged_preprocessed.pkl', 'wb') as f:
                pickle.dump(bureau_merged_aggregated, f)
            if self.verbose:
                logger.info('Pickling completed.')
        if self.verbose:
            logger.info('-' * 100)

        return bureau_merged_aggregated

    def main(self) -> pd.DataFrame:
        '''
        Function to be called for complete preprocessing and aggregation of the bureau and bureau_balance tables.

        Inputs:
            self

        Returns:
            pd.DataFrame: The final preprocessed and merged `bureau` and `bureau_balance` tables.
        '''

        # Preprocess the bureau_balance first
        aggregated_bureau_balance = self.preprocess_bureau_balance()

        # Preprocess the bureau table next, by combining it with the aggregated bureau_balance
        bureau_merged_aggregated = self.preprocess_bureau(aggregated_bureau_balance)

        return bureau_merged_aggregated
