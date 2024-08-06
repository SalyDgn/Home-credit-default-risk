import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from typing import Optional, Tuple

from utils import reduce_memory_usage

class preprocess_application_train_test:
    '''
    Preprocess the application_train and application_test tables.
    Contains 4 member functions:
        1. init method
        2. load_dataframes method
        3. data_cleaning method
        4. main method
    '''

    def __init__(self, file_directory1='', file_directory2='', verbose=True, dump_to_pickle=False):
        '''
        Initialize the class members.

        Args:
            file_directory1: str, default=''
                Path where the application_train.csv file exists. Include a '/' at the end.
            file_directory2: str, default=''
                Path where the application_test.csv file exists. Include a '/' at the end.
            verbose: bool, default=True
                Whether to enable verbosity or not.
            dump_to_pickle: bool, default=False
                Whether to pickle the final preprocessed tables or not.
          
        '''
        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.file_directory1 = file_directory1
        self.file_directory2 = file_directory2

    def load_dataframes(self):
        '''
        Load the application_train.csv and application_test.csv DataFrames.

        Returns:
            None
        '''
        if self.verbose:
            self.start = datetime.now()
            logger.info('#######################################################')
            logger.info('#        Pre-processing application_train.csv         #')
            logger.info('#        Pre-processing application_test.csv          #')
            logger.info('#######################################################')
            logger.info("\nLoading the DataFrames into memory...")

        self.application_train = reduce_memory_usage(pd.read_csv(self.file_directory1 + 'application_train.csv'))
        self.application_test = reduce_memory_usage(pd.read_csv(self.file_directory2 + 'application_test.csv'))
        self.initial_train_shape = self.application_train.shape
        self.initial_test_shape = self.application_test.shape

        if self.verbose:
            logger.info("Loaded application_train.csv and application_test.csv")
            logger.info(f"Time Taken to load = {datetime.now() - self.start}")

    def data_cleaning(self):
        '''
        Clean the tables by removing erroneous rows/entries.

        Returns:
            None
        '''
        if self.verbose:
            logger.info("\nPerforming Data Cleaning...")
        #there are some FLAG_DOCUMENT features having just one category for almost all data, we will remove those
        flag_cols_to_drop = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_20']

        self.application_train.drop(flag_cols_to_drop, axis=1, inplace=True)
        self.application_test.drop(flag_cols_to_drop, axis=1, inplace=True)

        if self.verbose:
            logger.info("Data Cleaning Done.")

    def main(self)-> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Complete preprocessing of application_train and application_test tables.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Final preprocessed application_train and application_test tables.
        '''
        # Load the DataFrames
        self.load_dataframes()
        
        # Perform Data Cleaning
        self.data_cleaning()
        
        # Feature Engineering
        if self.verbose:
            start = datetime.now()
            logger.info("\nStarting Feature Engineering...")
            logger.info("\nCreating Domain Based Features on Numeric Data")

        cat_col_train = [category for category in self.application_train.columns if self.application_train[category].dtype == 'object']
        cat_col_test = [category for category in self.application_test.columns if self.application_test[category].dtype == 'object']
        
        self.application_train = pd.get_dummies(self.application_train, columns=cat_col_train)
        self.application_test = pd.get_dummies(self.application_test, columns=cat_col_test)

        if self.verbose:
            logger.info("Creating features based on Categorical Interactions on some Numeric Features")
            logger.info("Feature Engineering Done.")
            logger.info(f"Time taken = {datetime.now() - start}")

        if self.verbose:
            logger.info('Preprocessing Done.')
            logger.info(f"\nInitial Size of application_train: {self.initial_train_shape}")
            logger.info(f"Size of application_train after Pre-Processing and Feature Engineering: {self.application_train.shape}")
            logger.info(f"\nInitial Size of application_test: {self.initial_test_shape}")
            logger.info(f"Size of application_test after Pre-Processing and Feature Engineering: {self.application_test.shape}")
            logger.info(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                logger.info('\nPickling pre-processed application_train and application_test to pickle files.')
            with open(self.file_directory1 + 'application_train_preprocessed.pkl', 'wb') as f:
                pickle.dump(self.application_train, f)
            with open(self.file_directory2 + 'application_test_preprocessed.pkl', 'wb') as f:
                pickle.dump(self.application_test, f)
            if self.verbose:
                logger.info('Pickling Done.')
        if self.verbose:
            logger.info('-'*100)

        return self.application_train, self.application_test