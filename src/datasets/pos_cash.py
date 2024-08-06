import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from utils import reduce_memory_usage

class preprocess_POS_CASH_balance:
    '''
    Preprocess the POS_CASH_balance table.

    This class contains methods to load, preprocess, and aggregate the `POS_CASH_balance` table.


    '''

    def __init__(self, file_directory: str = '', verbose: bool = True, dump_to_pickle: bool = False):
        '''
        Initializes the preprocess_POS_CASH_balance class.

        Args:
            file_directory (str): Path to the directory where the files are located.
            verbose (bool): Whether to enable verbose logging.
            dump_to_pickle (bool): Whether to pickle the final preprocessed table.
            nrows (Optional[int]): Number of rows to read from the CSV file.
        '''
        self.file_directory = file_directory
        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.nrows = nrows
        self.start = datetime.now()
        logger.info('Preprocessing class initialized.')

    def load_dataframe(self):
        '''
        Loads the `POS_CASH_balance.csv` DataFrame into memory.

        Returns:
            None
        '''
        if self.verbose:
            logger.info('#########################################################')
            logger.info('#          Pre-processing POS_CASH_balance.csv          #')
            logger.info('#########################################################')
            logger.info("Loading the DataFrame, POS_CASH_balance.csv, into memory...")

        self.pos_cash = reduce_memory_usage(pd.read_csv(self.file_directory + 'POS_CASH_balance.csv'))
        self.initial_size = self.pos_cash.shape

        if self.verbose:
            logger.info("Loaded POS_CASH_balance.csv")
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
        # self.pos_cash['new_feature'] = self.pos_cash['feature_1'] / self.pos_cash['feature_2']

        if self.verbose:
            logger.info("Data Pre-processing and Feature Engineering Done.")
            logger.info('Time Taken: {}', datetime.now() - start)

    def aggregations_sk_id_curr(self) -> pd.DataFrame:
        '''
        Aggregates the POS_CASH_balance table over SK_ID_CURR.

        Returns:
            pd.DataFrame: POS_CASH_balance table aggregated over SK_ID_CURR.
        '''
        if self.verbose:
            logger.info("Aggregating POS_CASH_balance over SK_ID_CURR...")

        # Combining numerical features
        pos_cash_numerical_aggregated = self.pos_cash.select_dtypes(include=[np.number]).drop('SK_ID_PREV', axis=1).groupby(by=['SK_ID_CURR']).mean().reset_index()

        # Combining categorical features
        pos_cash_categorical = pd.get_dummies(self.pos_cash.select_dtypes('object'))
        pos_cash_categorical['SK_ID_CURR'] = self.pos_cash['SK_ID_CURR']
        pos_cash_categorical_aggregated = pos_cash_categorical.groupby('SK_ID_CURR').mean().reset_index()

        # Merge numerical and categorical features
        pos_cash_aggregated = pos_cash_numerical_aggregated.merge(pos_cash_categorical_aggregated, on='SK_ID_CURR')
        pos_cash_aggregated.columns = ['POS_' + column if column != 'SK_ID_CURR' else column for column in pos_cash_aggregated.columns]
        pos_cash_aggregated.fillna(0, inplace=True)

        if self.verbose:
            logger.info('Aggregation Done.')
            logger.info('Size after aggregation: {}', pos_cash_aggregated.shape)

        return pos_cash_aggregated

    def main(self) -> pd.DataFrame:
        '''
        Performs complete preprocessing and aggregation of the `POS_CASH_balance` table.

        Returns:
            pd.DataFrame: Final preprocessed and aggregated `POS_CASH_balance` table.
        '''
        # Loading the DataFrame
        self.load_dataframe()

        # Performing preprocessing and feature engineering
        self.data_preprocessing_and_feature_engineering()

        # Aggregating the POS_CASH_balance over SK_ID_CURR
        pos_cash_aggregated = self.aggregations_sk_id_curr()

        if self.verbose:
            logger.info('Done preprocessing POS_CASH_balance.')
            logger.info('Initial Size of POS_CASH_balance: {}', self.initial_size)
            logger.info('Size of POS_CASH_balance after Pre-Processing, Feature Engineering and Aggregation: {}', pos_cash_aggregated.shape)
            logger.info('Total Time Taken: {}', datetime.now() - self.start)

        if self.dump_to_pickle:
            if self.verbose:
                logger.info('Pickling pre-processed POS_CASH_balance to POS_CASH_balance_preprocessed.pkl')
            with open(self.file_directory + 'POS_CASH_balance_preprocessed.pkl', 'wb') as f:
                pickle.dump(pos_cash_aggregated, f)
            if self.verbose:
                logger.info('Pickling completed.')

        return pos_cash_aggregated