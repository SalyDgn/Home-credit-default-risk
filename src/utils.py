import numpy as np
import pandas as pd
from loguru import logger

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize the memory usage of a DataFrame by downcasting numerical columns to more efficient types.
    
    Args:
        df (pd.DataFrame): The DataFrame for which memory usage should be optimized.

    Returns:
        pd.DataFrame: The DataFrame with optimized memory usage.

    Notes:
        - This function will downcast integer columns to the smallest possible integer type (int8, int16, int32, or int64) 
          based on their minimum and maximum values.
        - It will also downcast floating-point columns to the smallest possible floating-point type (float16, float32, or float64).
        - Columns of type object (e.g., strings) are not modified.
        - The function prints the memory usage before and after optimization and the percentage decrease in memory usage.
    """
    # Calculate and print initial memory usage
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    # Iterate through each column to optimize its data type
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                # Downcast integer columns
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Downcast float columns
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    # Calculate and print memory usage after optimization
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
