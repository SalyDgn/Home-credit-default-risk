import os
import sys
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from settings.params import PARAMS


def load_dataset(filename, competition='home-credit-default-risk', directory=PARAMS['file_directory']):
    """
    Download the specified Kaggle competition data into the specified directory.

    Args:
        competition (str): The name of the Kaggle competition. Default is 'home-credit-default-risk'.
        directory (str): The directory where the data will be downloaded. Default is 'data/home-credit-default-risk'.

    Returns:
        None
    """
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directory created: {directory}")
    else:
        logger.info(f"Directory already exists: {directory}")

    api = KaggleApi()
    api.authenticate()
    try:
        logger.info(f"Downloading data for into directory: {directory}")
        api.competition_download_file(
            competition='home-credit-default-risk', file_name=filename, path=directory, quiet=False
        )
        logger.info("Download successful.")
    except Exception as e:
        logger.error(f"Error during data download: {e}")

    if filename != 'HomeCredit_columns_description.csv':
        # Extract the downloaded ZIP files
        zip_path = f"{directory}/{filename}.zip"
        print(zip_path)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract(zip_path)
            logger.info(f"File successfully extracted into directory: {directory}")
        except zipfile.BadZipFile as e:
            logger.error(f"Error during file extraction: {e}")
            return


load_dataset('bureau.csv')
