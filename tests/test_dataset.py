import os
import sys

import pytest

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.load_dataset import load_dataset


@pytest.fixture
def setup_and_teardown():
    # Setup: Create a temporary directory
    test_dir = 'test_data'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    yield test_dir
    # Teardown: Remove the directory after test
    if os.path.exists(test_dir):
        for root, dirs, files in os.walk(test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(test_dir)


def test_load_all_dataset_downloads_and_extracts_files(setup_and_teardown):
    test_dir = setup_and_teardown
    if os.path.exists(test_dir):
        os.rmdir(test_dir)
    assert not os.path.exists(test_dir)

    load_dataset(directory=test_dir, filename="HomeCredit_columns_description.csv")

    assert os.path.exists(test_dir)

    file_path = os.path.join(test_dir, "HomeCredit_columns_description.csv")

    assert os.path.exists(file_path)

    assert len(os.listdir(test_dir)) == 1


# def test_load_all_dataset_downloads_and_extracts_files2(setup_and_teardown):
#     test_dir = setup_and_teardown
#     if os.path.exists(test_dir):
#         os.rmdir(test_dir)
#     assert not os.path.exists(test_dir)

#     load_dataset(directory=test_dir, filename="bureau.csv")

#     assert os.path.exists(test_dir)

#     file_path = os.path.join(test_dir, "bureau.csv.zip")

#     assert os.path.exists(file_path)

#     assert len(os.listdir(test_dir)) == 2
