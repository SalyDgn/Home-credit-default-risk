# import os
# import sys

# import pytest

# # Add the parent directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.datasets import *
# from src.load_dataset import load_all_dataset


# @pytest.fixture
# def setup_and_teardown():
#     # Setup: Create a temporary directory
#     test_dir = 'test_data'
#     if not os.path.exists(test_dir):
#         os.makedirs(test_dir)
#     yield test_dir
#     # Teardown: Remove the directory after test
#     if os.path.exists(test_dir):
#         for root, dirs, files in os.walk(test_dir, topdown=False):
#             for name in files:
#                 os.remove(os.path.join(root, name))
#             for name in dirs:
#                 os.rmdir(os.path.join(root, name))
#         os.rmdir(test_dir)


# test_dir = setup_and_teardown
# if os.path.exists(test_dir):
#     os.rmdir(test_dir)
# assert not os.path.exists(test_dir)
# load_all_dataset(directory=test_dir)

# # def test_application(setup_and_teardown):

# #     train_data = reduce_memory_usage(pd.read_csv(test_dir + '/application_train.csv'))
# #     test_data = reduce_memory_usage(pd.read_csv(test_dir + '/application_test.csv'))

# #     assert train_data.shape == (307511, 122)
# #     assert test_data.shape == (48744, 121)

# #     application_train, application_test = preprocess_application_train_test(file_directory1=test_dir, file_directory2=test_dir).main()

# #     assert application_train.shape[1] == 240
# #     assert application_test.shape == (48744, 237)

# # To do : others class and merge function
