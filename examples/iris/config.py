import os

experiment_name = "iris_dataset"
experiment_root_directory = os.path.join(os.path.expanduser("~"), "vitaFlow/")
use_char_embd = True

# https://github.com/tensorflow/models/blob/master/samples/core/get_started/premade_estimator.py
# https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py

experiments = {
    "num_epochs": 5,
    "dataset_name": "iris_dataset",
    "data_iterator_name": "iris_iterator",
    "model_name": "LiReg",

    # dataset details
    "iris_dataset": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "over_write": False,
    },

    # data iterator
    "iris_iterator": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "iterator_name": "iris_data_iterator",
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "text_col": "0",
        "entity_col": "3",
        "batch_size": 4,
        "seperator": "~",  # potential error point depending on the dataset
        "quotechar": "^",
        "max_word_length": 20,
        "use_char_embd": use_char_embd
    },

    # data model
    "LiReg": {
        "model_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "use_char_embd": use_char_embd,
        "learning_rate": 0.001,
        "word_level_lstm_hidden_size": 12,
        "char_level_lstm_hidden_size": 12,
        "word_emd_size": 8,
        "char_emd_size": 8,
        "num_lstm_layers": 1,
        "keep_propability": 0.5,
    }
}