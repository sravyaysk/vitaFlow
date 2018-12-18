import os

experiment_root_directory = os.path.join(os.path.expanduser("~"), "vitaFlow-clientx/")
experiment_name = "clientx_dataset"
use_char_embd = True

experiments = {
    "num_epochs": 6,
    "dataset_class_with_path": "examples.clientx.clientx_dataset.CLIENTXDataset",
    "iterator_class_with_path": "vitaflow.data.text.iterators.CSVSeqToSeqIterator",
    "model_class_with_path": "vitaflow.models.text.seq2seq.BiLSTMCrf",
    "save_checkpoints_steps" : 50,
    "keep_checkpoint_max" : 5,
    "save_summary_steps" : 25,
    "log_step_count_steps" : 10,

    # dataset - details
    "examples.clientx.clientx_dataset.CLIENTXDataset": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "minimum_num_words": 5,
        "over_write": False,
    },

    # data iterator
    "vitaflow.data.text.iterators.CSVSeqToSeqIterator": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "iterator_name": "conll_data_iterator",
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "text_col": "word",
        "entity_col": "entity_name_iob",
        "batch_size": 16,
        "seperator": "~",  # potential error point depending on the dataset
        "quotechar": "^",
        "max_word_length": 20,
        "use_char_embd": use_char_embd
    },

    # data model
    "vitaflow.models.text.seq2seq.BiLSTMCrf": {
        "model_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "use_char_embd": use_char_embd,
        "learning_rate": 0.001,
        "word_level_lstm_hidden_size": 128,
        "char_level_lstm_hidden_size": 300,
        "word_emd_size": 128,
        "char_emd_size": 300,
        "num_lstm_layers": 2,
        "keep_propability": 0.4,
    }
}