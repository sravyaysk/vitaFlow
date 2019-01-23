import os

FRAMES_PER_SAMPLE = 200  # number of frames forming a chunk of data
SAMPLING_RATE = 16000
FRAME_SIZE = 512
NEFF = 256  # effective FFT points
# amplification factor of the waveform sig
AMP_FAC = 10000 # i.e log(10000) = 4
MIN_AMP = 10000
# TF bins smaller than THRESHOLD will be
# considered inactive
THRESHOLD = 40
# prams for pre-whitening
GLOBAL_MEAN = 44
GLOBAL_STD = 15.5
LEARNING_RATE = 1e-3

experiment_root_directory = os.path.join(os.path.expanduser("~"), "vitaFlow/")
experiment_name = "TEDLiumDataset"
batch_size = 64

experiments = {
    "num_epochs": 5,
    "dataset_class_with_path": "examples.shabda.tedlium_dataset.TEDLiumDataset",
    "iterator_class_with_path": "examples.shabda.tedlium_parallel_iterator.TEDLiumIterator",
    "model_class_with_path": "examples.shabda.deep_clustering.DeepClustering",
    "save_checkpoints_steps": 1000,
    "keep_checkpoint_max": 5,
    "save_summary_steps": 100,
    "log_step_count_steps": 100,
    "clear_model_data" : False,

    "examples.shabda.tedlium_dataset.TEDLiumDataset": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "train_data_path": "train",
        "validation_data_path": "dev",
        "test_data_path": "test",
        "num_clips": 128,
        "duration": 5,
        "start_time": 0,
        "end_time": 600,
        "sampling_rate": SAMPLING_RATE,
    },

    "examples.shabda.tedlium_parallel_iterator.TEDLiumIterator": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "dev",
        "test_data_path": "test",
        "sampling_rate" : SAMPLING_RATE,
        "frame_size" : FRAME_SIZE,
        "neff" : NEFF,
        "min_amp" : MIN_AMP,
        "amp_fac" : AMP_FAC,
        "threshold" : THRESHOLD,
        "global_mean" : GLOBAL_MEAN,
        "global_std" : GLOBAL_STD,
        "frames_per_sample" : FRAMES_PER_SAMPLE,
        "batch_size" : batch_size,
        "prefetch_size" : 1,
        "num_threads" : 8
    },

    "examples.shabda.deep_clustering.DeepClustering" : {
        "model_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "neff" : NEFF,
        "batch_size" : batch_size,
        "lstm_hidden_size" : 320,
        "p_keep_ff" : 0.5,
        "p_keep_rc" : 0.2,
        "frames_per_sample" : FRAMES_PER_SAMPLE,
        "embd_dim" : 512,
    }
}
