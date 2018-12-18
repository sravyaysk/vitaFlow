"""
``` bash
cd /path/to/vitaflow/
python vitaflow/run/run.py --config_python_file=examples/asariri/config.py --mode=train
```
"""
import os

experiment_root_directory = os.path.join(os.path.expanduser("~"), "vitaFlow/")
experiment_name = "asariri_dataset"

experiments = {
    "num_epochs": 5,
    "dataset_name": "Cifar10Dataset",
    "data_iterator_name": "Cifar10BasicIterator",
    "model_name": "vitaflow.models.image.gan.vanilla_gan.VanillaGAN",
    "save_checkpoints_steps" : 1000,
    "keep_checkpoint_max" : 5,
    "save_summary_steps" : 25,
    "log_step_count_steps" : 10,

    # dataset - details
    "Cifar10Dataset": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "over_write": False,
    },

    # data iterator
    "Cifar10BasicIterator": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "iterator_name": "Cifar10BasicIterator",
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "batch_size": 8,
    },

    # data model
    "vitaflow.models.image.gan.vanilla_gan.VanillaGAN": {
        "model_directory": experiment_root_directory,
        "experiment_name": experiment_name
    },

    "vitaflow.models.image.gan.cgan.CGAN": {
        "model_directory": experiment_root_directory,
        "experiment_name": experiment_name
    },

    "vitaflow.models.image.gan.began.BEGAN": {
        "model_directory": experiment_root_directory,
        "experiment_name": experiment_name
    }

}

