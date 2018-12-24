"""
``` bash
cd /path/to/vitaflow/
python vitaflow/run/run.py --config_python_file=examples/cifar10/config.py --mode=train
```
"""
import os

experiment_root_directory = os.path.join(os.path.expanduser("~"), "vitaFlow/")
experiment_name = "cfiar10_dataset"

experiments = {
    "num_epochs": 5,
    "dataset_class_with_path": "examples.cifar10.dataset.Cifar10Dataset",
    "iterator_class_with_path": "examples.cifar10.cifar10_gan_iterator.Cifar10GanIterator",
    "model_class_with_path": "vitaflow.models.image.gan.vanilla_gan.VanillaGAN",
    "save_checkpoints_steps" : 45000/4500,
    "keep_checkpoint_max" : 5,
    "save_summary_steps" : 25,
    "log_step_count_steps" : 10,

    # dataset - details
    "examples.cifar10.dataset.Cifar10Dataset": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "over_write": False,
    },

    # data iterator
    "examples.cifar10.cifar10_gan_iterator.Cifar10GanIterator": {
        "experiment_root_directory": experiment_root_directory,
        "experiment_name": experiment_name,
        "iterator_name": "Cifar10BasicIterator",
        "preprocessed_data_path": "preprocessed_data",
        "train_data_path": "train",
        "validation_data_path": "val",
        "test_data_path": "test",
        "batch_size": 8,
        "noise_dim" : 30
    },

    # data model
    "vitaflow.models.image.gan.vanilla_gan.VanillaGAN": {
        "model_root_directory": experiment_root_directory,
        "experiment_name": experiment_name
    },

    "vitaflow.models.image.gan.cgan.CGAN": {
        "model_root_directory": experiment_root_directory,
        "experiment_name": experiment_name
    },

    "vitaflow.models.image.gan.began.BEGAN": {
        "model_root_directory": experiment_root_directory,
        "experiment_name": experiment_name
    }

}

