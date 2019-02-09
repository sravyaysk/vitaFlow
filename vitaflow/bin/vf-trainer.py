# coding=utf-8
# Copyright 2019 The vitaFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train and evaluate."""

import contextlib
import os
import sys

import tensorflow as tf

from vitaflow.utils import registry
from vitaflow.internal import hyperparams_lib
# from vitaflow.core import registry
# from vitaflow.core import hyperparams_lib
from vitaflow.utils import registry
from vitaflow.internal import hyperparams_lib

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("registry_help", False,
                  "If True, logs the contents of the registry and exits.")
flags.DEFINE_string("model", None, "Which model to use.")
flags.DEFINE_string("hparams_set", None, "Which parameters to use.")

# data_dir is a common flag name - catch conflicts and define it once.
try:
    flags.DEFINE_string("data_dir", None, "Directory with training data.")
except:  # pylint: disable=bare-except
    pass
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen",
                    "Temporary storage directory, used if --generate_data.")
flags.DEFINE_string("output_dir", "", "Base output directory for run.")


flags.DEFINE_string("schedule", "continuous_train_and_eval",
                    "Method of Experiment to run.")
flags.DEFINE_integer("train_steps", 250000,
                     "The number of steps to run training for.")
flags.DEFINE_integer("eval_steps", 100,
                     "Number of steps in evaluation. By default, eval will "
                     "stop after eval_steps or when it runs through the eval "
                     "dataset once in full, whichever comes first, so this "
                     "can be a very large number.")


def maybe_log_registry_and_exit():
    if FLAGS.registry_help:
        tf.logging.info(registry.help_string())
        sys.exit(0)

def create_hparams():
    """Create hparams."""
    if FLAGS.use_tpu and "tpu" not in FLAGS.hparams_set:
        tf.logging.warn("Not all hyperparameter sets work on TPU. "
                        "Prefer hparams_sets with a '_tpu' suffix, "
                        "e.g. transformer_tpu, if available for your model.")
    hparams_path = os.path.join(FLAGS.output_dir, "hparams.json")
    # return trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams,
    #                                   hparams_path=hparams_path)

def create_experiment_fn():
    pass

def create_run_config(hp, output_dir=None):
    """Create a run config.

    Args:
      hp: model hyperparameters
      output_dir: model's output directory, defaults to output_dir flag.

    Returns:
      a run config
    """
    pass

def generate_data():
    # Generate data if requested.
    data_dir = os.path.expanduser(FLAGS.data_dir)
    tmp_dir = os.path.expanduser(FLAGS.tmp_dir)
    tf.gfile.MakeDirs(data_dir)
    tf.gfile.MakeDirs(tmp_dir)

    problem_name = FLAGS.problem
    tf.logging.info("Generating data for %s" % problem_name)
    registry.problem(problem_name).generate_data(data_dir, tmp_dir)

def save_metadata(hparams):
    """Saves FLAGS and hparams to output_dir."""
    output_dir = os.path.expanduser(FLAGS.output_dir)
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    # Save FLAGS in txt file
    if hasattr(FLAGS, "flags_into_string"):
        flags_str = FLAGS.flags_into_string()
        t2t_flags_str = "\n".join([
            "--%s=%s" % (f.name, f.value)
            for f in FLAGS.flags_by_module_dict()["tensor2tensor.utils.flags"]
        ])
    else:
        flags_dict = FLAGS.__dict__["__flags"]
        flags_str = "\n".join(
            ["--%s=%s" % (name, str(f)) for (name, f) in flags_dict.items()])
        t2t_flags_str = None

    flags_txt = os.path.join(output_dir, "flags.txt")
    with tf.gfile.Open(flags_txt, "w") as f:
        f.write(flags_str)

    if t2t_flags_str:
        t2t_flags_txt = os.path.join(output_dir, "flags_t2t.txt")
        with tf.gfile.Open(t2t_flags_txt, "w") as f:
            f.write(t2t_flags_str)

    # Save hparams as hparams.json
    new_hparams = hyperparams_lib.copy_hparams(hparams)
    # Modality class is not JSON serializable so remove.
    new_hparams.del_hparam("modality")

    hparams_fname = os.path.join(output_dir, "hparams.json")
    with tf.gfile.Open(hparams_fname, "w") as f:
        f.write(new_hparams.to_json(indent=0, sort_keys=True))

@contextlib.contextmanager
def profile_context():
    if FLAGS.profile:
        with tf.contrib.tfprof.ProfileContext(
                "t2tprof", trace_steps=range(100), dump_steps=range(100)) as pctx:
            opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
            pctx.add_auto_profiling("op", opts, range(100))
            yield
    else:
        yield

def is_chief():
    schedules = ["train", "train_and_evaluate", "continuous_train_and_eval"]
    return FLAGS.worker_id == 0 and FLAGS.schedule in schedules

def execute_schedule(exp):
    if not hasattr(exp, FLAGS.schedule):
        raise ValueError(
            "Experiment has no method %s, from --schedule" % FLAGS.schedule)
    with profile_context():
        getattr(exp, FLAGS.schedule)()

def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    # If we just have to print the registry, do that and exit early.
    maybe_log_registry_and_exit()

    # Create HParams.
    hparams = create_hparams()

    if FLAGS.generate_data:
        generate_data()

    exp_fn = create_experiment_fn()
    exp = exp_fn(create_run_config(hparams), hparams)
    if is_chief():
        save_metadata(hparams)
    execute_schedule(exp)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
