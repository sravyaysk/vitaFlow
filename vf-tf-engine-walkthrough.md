# Walkthrough of vitaFlow - TensorFlow Engine

Here's a walkthrough training a good English-to-German translation model using the Transformer model from [Attention Is All You Need](https://arxiv.org/abs/1706.03762) on WMT data.

```sh
# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).

#TODO create installer and resolve path dependency
export PATH=/path/to/vitaflow_repo/vitaflow/bin/:$PATH
cd /path/to/vitaflow_repo/

vf-trainer --registry_help

PROBLEM=translate_ende_wmt8k
MODEL=transformer
#MODEL_HPARAMS=transformer_base_single_gpu
MODEL_HPARAMS=transformer_tiny
DATA_DIR=$HOME/vf_data
TMP_DIR=$HOME/vf_data/tmp
TRAIN_DIR=$HOME/vf_train/$PROBLEM/$MODEL-$MODEL_HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data, downloads ~1.7GB of data, network failures are expected
vf-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
vf-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$MODEL_HPARAMS \
  --output_dir=$TRAIN_DIR

# Decode

DECODE_FILE=$DATA_DIR/decode_this.txt
echo "Hello world" >> $DECODE_FILE
echo "Goodbye world" >> $DECODE_FILE
echo -e 'Hallo Welt\nAuf Wiedersehen Welt' > ref-translation.de

BEAM_SIZE=4
ALPHA=0.6

vf-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$MODEL_HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=translation.en

# See the translations
cat translation.en


# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
vf-bleu --translation=translation.en --reference=ref-translation.de
```
