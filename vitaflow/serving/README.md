# Serving

Tensor2Tensor and the TensorFlow ecosystem make it easy to serve a model once
trained.

## 0. Constants

```
PROBLEM=conll2002_es_ner
MODEL=lstm_seq2seq
MODEL_HPARAMS=lstm_seq2seq
DATA_DIR=~/vf_data
TEMP_DIR=~/vf_data/tmp
MODEL_OUT_DIR=~/vf_train/$PROBLEM\_$MODEL
```

```
PROBLEM=translate_ende_wmt8k
MODEL=transformer
MODEL_HPARAMS=transformer_tiny
DATA_DIR=~/vf_data
TEMP_DIR=~/vf_data/tmp
MODEL_OUT_DIR=~/vf_train/$PROBLEM\_$MODEL
```

## 1. Export for Serving
First, build a simple model:

```
python vitaflow/bin/vf-trainer \
  --generate_data \
  --problem=$PROBLEM \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TEMP_DIR \
  --model=$MODEL \
  --hparams_set=$MODEL_HPARAMS \
  --output_dir=$MODEL_OUT_DIR \
  --train_steps=1000 \
  --eval_steps=100
```
First, export it for serving:

```
python vitaflow/bin/vf-exporter \
  --model=$MODEL \
  --hparams_set=$MODEL_HPARAMS \
  --problem=$PROBLEM \
  --data_dir=$DATA_DIR \
  --output_dir=$MODEL_OUT_DIR
```

You should have an export directory in `output_dir` now.

## 2. Launch a Server

Install the `tensorflow-model-server`
([instructions](https://www.tensorflow.org/serving/setup#installing_the_modelserver)).

Start the server pointing at the export:

```
tensorflow_model_server \
  --port=9000 \
  --model_name=$MODEL \
  --model_base_path=$MODEL_OUT_DIR/export/
```

## 3. Query the Server

**Note**: The `vf-query-server` is meant only as an example. You may need to
modify it to suit your needs. The exported model expects an input
example that is structured identically to what would be found on disk during
training (serialized `tf.train.Example`). For text problems, that means that
it expects the inputs to already be encoded as integers. You can see how the
`vf-query-server` does this by reading the code.

Install some dependencies:

```
pip install tensorflow-serving-api
```

Query:

```
vf-query-server \
  --server=localhost:9000 \
  --servable_name=$MODEL \
  --problem=$PROBLEM \
  --data_dir=$DATA_DIR
```


## Serve Predictions with Cloud ML Engine

Alternatively, you can deploy a model on Cloud ML Engine to serve predictions.
To do so, export the model as in Step 1, then do the following:

[Install gcloud](https://cloud.google.com/sdk/downloads)

#### Copy exported model to Google Cloud Storage

```
ORIGIN=<your_gcs_path>
EXPORTS_PATH=/tmp/vf_train/export/Servo
LATEST_EXPORT=${EXPORTS_PATH}/$(ls ${EXPORTS_PATH} | tail -1)
gsutil cp -r ${LATEST_EXPORT}/* $ORIGIN
```

#### Create a model

```
MODEL_NAME=vf_test
gcloud ml-engine models create $MODEL_NAME
```

This step only needs to be performed once.

#### Create a model version

```
VERSION=v0
gcloud ml-engine versions create $VERSION \
  --model $MODEL_NAME \
  --origin $ORIGIN
```

**NOTE:** Due to overhead from VM warmup, prediction requests may timeout. To
mitigate this issue, provide a [YAML configuration
file](https://cloud.google.com/sdk/gcloud/reference/ml-engine/versions/create)
via the `--config flag`, with `minNodes > 0`. These nodes are always on, and
will be billed accordingly.

#### Query Cloud ML Engine

```
vf-query-server \
  --cloud_mlengine_model_name $MODEL_NAME \
  --cloud_mlengine_model_version $VERSION \
  --problem translate_ende_wmt8k \
  --data_dir ~/vf/data
```
