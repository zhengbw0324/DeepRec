

PORT=$1
MODEL=AddRet


export CUDA_VISIBLE_DEVICES=7
HOST=0.0.0.0

python -m server.recall \
  --host=$HOST \
  --port=$PORT \
  --model_name=$MODEL \
  --model_config_file_path=./server/config/$MODEL.yaml \
|tee ./log/recall/${PORT}.log


