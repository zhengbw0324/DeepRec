

PORT=6001

export CUDA_VISIBLE_DEVICES=7
HOST=0.0.0.0

python -m server.recall \
  --host=$HOST \
  --port=$PORT \
|tee ./log/recall/${PORT}.log


