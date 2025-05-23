
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export RAY_DEDUP_LOGS=0

export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_GID_INDEX=3

export WANDB_MODE=offline
export WANDB_API_KEY=XXXXXXX


ray stop --force


HEAD_NODE_IP=XXXXXX
ray start --head --node-ip-address ${HEAD_NODE_IP} --num-gpus 8 --disable-usage-stats



