export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export RAY_DEDUP_LOGS=0

export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_GID_INDEX=3

export WANDB_MODE=offline
export WANDB_API_KEY=XXXXXXX
wandb_group=RecTeam
wandb_project=DeepRec





reward_port=XXXX
filter_data="top0_100"

data=game
TBS=512
RBS=128
N_SAMPLES=8
LR=1e-6
stage=1
max_steps=20
recall_port=XXXXXX
SERVER_IP=XXXXXX
rec_k=10
recall_k=20

run_name=${data}_recall${recall_k}_${filter_data}_${TBS}_${RBS}x${N_SAMPLES}_stage${stage}_${max_steps}
echo "========================================================================="
echo "run_name: $run_name"
echo "========================================================================="

RM_SERVER=http://${SERVER_IP}:${reward_port}/reward
RECALL_SERVER=http://${SERVER_IP}:${recall_port}/recall
IDX_KEY=${filter_data}_stage${stage}

DATA_PATH=XXXXXXX.jsonl
BASE_PATH=XXXXXXX
IDX_FILE=XXXXXXXX/combine_idx_two_stage.json
LOG_BASE=log
SAVE_MODEL_NAME=${run_name}
RES_DI=./results/
CKPT_DIR=${RES_DI}ckpts/${SAVE_MODEL_NAME}
mkdir -p $RES_DI
mkdir -p $CKPT_DIR
mkdir -p $LOG_BASE/server/

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 2 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 2 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 16 \
   --vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --pretrain ${BASE_PATH} \
   --remote_rm_url $RM_SERVER \
   --rec_topk $rec_k \
   --recall_server $RECALL_SERVER \
   --recall_topk $recall_k \
   --save_path $CKPT_DIR \
   --ckpt_path $CKPT_DIR \
   --micro_train_batch_size 2 \
   --train_batch_size ${TBS} \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size ${RBS} \
   --advantage_estimator reinforce_baseline \
   --max_samples 1000000 \
   --max_epochs 1 \
   --num_episodes 10000 \
   --lr_warmup_ratio 0.0 \
   --n_samples_per_prompt $N_SAMPLES \
   --prompt_max_len 4096 \
   --generate_max_len 28000 \
   --zero_stage 2 \
   --bf16 \
   --load_checkpoint \
   --actor_learning_rate $LR \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.0 \
   --prompt_data $DATA_PATH \
   --input_key history \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 5 \
   --vllm_sync_backend nccl \
   --max_ckpt_num 100 \
   --temperature 1.0 \
   --packing_samples \
   --use_wandb ${wandb_token} \
   --wandb_group $wandb_group \
   --wandb_project $wandb_project \
   --wandb_run_name $SAVE_MODEL_NAME \
   --filtered_idx_file $IDX_FILE \
   --filtered_idx_key $IDX_KEY \
   --max_global_steps $max_steps


bash script/convert_ckpt.sh $run_name










