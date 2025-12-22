export NCCL_P2P_DISABLE=1
cd Kontext_train_ds2
accelerate launch --config_file ./train/deepspeed.yaml ./train/train_ds2.py --num_epochs 100 --lr 1e-4 --save_steps 1

#异步错误处理
当一个 GPU 节点发生 NCCL 错误时，其他节点能及时收到通知并优雅退出，而不是一直死等（卡死）。它让错误日志更清晰。
export NCCL_ASYNC_ERROR_HANDLING=1    

如果通信超时，程序会直接报错抛出异常，而不是永远卡在那。这对于定位哪一步通信出问题非常有用。
export NCCL_BLOCKING_WAIT=1

#调试与日志
启动时会打印大量底层细节
export NCCL_DEBUG=INFO
当发生错误时，PyTorch 会尝试打印最近的通信记录，增加这个值可以让你看到更完整的错误链路。
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

#兼容性与网络路径限制
强制禁用显卡间的 P2P（Peer-to-Peer）直接通信
export NCCL_P2P_DISABLE=1

禁用 InfiniBand (IB) 网络
export NCCL_IB_DISABLE=1

指定 NCCL 通信使用的物理网卡名称
export NCCL_SOCKET_IFNAME=ens9f0

#断点重续
accelerate launch --config_file ./train/deepspeed.yaml ./train/train_ds2.py --num_epochs 100 --lr 1e-4 --save_steps 500 \
                  --resume_lora_path "./lora_ckpt/checkpoint-4000/lora.safetensors"
