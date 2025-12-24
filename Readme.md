export NCCL_P2P_DISABLE=1
unset http_proxy
unset https_proxy
cd Kontext_train_ds2
accelerate launch --config_file ./train/deepspeed.yaml ./train/train_ds2.py --num_epochs 5 --lr 1e-4 --save_steps 500

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

问题1：
出图是很平滑的噪声，或者条纹型噪声，无图像内容
尝试1：
1.增加time shift
2.去掉RoPE offset
发现1：
loss的稳定值比之前从0.14下降到0.1以下，但是出图还是模糊噪声

问题2：
我发现好像不止是我自己的问题，回头尝试DreamOmni2的edit一样是出图噪声
猜测2：
估计是推理代码上有共同的问题，而我的推理代码是Dreamomni共用的，所以我要找出这个推理问题
1.已排除模型参数的问题，直接跑flux-kontext并无错误
2.把Dreamomni仓库原本的代码进行复制，替换掉pipeline_dreamomni2.py，发现无改善，证明不是该文件的问题
3.把Dreamomni仓库原本的代码进行复制，替换掉inference_edit.py，发现问题解决，定位问题在该文件
4.发现了0号gpu在跑Dreamomni2的时候很有问题，我在DreamOmni2的edit下做了实验，发现凡是用到了0号卡进行推理的，都会有或多或少的质量问题。
尝试2：
使用1号卡或者2号卡进行推理,发现问题依旧存在

问题3：
我尝试把我自己训练的lora导入DreamOmni2查看效果，提示有unexpect key。
猜测3：
难道我保存lora的时候dict不对？
尝试3：
首先我要打印出我保存的lora，然后跟pipe本身的lora进行名字上的比对，确认问题就是lora没有对齐

最后，我们使用训练时的导入lora方式，对lora进行一个导入，就可以了。

问题4：如果我们用回0号卡去跑，是否会出现Dreamomni的类似问题？
又没有这个问题了。那么那是DreamOmni2本身的问题吗？I Don't know.



