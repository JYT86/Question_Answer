import os

from logic.data import prepare_dataloaders_and_tokenizer, decode
from logic.transformer_model import Transformer, create_mask
from logic.trainer import Trainer

from omegaconf import OmegaConf

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

def main(rank, world_size, cfg):
    """每个进程的主函数"""
    # 设置分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    
    # 初始化进程组
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    
    # 创建Trainer并传入rank和world_size
    trainer = Trainer(cfg, rank=rank, world_size=world_size)
    trainer.run(epochs=10)
    
    # 清理
    dist.destroy_process_group()
    
if __name__ == '__main__':


    cfg = OmegaConf.load('./config/config.yaml')
    print('finish loading cfg.')
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        # 多GPU分布式训练
        mp.spawn(
            main, 
            args=(world_size, cfg), 
            nprocs=world_size, 
            join=True
        )
    else:
        # 单GPU训练
        trainer = Trainer(cfg)
        trainer.run(epochs=10)