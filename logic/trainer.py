from typing import Any, Dict, Tuple

from pathlib import Path
import math
from omegaconf import OmegaConf

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from logic.data import prepare_dataloaders_and_tokenizer, decode
from logic.transformer_model import Transformer, create_mask
from logic.evaluation import compute_perplexity

from time import time

from tqdm import tqdm

class Trainer():
    def __init__(self, cfg: OmegaConf, **kwargs):

        self.cfg = cfg

        self.rank = kwargs.get('rank', 0)
        self.world_size = kwargs.get('world_size', 1)

        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_perplexity = float('inf')

        self.is_main_process = self.rank == 0
        print(f'rank, {self.rank}, if main process, {self.is_main_process}')

        self.device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')

        self.dl_t = prepare_dataloaders_and_tokenizer(cfg.data.name, cfg.data.batch_size, cfg.data.max_len,
                                                      distributed=(self.world_size>1))

        self.train_loaders = self.dl_t['train']
        self.test_loaders = self.dl_t['valid']
        self.tokenizer = self.dl_t['tokenizer']
        self.train_sampler = self.dl_t.get('train_sampler')
        self.test_sampler = self.dl_t.get('test_sampler')

        self.model = Transformer(
            src_vocab_size=len(self.tokenizer),
            tgt_vocab_size=len(self.tokenizer),
            d_model = cfg.model.d_model,
            num_heads = cfg.model.num_heads,
            num_layers = cfg.model.num_layers,
            d_ff = cfg.model.d_model * cfg.model.mlp_mult,
            max_seq_length=cfg.data.max_len,
            dropout = cfg.model.dropout
        ).to(self.device)

        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,
            )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=cfg.optim.lr,
        )
        self.epoch_ = 0
        self.step = 0

        if self.is_main_process:
            print(f"Training configuration:")
            print(f"  World size: {self.world_size}")
            print(f"  Using device: {self.device}")
            print(f"  Checkpoint dir: {self.checkpoint_dir}")

    def _compute_loss(self, logits, tgts) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
        )
        loss = criterion(logits.reshape(-1, len(self.tokenizer)), tgts[:, 1:].reshape(-1))

        return loss

    def _get_lr(self,) -> float: 
        if self.step < self.cfg.optim.warmup:
            return self.cfg.optim.lr * (self.step / self.cfg.optim.warmup)
        
        else:
            total_steps = self.cfg.optim.n_iters
            eta_min = self.cfg.optim.eta_min_ratio * self.cfg.optim.lr
            cosine_decay = 0.5 * (
                1 + math.cos(math.pi * (self.step - self.cfg.optim.warmup) / (total_steps - self.cfg.optim.warmup))
            )
            return eta_min + (self.cfg.optim.lr - eta_min) * cosine_decay
    
    def _set_lr(self,) -> float: 
        lr = self._get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr
    
    def _optimizer_step(self, loss: torch.Tensor):
        lr = self._set_lr()
        loss.backward()

        if (self.step + 1) % self.cfg.optim.get('accumulation_steps', 1) == 0:
            if self.cfg.optim.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.cfg.optim.grad_clip
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            'loss': loss.item(),
            'lr': lr
        }

    def _train_step(self, batch: Dict[str, Any]):
        self.model.train()
        src, tgt, src_mask, tgt_mask = self._get_batch_data(batch)
        logits = self.model(src, tgt[:, :-1], src_mask, tgt_mask)
        
        loss = self._compute_loss(logits, tgt)
        log_dict = self._optimizer_step(loss)
        # if self.is_main_process and (self.step+1) % 100 == 0:
        #     print(f'Epoch', self.epoch_, 'Step', self.step, f'Loss {loss.item():.3f}')

        self.step += 1

        return log_dict
    
    def _valid_step(self, batch: Dict[str, Any]):
        self.model.eval()
        with torch.no_grad():
            src, tgt, src_mask, tgt_mask = self._get_batch_data(batch)
            tgt_tokens = [[self.tokenizer.bos_token_id] for _ in range(src.shape[0])]
            # if self.is_main_process:
            #     bar = tqdm(total=self.cfg.data.max_len-1)
            for i in range(self.cfg.data.max_len-1):
                
                temp_tgt = torch.tensor(tgt_tokens, dtype=torch.long).to(tgt.device)
                temp_tgt_mask = create_mask(src, temp_tgt, self.tokenizer.pad_token_id)[1]

                logits = self.model(src, temp_tgt, src_mask, temp_tgt_mask)

                # greedy
                if self.cfg.generate.topk == 0:
                    next_tokens = logits.argmax(2)[:, -1]
                # topk
                else:
                    topk_values, topk_indices = torch.topk(logits[:, -1, :], k=self.cfg.generate.topk, dim=-1)
                    topk_probs = torch.softmax(topk_values, dim=-1)
                    next_tokens = torch.multinomial(topk_probs, num_samples=1)
                    next_tokens = topk_indices.gather(dim=-1, index=next_tokens).squeeze(-1)

                for b in range(src.shape[0]):
                    if tgt_tokens[b][-1] != self.tokenizer.eos_token_id and \
                        tgt_tokens[b][-1] != self.tokenizer.pad_token_id:
                        tgt_tokens[b].append(next_tokens[b].item())
                    else:
                        tgt_tokens[b].append(self.tokenizer.pad_token_id)
            #     if self.is_main_process:
            #         bar.update(1)
            # if self.is_main_process:
            #     bar.close()
        perplexity = compute_perplexity(logits, tgt, pad_id=self.tokenizer.pad_token_id)

        # if self.is_main_process:
        #     for b in range(src.shape[0]):
        #         tgt_token_ids = torch.tensor(tgt_tokens[b])
        #         generated_text = decode(tgt_token_ids, self.tokenizer)
        #         question = decode(src[b], self.tokenizer)
        #         raw_text = decode(tgt[b], self.tokenizer)
        #         print(f'Question Text: {question}')
        #         print(f'Gen Answer: {generated_text}')
        #         print(f'Raw Answer: {raw_text}')
        #         print()
            
        return perplexity

        
    def epoch(self, ):
        # 训练阶段
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch_)

        total_loss = 0.0
        total_count = 0

        if self.is_main_process:
            bar = tqdm(total=len(self.train_loaders), desc=f"Epoch {self.epoch_}")

        for batch in self.train_loaders:
            log_dict = self._train_step(batch)
            total_loss += log_dict['loss']
            total_count += 1
            if self.is_main_process:
                bar.update(1)
                bar.set_postfix(loss=log_dict['loss'], lr=log_dict['lr'])
        
        if self.is_main_process:
            bar.close()

        # 收集所有进程的损失
        if self.world_size > 1:
            total_loss_tensor = torch.tensor([total_loss]).to(self.device)
            count_tensor = torch.tensor([total_count]).to(self.device)
            if self.is_main_process:
                print('before all reduce', time())

            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            
            if self.is_main_process:
                print('after all reduce', time())
            mean_loss = total_loss_tensor.item() / count_tensor.item()
        else:
            mean_loss = total_loss / total_count

        if self.is_main_process:
            print(f'Epoch {self.epoch_}, mean loss: {mean_loss:.3f}')

        # 验证阶段
        if self.test_sampler:
            self.test_sampler.set_epoch(self.epoch_)

        total_perp = 0.0
        valid_count = 0

        for batch in self.test_loaders:
            perp = self._valid_step(batch)
            total_perp += perp
            valid_count += 1

        # 收集所有进程的困惑度
        if self.world_size > 1:
            total_perp_tensor = torch.tensor([total_perp]).to(self.device)
            valid_count_tensor = torch.tensor([valid_count]).to(self.device)
            
            dist.all_reduce(total_perp_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(valid_count_tensor, op=dist.ReduceOp.SUM)
            
            mean_perplexity = total_perp_tensor.item() / valid_count_tensor.item()
        else:
            mean_perplexity = total_perp / valid_count

        # 更新最佳模型
        is_best = False
        if mean_perplexity < self.best_perplexity:
            self.best_perplexity = mean_perplexity
            is_best = True

        if self.is_main_process:
            print(f'Epoch {self.epoch_}, mean perplexity: {mean_perplexity:.3f}')
            
            self.save_checkpoint(self.epoch_, self.step, is_best)
        
        self.epoch_ += 1

    def run(self, epochs=10):
        for _ in range(epochs):
           self.epoch()

        if self.world_size > 1:
           dist.destroy_process_group()
           

    def _get_batch_data(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 src, tgt, src_mask, tgt_mask"""

        src, tgt =  batch['question_input_ids'].to(self.device), batch['answer_input_ids'].to(self.device)
        src_mask, tgt_mask = create_mask(src, tgt[:, :-1], self.tokenizer.pad_token_id)
        return src, tgt, src_mask, tgt_mask

    def save_checkpoint(self, epoch: int, step: int, is_best: bool = False):
        """保存模型checkpoint"""

        if self.world_size > 1:
            dist.barrier()
        if self.is_main_process:
        
            # 获取模型状态（如果是DDP，需要获取原始模型）
            if isinstance(self.model, DDP):
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()
            
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': OmegaConf.to_container(self.cfg),
                'best_perplexity': self.best_perplexity,
                'tokenizer_vocab_size': len(self.tokenizer),
            }
            
            # # 保存常规checkpoint
            # checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_step_{step}.pt'
            # torch.save(checkpoint, checkpoint_path)
            
            # 保存最新checkpoint
            latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
            torch.save(checkpoint, latest_path)
            
            # 如果是最好模型，额外保存
            if is_best:
                best_path = self.checkpoint_dir / 'best_model.pt'
                torch.save(checkpoint, best_path)
                print(f"Saved best model with perplexity: {self.best_perplexity:.4f}")
            
            print(f"Checkpoint saved at epoch {epoch}, step {step}")
        if self.world_size > 1:
            dist.barrier()

    
    def load_checkpoint(self, checkpoint_path: str = None):
        """加载模型checkpoint"""
        if checkpoint_path is None:
            # 尝试加载最新的checkpoint
            checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
            if not checkpoint_path.exists():
                print("No checkpoint found, starting from scratch")
                return
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型状态
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载训练状态
        self.epoch_ = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_perplexity = checkpoint.get('best_perplexity', float('inf'))
        
        print(f"Loaded checkpoint: epoch {self.epoch_}, step {self.step}, "
              f"best perplexity {self.best_perplexity:.4f}")
    
