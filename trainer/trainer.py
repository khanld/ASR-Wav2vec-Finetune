import torch
import os
import numpy as np

from torch import nn
from torch.nn import functional  as F
from itertools import cycle
from base.base_trainer import BaseTrainer
from tqdm import tqdm
from logger.pbar import PBar
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from torch.cuda.amp import autocast

class Trainer(BaseTrainer):
    def __init__(self, 
                dist,
                rank,
                n_gpus,
                config,
                resume,
                preload,
                epochs,
                model,
                metric,
                processor,
                train_dl,
                val_dl,
                optimizer,
                scheduler,
                save_dir,
                log_dir,
                gradient_accumulation_steps,
                use_amp
                ):
        super(Trainer, self).__init__(
                                        dist, 
                                        rank, 
                                        config,
                                        resume, 
                                        preload, 
                                        epochs, 
                                        model, 
                                        optimizer, 
                                        scheduler,
                                        save_dir, 
                                        log_dir,
                                        use_amp)
        self.metric = metric
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.sr = config["meta"]["sr"]
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.n_gpus = n_gpus
        self.processor = processor

    def get_grad_norm(self, params, scale=1):
        """Compute grad norm given a gradient scale."""
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                param_norm = (p.grad.detach().data / scale).norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def preprocess_data(self, features, transcripts):
        batch = self.processor(features, sampling_rate=self.sr, padding="longest", return_tensors="pt")
    
        with self.processor.as_target_processor():
            labels_batch = self.processor(transcripts, padding="longest", return_tensors="pt")

        batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        return batch

    def gather(self, value: torch.tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        output_tensors = [tensor.clone() for _ in range(self.dist.get_world_size())]
        self.dist.all_gather(output_tensors, tensor)
        return torch.cat(output_tensors, dim=0)

    def _train_epoch(self, epoch):
        pbar_step = 1
        if self.rank == 0:
            print("Epoch {}: ".format(epoch+1))
            pbar = PBar(int(len(self.train_dl) // self.gradient_accumulation_steps + (len(self.train_dl) % self.gradient_accumulation_steps != 0)), 10)

        with torch.autograd.set_detect_anomaly(True):
            for step, (features, transcripts) in enumerate(self.train_dl):
                batch = self.preprocess_data(features, transcripts)
                with autocast(enabled = self.use_amp):
                    # forward
                    outputs = self.model(**batch)
                    # divide loss by gradient accumulation steps since gradients
                    # are accumulated for multiple backward passes in PyTorch
                    loss = outputs.loss / self.gradient_accumulation_steps
                    wer = self.metric(outputs.logits.detach(), batch['labels'])

                loss = self.scaler.scale(loss)
                loss.backward()


                 # update step
                if (step + 1) % self.gradient_accumulation_steps == 0 or step == len(self.train_dl) - 1:
                    # compute grad norm for monitoring
                    grad_norm = self.get_grad_norm(self.model.parameters(), scale = self.scaler.get_scale())
                    #gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)



                    # update parameters
                    self.optimizer.zero_grad()
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                    is_overflown = scale_after < scale_before
                    if is_overflown:
                        print("-----Skip update gradients, encounter overflow-----")
                    else:
                      self.scheduler.step()
                    

                    # Logging
                    if self.n_gpus > 1:
                        loss = self.dist.all_gather(loss).mean()
                        wer = self.dist.all_gather(wer).mean()

                    train_logs = {
                        "loss": loss.item() * self.gradient_accumulation_steps,
                        "lr": self.scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm,
                        "wer": wer
                    }
                    
                    if self.rank == 0:
                        # write logs
                        self.writer.update(self.completed_steps, 'Train', train_logs)
                        pbar.update(pbar_step, "train_", train_logs)

                        # save the latest checkpoint in the current epoch
                        if self.save_checkpoint_interval != 0 and self.completed_steps % self.save_checkpoint_interval == 0:
                            self._save_checkpoint(epoch+1)

                    self.completed_steps += 1
                    pbar_step += 1
                


        
            

    def _valid_epoch(self, epoch):
        # init logs
        val_logs = {
            "loss": 0,
            "wer": 0
        }
        for step, (features, transcripts) in tqdm(enumerate(self.val_dl), total = len(self.val_dl)):
            with torch.no_grad():
                batch  = self.preprocess_data(features, transcripts)
                outputs = self.model(**batch)

            val_logs["loss"] += outputs.loss / len(self.val_dl)
            val_logs["wer"] += self.metric(outputs.logits, batch['labels']) / len(self.val_dl)

        # sum over devices in multi-processing
        if self.n_gpus > 1:
            val_logs = {k: self.dist.all_gather(v).mean() for k, v in val_logs.items()}
        val_logs = {k: v.item() if hasattr(v, 'item') else v for k, v in val_logs.items()}
        # write logs
        if self.rank == 0:
            self.writer.update(epoch, 'Validation', val_logs)
        
        return val_logs["wer"] 



   