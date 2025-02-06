from tracemalloc import start
import torch
import os
import numpy as np
from typing import Dict, List
import shutil

from logger.tensorboard import TensorboardWriter
from huggingface_hub import Repository


class BaseTrainer:
    def __init__(
                    self, 
                    dist, 
                    rank, 
                    config, 
                    resume, 
                    preload, 
                    epochs, 
                    steps_per_epoch,
                    model, 
                    processor,
                    train_dl,
                    val_dl,
                    train_sampler,
                    val_sampler,
                    optimizer, 
                    scheduler, 
                    save_dir, 
                    log_dir,
                    use_amp,
                    gradient_accumulation_steps):
        self.dist = dist
        self.rank = rank
        self.config = config
        self.resume = resume
        self.preload = preload
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.start_epoch = 0
        self.pbar_step = 0
        self.model = model
        self.processor = processor
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.use_distill = False
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.completed_steps = 0
        
        self.save_max_metric_score = config["trainer"]["args"]["save_max_metric_score"]
        self.best_score = -np.inf if self.save_max_metric_score else np.inf

        # Huggingface_hub
        if self.config["huggingface"]["push_to_hub"]:
            if self.config["huggingface"]["overwrite_output_dir"]:
                shutil.rmtree(config["huggingface"]["args"]["local_dir"])
            self.repo = Repository(**self.config["huggingface"]["args"])
            
        # save processor
        self.processor.save_pretrained(config["huggingface"]["args"]["local_dir"])

        if preload is not None:
            self._preload_model(preload)
        if resume:
            self._resume_checkpoint()

        if self.rank == 0:
            self.writer = TensorboardWriter(self.log_dir)
            self._count_parameters()
            self._count_trainable_parameters()

    def _count_trainable_parameters(self) -> None:
        print("Number of trainable params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6)

    def _count_parameters(self) -> None:
        params_of_network = 0
        for param in self.model.parameters():
            params_of_network += param.numel()
        print(f"The amount of parameters in the project is {params_of_network / 1e6} million.")

    def _push_to_hub(self, commit_message : str = "End of training") -> None:
        """
        Read https://huggingface.co/docs/hub/how-to-upstream#repository
        Args:
            commit_message: Message to commit
        """

        self.repo.git_pull()
        return_message = self.repo.push_to_hub(
            commit_message=commit_message, blocking=self.config["huggingface"]["blocking"], auto_lfs_prune=True
        )
        print(f"*****{return_message}*****")

            

    def _preload_model(self, model_path) -> None:
        """
        Preload model parameters (in "*.tar" format) at the start of experiment.
        Args:
            model_path: The file path of the *.tar file
        """
        assert os.path.exists(model_path), f"The file {model_path} is not exist. please check path."

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(model_path, map_location=map_location)


        self.start_epoch = checkpoint["epoch"] + 1
        self.completed_steps = checkpoint["completed_steps"] + 1
        self.best_score = checkpoint["best_score"]
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(checkpoint["model"], strict=True)
        else:
            self.model.load_state_dict(checkpoint["model"], strict=True)

        if self.rank == 0:
            print(f"Model preloaded successfully from {model_path}.")

    def _resume_checkpoint(self) -> None:
        """
        Resume experiment from the latest checkpoint.
        """
        latest_model_path = os.path.join(self.save_dir, "latest_model.tar")
        print("Loading model from ", latest_model_path)
        assert os.path.exists(latest_model_path), f"{latest_model_path} does not exist, can not load latest checkpoint."

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(latest_model_path, map_location=map_location)
        
        self.start_epoch = checkpoint["epoch"] + 1
        self.completed_steps = checkpoint["completed_steps"] + 1
        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(checkpoint["model"], strict=True)
        else:
            self.model.load_state_dict(checkpoint["model"], strict=True)
        self.scaler.load_state_dict(checkpoint["scaler"])

        if self.rank == 0:
            print("*****Note that any changes in your config file or your training dataset may cause the resume to run incorrectly*****")
            print(f"Start training at epoch {self.start_epoch}")


    def _save_checkpoint(self, epoch: int,  is_best_epoch: bool = False) -> None:
        """
        Save checkpoint to "<save_dir>" directory, which consists of:
        Args:
        - is_best_epoch (bool): In the current epoch, if the model get a best metric score (is_best_epoch=True),
                                the checkpoint of model will be saved as "<save_dir>/best_model.tar".
        """
        print(f"\n Saving model checkpoint...")

        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "completed_steps": self.completed_steps
        }

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict["model"] = self.model.module.state_dict()
        else:
            state_dict["model"] = self.model.state_dict()

        # "latest_model.tar"
        # Contains all checkpoint information, including the optimizer parameters, the model parameters, etc.
        # New checkpoint will overwrite the older one.
        torch.save(state_dict, os.path.join(self.save_dir, "latest_model.tar"))

        # "model_{epoch_number}.tar"
        # Contains all checkpoint information, like "latest_model.tar". However, the newer information will no overwrite the older one.
        torch.save(state_dict, os.path.join(self.save_dir, f"model_{str(epoch)}.tar"))

        # If the model get a best metric score (is_best_epoch=True) in the current epoch,
        # the model checkpoint will be saved as "best_model.tar."
        # The newer best-scored checkpoint will overwrite the older one.
        if is_best_epoch:
            torch.save(state_dict, os.path.join(self.save_dir, "best_model.tar"))
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.save_pretrained(self.config["huggingface"]["args"]["local_dir"])
            else:
                self.model.save_pretrained(self.config["huggingface"]["args"]["local_dir"])
            
            if self.config["huggingface"]["push_to_hub"] and self.config["huggingface"]["push_every_validation_step"]:
                self._push_to_hub("update_best_model", True)


    def _is_best_epoch(self, score, save_max_metric_score=True) -> bool:
        """
        Check if the current model got the best metric score
        """
        if save_max_metric_score and score >= self.best_score:
            self.best_score = score
            return True
        elif not save_max_metric_score and score <= self.best_score:
            self.best_score = score
            return True
        else:
            return False
        

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            self._train_epoch(epoch)

        if self.rank == 0 and self.config["huggingface"]["push_to_hub"] and not self.config["huggingface"]["push_every_validation_step"]:
                self._push_to_hub("update_best_model", True)


    def _train_epoch(self, epoch) -> None:
        raise NotImplementedError

    def _valid_epoch(self, epoch) -> None:
        raise NotImplementedError

   
