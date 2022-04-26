import torch
import os
import numpy as np

from torch import nn
from torch.nn import functional  as F
from itertools import cycle
from logger.tensorboard import TensorboardWriter


class BaseTrainer:
    def __init__(
                    self, 
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
                    use_amp):
        self.dist = dist
        self.rank = rank
        self.config = config
        self.resume = resume
        self.preload = preload
        self.start_epoch = 0
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.use_distill = False
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(
          init_scale=10.0, 
          growth_factor=2.0, 
          backoff_factor=0.5, 
          growth_interval=200, 
          enabled=self.use_amp)
        self.completed_steps = 0
        

        self.save_checkpoint_interval = config["trainer"]["args"]["save_checkpoint_interval"]
        self.validation_interval = config["trainer"]["args"]["validation_interval"]
        self.save_max_metric_score = config["trainer"]["args"]["save_max_metric_score"]
        self.best_score = -np.inf if self.save_max_metric_score else np.inf

        if preload is not None:
            self._preload_model(preload)
        if resume:
            self._resume_checkpoint()

        if self.rank == 0:
            self.writer = TensorboardWriter(self.log_dir)
            self._print_networks([self.model])

    @staticmethod
    def _print_networks(models: list):
        print(f"This project contains {len(models)} models, the number of the parameters is: ")

        params_of_all_networks = 0
        for idx, model in enumerate(models, start=1):
            params_of_network = 0
            for param in model.parameters():
                params_of_network += param.numel()

            print(f"\tNetwork {idx}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")

    def _preload_model(self, model_path):
        """
        Preload model parameters (in "*.tar" format) at the start of experiment.
        Args:
            model_path (Path): The file path of the *.tar file
        """
        model_path = model_path
        assert model_path.exists(), f"The file {model_path} is not exist. please check path."

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        model_checkpoint = torch.load(model_path, map_location=map_location)
        self.model.load_state_dict(model_checkpoint["model"], strict=False)

        if self.rank == 0:
            print(f"Model preloaded successfully from {model_path}.")

    def _resume_checkpoint(self):
        """
        Resume experiment from the latest checkpoint.
        """
        latest_model_path = os.path.join(self.save_dir, "latest_model.tar")
        print("Loading model from ", latest_model_path)
        assert os.path.exists(latest_model_path), f"{latest_model_path} does not exist, can not load latest checkpoint."

        self.dist.barrier()  # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(latest_model_path, map_location=map_location)

        self.start_epoch = checkpoint["epoch"] + 1
        self.completed_steps = checkpoint["completed_steps"] 
        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.model.load_state_dict(checkpoint["model"], strict = False)
        self.scaler.load_state_dict(checkpoint["scaler"])

        if self.rank == 0:
            print(f"Model checkpoint loaded. Training will begin at {self.start_epoch + 1} epoch and {self.completed_steps + 1} step")

    def _save_checkpoint(self, epoch, is_best_epoch=False):
        """
        Save checkpoint to "<save_dir>/<config name>/checkpoints" directory, which consists of:
            - epoch
            - best metric score in historical epochs
            - optimizer parameters
            - model parameters
        Args:
            is_best_epoch (bool): In the current epoch, if the model get a best metric score (is_best_epoch=True),
                                the checkpoint of model will be saved as "<save_dir>/checkpoints/best_model.tar".
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")

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
        torch.save(state_dict, os.path.join(self.save_dir, f"model_{str(epoch).zfill(4)}.tar"))

        # If the model get a best metric score (is_best_epoch=True) in the current epoch,
        # the model checkpoint will be saved as "best_model.tar."
        # The newer best-scored checkpoint will overwrite the older one.
        if is_best_epoch:
            torch.save(state_dict, os.path.join(self.save_dir, "best_model.tar"))

    def _is_best_epoch(self, score, save_max_metric_score=True):
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

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            self._train_epoch(epoch)

            if epoch % self.validation_interval == 0:
                print("Training has finished, validation is in progress...")
                self.model.eval()
                score = self._valid_epoch(epoch)
                if self._is_best_epoch(score, save_max_metric_score=self.save_max_metric_score):
                    # Since we start from 0, add 1 to epoch 
                    self._save_checkpoint(epoch+1, is_best_epoch=True)
                

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

   