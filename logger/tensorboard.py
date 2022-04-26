from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir, max_queue=5, flush_secs=30)
    
    def add_scalar(self, tag, scalar_value, global_step):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def update(self, step, mode, scores: dict):
        for k, v in scores.items():
            self.writer.add_scalar(mode + '/' + k, v, step)
        