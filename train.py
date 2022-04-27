import argparse
import json
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import toml


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from time import gmtime, strftime
from utils.utils import *
from utils.metric import Metric
from dataloader.dataset import DefaultCollate
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '3456'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, config, resume, preload):
    os.environ['CUDA_VISIBLE_DEVICES']=config["meta"]["device_ids"]
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    set_seed(config["meta"]["seed"])
    setup(rank, world_size)
    
    config['val_dataset']['args']['sr'] = config['meta']['sr']
    config['train_dataset']['args']['sr'] = config['meta']['sr']

    epochs = config["meta"]["epochs"]
    gradient_accumulation_steps = config["meta"]["gradient_accumulation_steps"]
    use_amp = config["meta"]["use_amp"]
    max_clip_grad_norm = config["meta"]["max_clip_grad_norm"]
    save_dir =  os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '/checkpoints')
    log_dir = os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '/log_dir')
    
    if rank == 0:
        # Creatr dirs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Store config file
        config_name = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(' ', '_') + '.json'
        with open(os.path.join(config["meta"]["save_dir"], config["meta"]['name'] + '/' + config_name), 'w+') as f:
            json.dump(config, f)
            f.close()

    default_collate = DefaultCollate(processor, config['meta']['sr'])
    
    # Create train dataloader
    train_base_ds = initialize_module(config["train_dataset"]["path"], args=config["train_dataset"]["args"])
    train_ds = train_base_ds.get_data()
    # If val_size is provided, get the val_ds by:
    # val_ds  = train_base_ds.get_data(mode = 'val')
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        **config["train_dataset"]["sampler"]
    )
    train_dl = DataLoader(
        dataset=train_ds,
        **config["train_dataset"]["dataloader"],
        sampler = train_sampler,
        collate_fn=default_collate
    )

    # If you have already split the dataset, process the validation data separately as below
    # Create val dataloader
    val_base_ds = initialize_module(config["val_dataset"]["path"], args=config["val_dataset"]["args"])
    val_ds = val_base_ds.get_data()
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        **config["val_dataset"]["sampler"]
    )
    val_dl = DataLoader(
        dataset=val_ds,
        **config["val_dataset"]["dataloader"],
        sampler = val_sampler,
        collate_fn=default_collate
    )

    # Generate the vocab_dict file
    vocab_dict = train_base_ds.get_vocab_dict()
    with open('vocab.json', 'w') as f:
        json.dump(vocab_dict, f)

    # Create processor
    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Load pretrained model
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base", 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        gradient_checkpointing=False
    )

    model.freeze_feature_encoder()
    # DDP for multi-processing
    model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=True)
        

    # Set up metric, scheduler, optmizer
    compute_metric = Metric(processor)

    optimizer = torch.optim.AdamW(
        params = model.parameters(),
        lr = config["optimizer"]["lr"]
    )
  
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config["scheduler"]["max_lr"], 
        epochs=epochs, 
        steps_per_epoch=(len(train_dl)//gradient_accumulation_steps) + (len(train_dl)%gradient_accumulation_steps != 0))


    if rank == 0:
        print(vocab_dict)
        print("Number of training utterances: ", len(train_ds))
        print("Number of validation utterances: ", len(val_ds))

    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)
    trainer = trainer_class(
        dist = dist,
        rank = rank,
        n_gpus = world_size,
        config = config,
        resume = resume,
        preload = preload,
        epochs = epochs,
        model = model,
        compute_metric = compute_metric,
        processor = processor,
        train_dl = train_dl,
        val_dl = val_dl,
        train_sampler = train_sampler,
        val_sampler = val_sampler,
        optimizer = optimizer,
        scheduler = scheduler,
        save_dir = save_dir,
        log_dir = log_dir,
        gradient_accumulation_steps = gradient_accumulation_steps,
        use_amp = use_amp,
        max_clip_grad_norm = max_clip_grad_norm
    )
    trainer.train()
    # cleanup()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR')
    args.add_argument('-c', '--config', required=True, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', action="store_true",
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-p', '--preload', default=None, type=str,
                      help='Path to pretrained Model')            
    
    args = args.parse_args()
    config = toml.load(args.config)
    n_gpus = len(config['meta']["device_ids"].split(','))

    mp.spawn(
        main,
        args = (n_gpus, config, args.resume, args.preload),
        nprocs = n_gpus,
        join = True
    )

