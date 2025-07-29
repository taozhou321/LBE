import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import yaml
from data_loaders import KT_DatasetWithBehavior
from models import (
    AKT_LBE1,
    AKT_LBE2
)
from train import model_train
from utils.config import ConfigNode as CN
from utils.file_io import PathManager
from scipy.sparse import load_npz
import json

def main(config):
    device_name = config.device_name
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    if torch.cuda.is_available():
        torch.set_default_device(device)


    model_name = config.model_name
    dataset_path = config.dataset_path
    data_name = config.data_name
    seed = config.seed

    np.random.seed(seed)
    torch.manual_seed(seed)

    data_path = os.path.join(dataset_path, data_name)
    q_mat_path = os.path.join(os.path.join(dataset_path, data_name), "q_mat.npz")
    q_mat_loaded = load_npz(q_mat_path)
    q_mat = q_mat_loaded.toarray()

    dataset_info_path = os.path.join(data_path, "dataset_info.json")
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    
    num_questions = dataset_info["num_questions"]
    num_skills = dataset_info["num_skills"]
    num_problems = dataset_info["num_problems"]

    train_config = config.train_config
    checkpoint_dir = config.checkpoint_dir

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ckpt_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, data_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    batch_size = train_config.batch_size
    eval_batch_size = train_config.eval_batch_size
    learning_rate = train_config.learning_rate
    optimizer = train_config.optimizer
    seq_len = train_config.seq_len
    min_len = train_config.min_len

    print("MODEL", model_name)

        
    if model_name == "akt_lbe1":
        model_config = config.lbe_config
        model_config["model_name"] = model_name
        model = AKT_LBE1(num_questions=num_questions, num_skills=num_skills, num_problems=num_problems, device=device, **model_config)
    elif model_name == "akt_lbe2":
        model_config = config.lbe_config
        model_config["model_name"] = model_name
        model = AKT_LBE2(num_questions=num_questions, num_skills=num_skills, num_problems=num_problems, device=device, **model_config)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    

    train_dataset = KT_DatasetWithBehavior(seq_len, separate_char=',', min_seq_len=min_len, q_mat=q_mat, data_path=data_path + f'/train_with_behaviors.txt', device=device)
    valid_dataset = KT_DatasetWithBehavior(seq_len, separate_char=',', min_seq_len=min_len, q_mat=q_mat, data_path=data_path + f'/valid_with_behaviors.txt', device=device)
    test_dataset = KT_DatasetWithBehavior(seq_len, separate_char=',', min_seq_len=min_len, q_mat=q_mat, data_path=data_path + f'/test_with_behaviors.txt', device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size)
        

    model = model.to(device)

    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate, weight_decay=model_config.l2)

    model_train(
        model,
        opt,
        train_loader,
        valid_loader, 
        test_loader,
        config,
    )

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="akt_lbe1",
        help="The name of the model to train."
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="assistments09",
        help="The name of the dataset to use in training.",
    )
    parser.add_argument(
        "--device_name",
        type=str,
        default="cuda:0",
        help="The name of the device to use in training.",
        choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    )
    parser.add_argument(
        "--batch_size", type=float, default=128, help="train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=float, default=128, help="eval batch size"
    )
    parser.add_argument("--l2", type=float, default=0.0, help="l2 regularization param")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    args = parser.parse_args()
    base_cfg_file = PathManager.open("configs/example.yaml", "r")
    base_cfg = yaml.safe_load(base_cfg_file)
    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)
    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.device_name = args.device_name
    cfg.train_config.batch_size = int(args.batch_size)
    cfg.train_config.eval_batch_size = int(args.eval_batch_size)
    cfg.train_config.learning_rate = args.lr
    cfg.train_config.optimizer = args.optimizer

    cfg.freeze()

    print(cfg)
    main(cfg)
