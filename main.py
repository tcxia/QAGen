import linecache
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, TensorDataset

import subprocess
import pickle
import argparse

from qa_train import Trainer


def distributed_main(args):
    ngpus_per_node = len(args.devices)
    assert ngpus_per_node <= torch.cuda.device_count(
    ), "GPU device num exceeds max capacity"

    args.world_size = ngpus_per_node * args.world_size

    mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def worker(gpu, ngpus_per_node, args):
    trainer = Trainer(args)
    trainer.make_model_env(gpu, ngpus_per_node)
    trainer.train()


class HarvestingQADataset(Dataset):
    def __init__(self, filename, ratio) -> None:
        super().__init__()
        self.filename = filename
        self.total_size = int(
            int(
                subprocess.check_output("wc -l " + filename,
                                        shell=True).split()[0]) * ratio)

    def __getitem__(self, index: int):
        line = linecache.getline(self.filename, index + 1)
        str_loaded = line.split("\t")

        input_ids = str_loaded[0].split()
        input_mask = str_loaded[1].split()
        segment_ids = str_loaded[2].split()
        start_pos = str_loaded[3]
        end_pos = str_loaded[4]

        input_ids = torch.tensor([int(idx) for idx in input_ids],
                                 dtype=torch.long)
        input_mask = torch.tensor([int(idx) for idx in input_mask],
                                  dtype=torch.long)
        segment_ids = torch.tensor([int(idx) for idx in segment_ids],
                                   dtype=torch.long)
        start_pos = torch.tensor([int(start_pos)], dtype=torch.long)
        end_pos = torch.tensor([int(end_pos)], dtype=torch.long)

        return input_ids, input_mask, segment_ids, start_pos, end_pos

    def __len__(self) -> int:
        return self.total_size


def main(args):

    args.dev_features_file = "dev_features.pkl"
    args.dev_examples_file = "dev_examples.pkl"
    args.dev_json_file = "my_dev.json"

    args.test_features_file = "test_features.pkl"
    args.test_examples_file = "test_examples.pkl"
    args.test_json_file = "my_test.json"

    args.devices = [int(gpu) for gpu in args.devices.split('-')]
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    if args.lazy_loader:
        args.pretrain_dataset = HarvestingQADataset(args.pretrain_file,
                                                    args.unlabel_ratio)
    else:
        with open(args.pretrain_file, "rb") as f:
            features = pickle.load(f)

        features = features[:int(len(features) * args.unlabel_ratio)]
        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features],
                                      dtype=torch.long)
        all_seg_ids = torch.tensor([f.segment_ids for f in features],
                                   dtype=torch.long)
        all_start_pos = torch.tensor([f.start_pos for f in features],
                                     dtype=torch.long)
        all_end_pos = torch.tensor([f.end_pos for f in features],
                                   dtype=torch.long)
        args.pretrain_dataset = TensorDataset(all_input_ids, all_input_mask,
                                              all_seg_ids, all_start_pos,
                                              all_end_pos)

    with open(args.dev_examples_file, "rb") as f:
        args.dev_examples = pickle.load(f)
    with open(args.dev_features_file, "rb") as f:
        features = pickle.load(f)
    args.dev_features = features
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    all_seg_ids = torch.tensor([f.segment_ids for f in features],
                               dtype=torch.long)
    args.dev_dataset = TensorDataset(all_input_ids, all_input_mask,
                                     all_seg_ids)

    with open(args.test_examples_file, "rb") as f:
        args.test_examples = pickle.load(f)
    with open(args.test_features_file, "rb") as f:
        features = pickle.load(f)
    args.test_features = features
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    all_seg_ids = torch.tensor([f.segment_ids for f in features],
                               dtype=torch.long)
    args.test_dataset = TensorDataset(all_input_ids, all_input_mask,
                                      all_seg_ids)

    distributed_main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debugging mode")

    # preprocess option
    parser.add_argument("--max_seq_length",
                        default=384,
                        type=int,
                        help="max sequence length")
    parser.add_argument("--max_query_length",
                        default=64,
                        type=int,
                        help="max query length")
    parser.add_argument("--doc_stride", default=128, type=int)
    parser.add_argument("--do_lower_case",
                        default=True,
                        help="do lower case on text")

    # training option
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
    parser.add_argument("--pretrain_epochs",
                        default=2,
                        type=int,
                        help="number of epochs")
    parser.add_argument("--batch_size",
                        default=24,
                        type=int,
                        help="batch size")
    parser.add_argument("--pretrain_lr", default=5e-5, type=float)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--unlabel_ratio", default=0.1, type=float)

    # directory option
    parser.add_argument("--lazy_loader",
                        action="store_true",
                        help="lazy loader")
    parser.add_argument(
        "--pretrain_file",
        default=
        "../data/synthetic_data/vae98/harv1.0/0.2_replaced_1.0_harv_features.txt",
        type=str,
        help="path of training data file")
    # gpu option
    parser.add_argument("--use_cuda", default=True, help="use cuda or not")
    parser.add_argument("--devices",
                        type=str,
                        default='0_1_2_3',
                        help="gpu device ids to use")
    parser.add_argument(
        "--workers",
        default=4,
        help="Number of processes(workers) per node."
        "It should be equal to the number of gpu devices to use in one node")
    parser.add_argument(
        "--world_size",
        default=1,
        help=
        "Number of total workers. Initial value should be set to the number of nodes."
        "Final value will be Num.nodes * Num.devices")
    parser.add_argument("--rank",
                        default=0,
                        help="The priority rank of current node.")
    parser.add_argument(
        "--dist_backend",
        default="nccl",
        help=
        "Backend communication method. NCCL is used for DistributedDataParallel"
    )
    parser.add_argument("--dist_url",
                        default="tcp://127.0.0.1:9990",
                        help="DistributedDataParallel server")
    parser.add_argument("--multiprocessing_distributed",
                        default=True,
                        help="Use multiprocess distribution or not")
    parser.add_argument("--random_seed",
                        default=2019,
                        help="random state (seed)")
    args = parser.parse_args()

    main(args)