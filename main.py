import math

import numpy as np
from torch import nn
from torch.utils.data import WeightedRandomSampler

from MyDataloader import (
    load_server_data,
    load_client_weight_data,
    load_corrupt_client_data,
    load_non_iid_data,
    load_non_iid_class_data,
    mnist_non_iid,
)
import random
import torch
import torch.nn.functional as F
import logging
import os
import json
import sklearn

from meta import MetaSGD
from models import ResNet32, load_VNet, ModelCNNMnist
import argparse
import time

def tamper_weights_reverse(state_dict):
    # Reverse the sign of all weights
    return {k: -v for k, v in state_dict.items()}

def tamper_weights_large_neg(state_dict):
    # Set all weights to -9999
    return {k: torch.ones_like(v) * -9999 for k, v in state_dict.items()}

def tamper_weights_random(state_dict):
    # Replace weights with random noise
    return {k: torch.randn_like(v) for k, v in state_dict.items()}



def build_model(args_in):
    model = ModelCNNMnist()

    if torch.cuda.is_available():
        model.to(device)
        torch.backends.cudnn.benchmark = True

    return model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy using sklearn.metrics.accuracy_score"""
    _, pred = torch.max(output, 1)
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    acc = sklearn.metrics.accuracy_score(target_np, pred_np) * 100.0
    return [acc]


def client_train(train_loader, model, epoch, client_index):
    train_loss = 0
    train_acc = []
    optimizer_model = client_optimizer_list[client_index]

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model = model.to(device)
        model.train()
        inputs, targets = inputs.to(device), targets.to(device).long()

        outputs = model(inputs)

        train_acc.append(accuracy(outputs.data, targets.data, topk=(1,))[0])

        loss = criterion(outputs, targets)
        loss_regular = 0.0
        loss = loss + loss_regular
        optimizer_model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer_model.step()
        train_loss += loss.item()

        if (batch_idx + 1) == len(train_loader):
            logger.info(
                f"Epoch: {epoch + 1}\t"
                f"Client: {client_index}\t"
                f"Batch: {batch_idx + 1}\t"
                f"Loss: %.4f\t"
                f"Prec@1 %.2f"
                % ((train_loss / (batch_idx + 1)), sum(train_acc) / len(train_acc))
            )


def server_train(client_loader, train_loader, model, vnet, epoch):
    meta_loss = 0
    meta_acc = []
    vnet.train()
    for batch_idx, (inputs_val, targets_val) in enumerate(train_loader):
        meta_model = build_model(args)
        meta_model.load_state_dict(model.state_dict())
        for inputs, targets in client_loader:
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = meta_model(inputs)
            cost = F.cross_entropy(outputs, targets, reduce=False)
            cost_v = torch.reshape(cost, (len(cost), 1))
            v_lambda = vnet(cost_v.data)
            l_f_meta = torch.sum(cost_v * v_lambda) / len(cost_v)

            grads = torch.autograd.grad(
                l_f_meta, meta_model.parameters(), create_graph=True
            )
            meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))
            pseudo_optimizer = MetaSGD(
                meta_model, meta_model.parameters(), lr=meta_lr, momentum=args.momentum
            )
            pseudo_optimizer.meta_step(grads)
            del grads
            break

        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        y_g_hat = meta_model(inputs_val)
        l_g_meta = F.cross_entropy(y_g_hat, targets_val.long())
        meta_acc.append(accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0])

        server_optimizer.zero_grad()
        l_g_meta.backward()

        server_optimizer.step()

        meta_loss += l_g_meta.item()

        if (batch_idx + 1) == len(train_loader):
            logger.info(
                f"Epoch: {epoch + 1}\t"
                f"Server: 0\t"
                f"Batch: {batch_idx + 1}\t"
                f"Loss: %.4f\t"
                f"Prec@1 %.2f"
                % ((meta_loss / (batch_idx + 1)), sum(meta_acc) / len(meta_acc))
            )


def server_eval(train_loader, model, epoch):
    server_loss = 0
    server_acc = []
    all_targets = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs_val, targets_val) in enumerate(train_loader):
            inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
            y_g_hat = model(inputs_val)

            l_g_meta = F.cross_entropy(y_g_hat, targets_val)
            server_acc.append(accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0])

            server_loss += l_g_meta.item()

            # Collect predictions and targets for metrics
            _, predicted = torch.max(y_g_hat.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets_val.cpu().numpy())

            if (batch_idx + 1) == len(train_loader):
                acc = sum(server_acc) / len(server_acc)
                precision = sklearn.metrics.precision_score(all_targets, all_preds, average="macro", zero_division=0)
                recall = sklearn.metrics.recall_score(all_targets, all_preds, average="macro", zero_division=0)
                f1 = sklearn.metrics.f1_score(all_targets, all_preds, average="macro", zero_division=0)

                logger.info("eval")
                logger.info(
                    f"Epoch: {epoch + 1}\t"
                    f"Server: 0\t"
                    f"Batch: {batch_idx + 1}\t"
                    f"Loss: %.4f\t"
                    f"Acc: %.2f\t"
                    f"Prec: %.2f\t"
                    f"Recall: %.2f\t"
                    f"F1: %.2f"
                    % (
                        (server_loss / (batch_idx + 1)),
                        acc,
                        precision * 100,
                        recall * 100,
                        f1 * 100,
                    )
                )


def meta_train():
    # train
    # select client
    for epoch in range(0, args.epochs):
        logger.info(f"Epoch: {epoch + 1}")
        loss_sum = [torch.tensor(0.0) for i in range(client_num)]
        weight_sum = [0.0 for i in range(client_num)]
        if args.select_client is True and epoch != 0:  # !=
            for idx in range(client_num):
                server_model.eval()
                server_meta_model.eval()
                # cal loss_sum
                for inputs, targets in client_data_loader[idx]["train"]:
                    with torch.no_grad():
                        inputs, targets = inputs.to(device), targets.to(device).long()
                        outputs = server_model(inputs)

                        loss = F.cross_entropy(outputs, targets, reduce=False)
                        weight_sum[idx] += sum(
                            server_meta_model(torch.reshape(loss, (len(loss), 1)))
                        ).item()
            client_weight = torch.ones(client_num)
            client_avg_loss = torch.ones(client_num)
            for k in range(client_num):
                client_weight[k] = (
                    weight_sum[k]
                    / len(client_data_loader[k]["train"])
                    / args.batch_size
                )
            print(f"client_weight: {client_weight}")

            client_select_list = list(
                WeightedRandomSampler(
                    client_weight, client_select_num, replacement=False
                )
            )
        else:
            client_select_list = random.sample(range(0, client_num), client_select_num)

        # client
        for j in client_select_list:
            # get global_model from server
            client_model_list[j].load_state_dict(server_model.state_dict())
            # get MetaModel from server
            client_meta_model.load_state_dict(server_meta_model.state_dict())
            # get loss of trainSet from Model
            client_train_set = client_data_loader[j]["train"].dataset
            client_train_weight = torch.ones(0)
            client_train_loss = torch.ones(0)
            start_time = time.time()
            print("start time: ", start_time)
            if epoch != 0:  # !=
                client_model_list[j].eval()
                client_meta_model.eval()
                for index, (data, label) in enumerate(client_data_loader[j]["train"]):
                    # client_train_loss
                    with torch.no_grad():
                        data, label = data.to(device), label.to(device).long()
                        y_pred = client_model_list[j](data)
                        loss = F.cross_entropy(y_pred, label, reduce=False)
                        # loss -> MetaModel -> weight
                        client_train_loss = torch.cat(
                            (client_train_loss, loss.cpu()), -1
                        )
                        weight = client_meta_model(torch.reshape(loss, (len(loss), -1)))
                        weight = torch.reshape(weight, loss.shape)
                        client_train_weight = torch.cat(
                            (client_train_weight, weight.cpu()), -1
                        )
            else:
                client_train_weight = torch.ones(len(client_train_set))

            # use weight to create dataLoader
            if args.gaussian != 0:
                sigma = math.sqrt(2 * math.log(1.25 / 1e-5, math.e)) / args.gaussian
                gaussian_weight = torch.normal(
                    0, sigma, size=(1, len(client_train_weight))
                )
                gaussian_weight = gaussian_weight.view(len(client_train_weight))
                client_train_weight = client_train_weight + gaussian_weight
                client_train_weight = torch.clamp(client_train_weight, 0)
            client_data_loader[j]["meta_train"] = load_client_weight_data(
                args.dataset_name,
                client_num,
                args.batch_size,
                client_train_weight,
                j,
                client_data_loader[j]["train"],
            )
            client_train(
                client_data_loader[j]["meta_train"], client_model_list[j], epoch, j
            )
            end_time = time.time()
            print("middle time: ", end_time - start_time, "s")
        # server
        for name, params in server_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        for idx, j in enumerate(client_select_list):
            # Tamper weights if client is malicious
            if args.corrupt_num > 0 and j < args.corrupt_num:
                if args.weight_tampering == "reverse":
                    tampered_state = tamper_weights_reverse(client_model_list[j].state_dict())
                elif args.weight_tampering == "large_neg":
                    tampered_state = tamper_weights_large_neg(client_model_list[j].state_dict())
                elif args.weight_tampering == "random":
                    tampered_state = tamper_weights_random(client_model_list[j].state_dict())
                else:
                    tampered_state = client_model_list[j].state_dict()
                for name, params in tampered_state.items():
                    weight_accumulator[name] += params
            else:
                for name, params in client_model_list[j].state_dict().items():
                    weight_accumulator[name] += params
        for name in weight_accumulator.keys():
            weight_accumulator[name] = weight_accumulator[name] / client_select_num
        server_model.load_state_dict(weight_accumulator)

        # get train loss
        client_i = client_select_list[0]
        client_meta_model.load_state_dict(server_meta_model.state_dict())
        client_model_list[client_i].load_state_dict(server_model.state_dict())
        # train meta model
        server_train(
            client_data_loader[client_i]["train"],
            server_data_loader["train"],
            server_model,
            server_meta_model,
            epoch,
        )
        server_eval(server_data_loader["val"], server_model, epoch)


def server_test(train_loader, model):
    server_acc = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs_val, targets_val) in enumerate(train_loader):
            inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
            y_g_hat = model(inputs_val)

            server_acc.append(accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0])
        acc = sum(server_acc) / len(server_acc)
        return acc


def avg_loss_on_data(train_loader, model):
    train_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()

    return train_loss / (len(train_loader) * args.batch_size)


def logging_config():
    logger.info(f"Config:")
    logger.info(f"\tselect_client: {args.select_client}")
    logger.info(f"\tcorrupt_num: {args.corrupt_num}")
    logger.info(f"\tclient_num: {args.client_num}")
    logger.info(f"\tselect_ratio: {args.select_ratio}")
    logger.info(f"\tdataset_name: {args.dataset_name}")
    logger.info(f"\tvalidation_num: {args.validation_num}")
    logger.info(f"\tnoniid_ratio: {args.noniid_ratio}")
    logger.info(f"\ttrain_type: {args.train_type}")
    logger.info(f"\ttest_name: {args.test_name}")
    logger.info(f"\tcorruption_prob: {args.corruption_prob}")
    logger.info(f"\tepochs: {args.epochs}")
    logger.info(f"\tbatch_size: {args.batch_size}")
    logger.info(f"\tlr: {args.lr}")
    logger.info(f"\tmomentum: {args.momentum}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch WideResNet Training")
    parser.add_argument(
        "--select_client",
        default=False,
        action="store_true",
        help="select client by meta",
    )
    parser.add_argument("--client_num", type=int, default=100)
    parser.add_argument("--select_ratio", type=float, default=0.4)
    parser.add_argument("--corrupt_num", type=int, default=-10)
    parser.add_argument(
        "--corruption_type",
        default="random",
        type=str,
        choices=["random", "reverse", "zero"],
        help="corruption type",
    )
    parser.add_argument("--weight_tampering", type=str, default="none", choices=["none", "reverse", "large_neg", "random"], help="weight tampering type")
    parser.add_argument(
        "--dataset_name",
        default="mnist",
        type=str,
        help="dataset (cifar10 [default] or mnist or cifar100)",
    )
    parser.add_argument(
        "--validation_num",
        type=int,
        default=1000,
        help="number of server validation set",
    )
    parser.add_argument(
        "--noniid_ratio",
        type=float,
        default=0.7,
        help="class ratio of each client in non-iid situation",
    )

    parser.add_argument(
        "--noniid_class_ratio",
        type=float,
        default=0.7,
        help="class ratio of each client in non-iid situation",
    )

    parser.add_argument("--imbalanced_factor", type=float, default=None)
    parser.add_argument("--imbalanced_server", type=float, default=None)

    parser.add_argument("--gaussian", type=float, default=0)

    parser.add_argument(
        "--train_type",
        default="normal",
        type=str,
        help="normal,meta,sampleloss,clientloss," "focalloss,shapley",
    )
    parser.add_argument("--test_name", default="test", type=str)
    parser.add_argument(
        "--corruption_prob", type=float, default=0.4, help="label noise"
    )
    parser.add_argument(
        "--epochs", default=200, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--batch_size", default=100, type=int, help="mini-batch size (default: 100)"
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1)
    parser.set_defaults(augment=True)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    np.random.seed(args.seed)

    # logger
    os.makedirs(f"./results/{args.dataset_name}/{args.test_name}", exist_ok=True)
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    test_log = logging.FileHandler(
        f"./logs/{args.dataset_name}/{args.test_name}/{args.test_name}.log",
        "a",
        encoding="utf-8",
    )
    test_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter("")
    test_log.setFormatter(formatter)
    logger.addHandler(test_log)

    KZT = logging.StreamHandler()
    KZT.setLevel(logging.DEBUG)
    formatter = logging.Formatter("")
    KZT.setFormatter(formatter)
    logger.addHandler(KZT)

    # log config
    logging_config()

    # weight dict of meta
    weight_dict = {}
    loss_dict = {}
    class_acc = {}

    # init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_num = args.client_num
    client_select_num = int(client_num * args.select_ratio)

    if args.dataset_name == "cifar100":
        class_num = 100
    elif args.dataset_name == "cifar10" or args.dataset_name == "mnist":
        class_num = 10

    # data
    if args.noniid_ratio == -1 and args.noniid_class_ratio == -1:
        client_data_loader = load_corrupt_client_data(
            args,
            client_num,
            corruption_type=args.corruption_type,
            corruption_ratio=args.corruption_prob,
            corrupt_num=args.corrupt_num,
            imbalanced_factor=args.imbalanced_factor,
        )
    elif args.noniid_ratio != -1:
        client_data_loader = mnist_non_iid(
            args,
            client_num,
            corruption_type=args.corruption_type,
            corruption_ratio=args.corruption_prob,
            corrupt_num=args.corrupt_num,
        )

    elif args.noniid_class_ratio != -1:
        client_data_loader = load_non_iid_class_data(
            args,
            client_num,
            corruption_type=args.corruption_type,
            corruption_ratio=args.corruption_prob,
            corrupt_num=args.corrupt_num,
        )
    server_data_loader = load_server_data(args)

    # model
    client_model_list = [build_model(args) for i in range(0, client_num)]
    server_model = build_model(args)
    server_meta_model = load_VNet().to(device)
    client_meta_model = load_VNet().to(device)

    # optimizer
    client_optimizer_list = [
        torch.optim.SGD(
            client_model_list[i].parameters(), args.lr, momentum=args.momentum
        )
        for i in range(0, client_num)
    ]
    server_optimizer = torch.optim.Adam(
        server_meta_model.parameters(), 1e-3, weight_decay=1e-4
    )

    weight_accumulator = {}
    for name, params in server_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(params)

    criterion = nn.CrossEntropyLoss().to(device)

    meta_train()
