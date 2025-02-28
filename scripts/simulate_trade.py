import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import torch
import numpy as np
import seaborn as sns
from utils import io_tools
from utils.trade import trade
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform

sns.set_theme(style='whitegrid', context='paper', font_scale=2)
palette = sns.color_palette('muted')

ROOT = io_tools.get_root(__file__, num_returns=2)

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--accelerator",
        type=str,
        default='gpu',
        help="The type of accelerator.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of computing devices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Logging directory.",
    )
    parser.add_argument(
        "--expname",
        type=str,
        default='Cmamba',
        help="Experiment name. Reconstructions will be saved under this folder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "--logger_type",
        default='tb',
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch_size",
    )

    parser.add_argument(
        "--balance",
        type=float,
        default=100,
        help="initial money",
    )

    parser.add_argument(
        "--risk",
        type=float,
        default=2,
    )

    parser.add_argument(
        "--split",
        type=str,
        default='test',
        choices={'test', 'val', 'train'},
    )

    parser.add_argument(
        "--trade_mode",
        type=str,
        default='smart',
        choices={'smart', 'smart_w_short', 'vanilla', 'no_strategy'},
    )

    args = parser.parse_args()
    return args

def load_model(config, ckpt_path):
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)
    normalize = model_config.get('normalize', False)
    model_class = io_tools.get_obj_from_str(model_config.get('target'))
    model = model_class.load_from_checkpoint(ckpt_path, **model_config.get('params'))
    model.cuda()
    model.eval()
    return model, normalize


def init_dirs(args, name):
    path = f'{ROOT}/Results/{name}/{args.config}'
    if name == 'all':
        path = f'{ROOT}/Results/all/'
    if not os.path.isdir(path):
        os.makedirs(path)

def max_drawdown(prices):
    prices = np.array(prices)
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    mdd = drawdown.min()
    return -mdd


@torch.no_grad()
def run_model(model, dataloader, factors=None):
    target_list = []
    preds_list = []
    timetamps = []
    with torch.no_grad():
        for batch in dataloader:
            ts = batch.get('Timestamp').numpy().reshape(-1)
            target = batch.get(model.y_key).numpy().reshape(-1)
            features = batch.get('features').to(model.device)
            preds = model(features).cpu().numpy().reshape(-1)
            target_list += [float(x) for x in list(target)]
            preds_list += [float(x) for x in list(preds)]
            if factors is not None:
                timetamps += [float(x) for x in list(batch.get('Timestamp_orig').numpy().reshape(-1))]
            else:
                timetamps += [float(x) for x in list(ts)]

    if factors is not None:
        scale = factors.get(model.y_key).get('max') - factors.get(model.y_key).get('min')
        shift = factors.get(model.y_key).get('min')
        target_list = [x * scale + shift for x in target_list]
        preds_list = [x * scale + shift for x in preds_list]

    targets = np.asarray(target_list)
    preds = np.asarray(preds_list)

    return timetamps, targets, preds

if __name__ == '__main__':
    args = get_args()
    init_dir_flag = False
    if args.config == 'all':
        config_list = ['cmamba_nv', 'lstm_nv', 'lstm_bi_nv', 'gru_nv', 'smamba_nv']
        colors = ['darkblue', 'yellowgreen', 'crimson', 'darkviolet', 'orange']
        init_dirs(args, 'all')
    elif args.config == 'all_v':
        config_list = ['cmamba_v', 'lstm_v', 'lstm_bi_v', 'gru_v', 'smamba_v']
        colors = ['darkblue', 'yellowgreen', 'crimson', 'darkviolet', 'orange']
        init_dirs(args, 'all')
    else:
        config_list = [args.config]
        colors = ['darkblue']
        init_dir_flag = True
    
    plt.figure(figsize=(15, 10))
    for conf, c in zip(config_list, colors):
        config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{conf}.yaml')
        if init_dir_flag:
            init_dir_flag = False
            init_dirs(args, config.get('name', args.expname))
        data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")
        
        model, normalize = load_model(config, args.ckpt_path) 

        use_volume = config.get('use_volume', False)
        feature_list = config.get('features', ['Timestamp', 'Open', 'High', 'Low', 'Close'])

        if feature_list == "all":
            feature_list = io_tools.get_train_csv_columns(data_config)
        elif use_volume:
            feature_list.append('Volume')

        test_transform = DataTransform(is_train=False, features=feature_list)
        data_module = CMambaDataModule(data_config,
                                        train_transform=test_transform,
                                        val_transform=test_transform,
                                        test_transform=test_transform,
                                        batch_size=args.batch_size,
                                        distributed_sampler=False,
                                        num_workers=args.num_workers,
                                        normalize=normalize,
                                        )
        
        if args.split == 'test':
            test_loader = data_module.test_dataloader()
        if args.split == 'val':
            test_loader = data_module.val_dataloader()
        if args.split == 'train':
            test_loader = data_module.train_dataloader()

        factors = None
        if normalize:
            factors = data_module.factors
        timstamps, targets, preds = run_model(model, test_loader, factors)

        data = test_loader.dataset.data
        tmp = data.get('Close')
        time_key = 'Timestamp'
        if normalize:
            time_key = 'Timestamp_orig'
            scale = factors.get(model.y_key).get('max') - factors.get(model.y_key).get('min')
            shift = factors.get(model.y_key).get('min')
            data[model.y_key] = data[model.y_key] * scale + shift

        balance, balance_in_time, total_trades = trade(data, time_key, timstamps, targets, preds, 
                                         balance=args.balance, mode=args.trade_mode, 
                                         risk=args.risk, y_key=model.y_key)
        # raise ValueError(max_drawdown(balance_in_time))

        print(f'{conf} -- Final balance: {round(balance, 2)}')
        print(f'{conf} -- Maximum Draw Down : {round(max_drawdown(balance_in_time) * 100, 2)}')
        print(f'{conf} -- Total gain in %: {round((balance - args.balance) / args.balance * 100, 2)}')
        print(f'{conf} -- Price increase in %: {round((targets[-1] - targets[0]) / targets[0] * 100, 2)}')
        print(f'{conf} -- Trading gain diff in %: {round((((balance - args.balance)/args.balance) - ((targets[-1] - targets[0]) )/ targets[0]) * 100, 2)}')
        print(f'{conf} -- Number of trades: {total_trades}')

        label = conf.replace("_nv", "").replace("_v", "")
        if label == 'cmamba':
            label = 'CryptoMamba'
        elif label == 'lstm':
            label = 'LSTM'
        elif label == 'lstm_bi':
            label = 'Bi-LSTM'
        elif label == 'gru':
            label = 'GRU'
        elif label == 'smamba':
            label = 'S-Mamba'
        tmp = [timstamps[0] - 24 * 60 * 60] + timstamps
        tmp = [datetime.fromtimestamp(int(x)) for x in tmp]
        sns.lineplot(x=tmp, 
                     y=balance_in_time, 
                     color=c, 
                     zorder=0, 
                     linewidth=2.5, 
                     label=label)

    name = config.get('name', args.expname)
    if args.trade_mode == 'no_strategy':
        plot_path = f'./balance_{args.split}.jpg'
    else:
        if len(config_list) == 1:
            plot_path = f'{ROOT}/Results/{name}/{args.config}/balance_{args.split}_{args.trade_mode}.jpg'
        else:
            plot_path = f'{ROOT}/Results/all/balance_{args.config}_{args.split}_{args.trade_mode}.jpg'
    plt.xticks(rotation=30)
    plt.axhline(y=100, color='r', linestyle='--')

    if len(config_list) == 1:
        ax = plt.gca()
        ax.get_legend().remove() 
        plt.title(f'Balance in time (final: {round(balance, 2)})')
    else:
        plt.title(f'Net Worth in Time')

    # matplotlib.rcParams.update({'font.size': 100})
    plt.xlim([tmp[0], tmp[-1]])
    plt.ylabel('Balance ($)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')