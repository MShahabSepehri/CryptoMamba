import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import yaml
from utils import io_tools
import pytorch_lightning as pl
from argparse import ArgumentParser
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger


ROOT = io_tools.get_root(__file__, num_returns=2)

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        help="Logging directory.",
    )
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
        default='cmamba_nv',
        help="Path to config file.",
    )
    parser.add_argument(
        "--logger_type",
        default='tb',
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
        '--save_checkpoints', 
        default=False,   
        action='store_true',          
    )
    parser.add_argument(
        '--use_volume', 
        default=False,   
        action='store_true',          
    )

    parser.add_argument(
        '--resume_from_checkpoint',
        default=None,
    )

    parser.add_argument(
        '--max_epochs',
        type=int,
        default=200,
    )

    args = parser.parse_args()
    return args


def save_all_hparams(log_dir, args):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_dict = vars(args)
    path = log_dir + '/hparams.yaml'
    if os.path.exists(path):
        return
    with open(path, 'w') as f:
        yaml.dump(save_dict, f)


def load_model(config, logger_type):
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)

    normalize = model_config.get('normalize', False)
    hyperparams = config.get('hyperparams')
    if hyperparams is not None:
        for key in hyperparams.keys():
            model_config.get('params')[key] = hyperparams.get(key)

    model_config.get('params')['logger_type'] = logger_type
    model = io_tools.instantiate_from_config(model_config)
    model.cuda()
    model.train()
    return model, normalize


if __name__ == "__main__":

    args = get_args()
    pl.seed_everything(args.seed)
    logdir = args.logdir

    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')

    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")
    use_volume = args.use_volume
    if not use_volume:
        use_volume = config.get('use_volume')

    feature_list = config.get('features', ['Timestamp', 'Open', 'High', 'Low', 'Close'])
    if feature_list == "all":
        # get header from data file
        data = io_tools.load_data(data_config.get('data_path'))
        feature_list = list(data.columns)
    elif use_volume:
        feature_list.append('Volume')
    
    train_transform = DataTransform(is_train=True, features=feature_list)
    val_transform = DataTransform(is_train=False, features=feature_list)
    test_transform = DataTransform(is_train=False, features=feature_list)

    model, normalize = load_model(config, args.logger_type)

    tmp = vars(args)
    tmp.update(config)

    name = config.get('name', args.expname)
    if args.logger_type == 'tb':
        logger = TensorBoardLogger("logs", name=name)
        logger.log_hyperparams(args)
    elif args.logger_type == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.expname, config=tmp)
    else:
        raise ValueError('Unknown logger type.')

    data_module = CMambaDataModule(data_config,
                                   train_transform=train_transform,
                                   val_transform=val_transform,
                                   test_transform=test_transform,
                                   batch_size=args.batch_size,
                                   distributed_sampler=True,
                                   num_workers=args.num_workers,
                                   normalize=normalize,
                                   window_size=model.window_size,
                                   )
    
    callbacks = []
    if args.save_checkpoints:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val/rmse",
            mode="min",
            filename='epoch{epoch}-val-rmse{val/rmse:.4f}',
            auto_insert_metric_name=False,
            save_last=True
        )
        callbacks.append(checkpoint_callback)


    max_epochs = config.get('max_epochs', args.max_epochs)
    trainer = pl.Trainer(accelerator=args.accelerator, 
                         devices=args.devices,
                         max_epochs=max_epochs,
                         enable_checkpointing=args.save_checkpoints,
                         logger=logger,
                         callbacks=callbacks,
                         strategy = DDPStrategy(find_unused_parameters=True),
                         )

    trainer.fit(model, datamodule=data_module)
    if args.save_checkpoints:
        trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
