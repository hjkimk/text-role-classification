import os
import sys
import json
import numpy as np
import tempfile

import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoProcessor, TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
from datasets import load_dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D, Dataset, load_from_disk

from utils import *
from dab import *


def main():
    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    use_cuda = not config["no_cuda"] and torch.cuda.is_available()
    if not config["no_cuda"] and not use_cuda:
        raise ValueError('You wanted to use cuda but it is not available. '
                         'Check nvidia-smi and your configuration. If you do '
                         'not want to use cuda, pass the --no-cuda flag.')
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {torch.cuda.get_device_name()}")

    # trainer uses seed number 42 as default
    """
    if config["seed"] is None:
        seed = torch.randint(0, 2 ** 32, (1,)).item()
        print(f"You did not set --seed, {seed} was chosen")
    else:
        seed = config["seed"]
    torch.manual_seed(seed)
    """

    if use_cuda:
        if device.index:
            device_str = f"{device.type}:{device.index}"
        else:
            device_str = f"{device.type}"
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        # torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not os.path.exists(config["output_dir"]):
        print(f'{config["output_dir"]} does not exist, creating...')
        os.makedirs(config["output_dir"])

    # LOAD DATASET #

    dataset_config_icpr22 = {
        "LOADING_SCRIPT_FILES": config["loading_script"],
        "CONFIG_NAME": config["datasets"][2]
    }

    icpr22_cache_dir = tempfile.mkdtemp(prefix="my_cache_icpr22")
    icpr22 = load_dataset(
        dataset_config_icpr22["LOADING_SCRIPT_FILES"],
        dataset_config_icpr22["CONFIG_NAME"],
        cache_dir=icpr22_cache_dir
    )

    if config['combine_type'] is None:

        dataset_config_chimer = {
            "LOADING_SCRIPT_FILES": config["loading_script"],
            "CONFIG_NAME": config["datasets"][3]
        }

        dataset_config_degruyter = {
            "LOADING_SCRIPT_FILES": config["loading_script"],
            "CONFIG_NAME": config["datasets"][4]
        }

        dataset_config_econbiz = {
            "LOADING_SCRIPT_FILES": config["loading_script"],
            "CONFIG_NAME": config["datasets"][5]
        }

        chimer_cache_dir = tempfile.mkdtemp(prefix="my_cache_chimer")
        chimer = load_dataset(
            dataset_config_chimer["LOADING_SCRIPT_FILES"],
            dataset_config_chimer["CONFIG_NAME"],
            cache_dir=chimer_cache_dir
        )

        degruyter_cache_dir = tempfile.mkdtemp(prefix="my_cache_degruyter")
        degruyter = load_dataset(
            dataset_config_degruyter["LOADING_SCRIPT_FILES"],
            dataset_config_degruyter["CONFIG_NAME"],
            cache_dir=degruyter_cache_dir
        )

        econbiz_cache_dir = tempfile.mkdtemp(prefix="my_cache_econbiz")
        econbiz = load_dataset(
            dataset_config_econbiz["LOADING_SCRIPT_FILES"],
            dataset_config_econbiz["CONFIG_NAME"],
            cache_dir=econbiz_cache_dir
        )

    else:

        chimer_path = utils_config["data_dir"]["CHIME-R"].rsplit('/', 1)[0]
        chimer = {'train': load_from_disk(f'{chimer_path}/train'),
                  'test': load_from_disk(f'{chimer_path}/test')}

        degruyter_path = utils_config["data_dir"]["DeGruyter"].rsplit('/', 1)[0]
        degruyter = {'train': load_from_disk(f'{degruyter_path}/train'),
                     'test': load_from_disk(f'{degruyter_path}/test')}

        econbiz_path = utils_config["data_dir"]["EconBiz"].rsplit('/', 1)[0]
        econbiz = {'train': load_from_disk(f'{econbiz_path}/train'),
                   'test': load_from_disk(f'{econbiz_path}/test')}

    ds = [icpr22, chimer, degruyter, econbiz]

    # PREPARE DATASET #
    column_names = icpr22["train"].column_names
    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(feature=Value(dtype='int64')),
    })

    label_list, id2label, label2id = get_labels(icpr22["train"])

    if config["combine_type"] is None:
        if config["eval_mode"]:
            train_dataset, train_dataset_orig, eval_dataset, test_dataset = make_train_eval_test_ds(ds,
                                                                                                    column_names,
                                                                                                    features,
                                                                                                    eval=True)

        elif config["transforms_augmentations"] is None and config["transforms_balancing"] is None:
            train_dataset, train_dataset_orig, test_dataset = make_train_eval_test_ds(ds,
                                                                                      column_names,
                                                                                      features)

        else:
            train_dataset, train_dataset_orig, test_dataset = make_train_eval_test_ds(ds,
                                                                                      column_names,
                                                                                      features,
                                                                                      transforms_augmentations=config["transforms_augmentations"],
                                                                                      transforms_balancing=config["transforms_balancing"],
                                                                                      experiment_mode=config["experiment_mode"],
                                                                                      concat_datasets=config[ "concat_datasets"])
    else:
        if config["transforms_augmentations"] is None and config["transforms_balancing"] is None:
            train_dataset, train_dataset_orig, test_dataset = make_train_eval_test_ds(ds,
                                                                                      column_names,
                                                                                      features,
                                                                                      combine_type=config["combine_type"])
        else:
            train_dataset, train_dataset_orig, test_dataset = make_train_eval_test_ds(ds,
                                                                                      column_names,
                                                                                      features,
                                                                                      combine_type=config["combine_type"],
                                                                                      transforms_augmentations=config["transforms_augmentations"],
                                                                                      transforms_balancing=config["transforms_balancing"],
                                                                                      concat_datasets=config["concat_datasets"])

    # PREPARE TRAINER #

    if config["debug_mode"]:
        max_steps = 1000
    else:
        max_steps = config["max_steps"]

    for learning_rate in config["learning_rate"]:

        # PREPARE MODEL #
        model = CustomLayoutLMv3ForTokenClassification.from_pretrained(config["pretrained_model"],
                                                                 id2label=id2label,
                                                                 label2id=label2id,
                                                                 classifier=config["classifier"])
        processor = AutoProcessor.from_pretrained(config["pretrained_model"], apply_ocr=False)
        special_tokens = {"additional_special_tokens": config["special_tokens"]}
        processor.tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(processor.tokenizer))

        # APPLY DATA BALANCING #
        if config["transforms_balancing"] is not None and "weighted_loss" in config["transforms_balancing"]:
            # weighted cross entropy loss
            label_weights = get_label_weights(train_dataset_orig)
            model.weights = label_weights

        config_args = [str(vv) for kk, vv in config.items()
                       if kk in ["model_name",
                                 "max_steps",
                                 "batch_size",
                                 "gradient_accumulation_steps",
                                 "combine_type",
                                 "concat_datasets",
                                 "classifier"]]

        if not config["debug_mode"]:
            model_name = '_'.join(config_args)
        else:
            model_name = 'debug_'
            model_name += '_'.join(config_args)

        model_name += f'_{learning_rate}'

        if isinstance(config["transforms_augmentations"], dict):
            for k, v in config["transforms_augmentations"].items():
                model_name += f'_{k}'
        else:
            model_name += f'_noaug'

        if isinstance(config["transforms_balancing"], list):
            for i in config["transforms_balancing"]:
                model_name += f'_{i}'
        else:
            model_name += f'_nobal'

        train_dataset.set_format("torch")
        if config["eval_mode"] is True:

            training_args = TrainingArguments(output_dir=f'{config["output_dir"]}/{model_name}',
                                              max_steps=max_steps,
                                              per_device_train_batch_size=config["batch_size"],
                                              per_device_eval_batch_size=config["batch_size"],
                                              learning_rate=learning_rate,
                                              evaluation_strategy=config["eval_strategy"],
                                              eval_steps=config["eval_steps"],
                                              logging_dir=f'{config["logging_dir"]}/{model_name}',
                                              logging_strategy=config["logging_strategy"],
                                              logging_steps=config["logging_steps"],
                                              gradient_accumulation_steps=config["gradient_accumulation_steps"],
                                              )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=processor,
                data_collator=default_data_collator,
                compute_metrics=compute_metrics,
            )

        else:

            training_args = TrainingArguments(output_dir=f'{config["output_dir"]}/{model_name}',
                                              max_steps=max_steps,
                                              per_device_train_batch_size=config["batch_size"],
                                              learning_rate=learning_rate,
                                              logging_dir=f'{config["logging_dir"]}/{model_name}',
                                              logging_strategy=config["logging_strategy"],
                                              logging_steps=config["logging_steps"],
                                              gradient_accumulation_steps=config["gradient_accumulation_steps"],
                                              )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=processor,
                data_collator=default_data_collator,
                compute_metrics=compute_metrics,
            )

        # TRAIN #
        old_collator = trainer.data_collator
        trainer.data_collator = lambda data: dict(old_collator(data))
        trainer.train()

        # LOG #
        log(trainer, model_name, config)

        # EVALUATE/TEST #
        if config["eval_mode"] is True:
            evaluate(trainer,
                     predictions_file=f'{config["output_dir"]}/{model_name}/predictions_eval.txt',
                     output_file=f'{config["output_dir"]}/{model_name}/output_eval.txt',
                     dataset=eval_dataset,
                     label_list=label_list)
        for k, v in test_dataset.items():
            evaluate(trainer,
                     predictions_file=f'{config["output_dir"]}/{model_name}/predictions_{k}.txt',
                     output_file=f'{config["output_dir"]}/{model_name}/output_{k}.txt',
                     dataset=v,
                     label_list=label_list)


if __name__ == "__main__":
    main()
