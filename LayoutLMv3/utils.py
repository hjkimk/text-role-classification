import os
import matplotlib.pyplot as plt

from transformers import AutoProcessor
import datasets
from datasets import concatenate_datasets, interleave_datasets
from datasets.features import ClassLabel
from seqeval.metrics import accuracy_score, classification_report

from config import utils_config
from dab import *


# adapted from: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv3/Fine_tune_LayoutLMv3_on_FUNSD_(HuggingFace_Trainer).ipynb

def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def get_labels(ds):
    features = ds.features
    if isinstance(features[utils_config["column_names"]["label"]].feature, ClassLabel):
        label_list = features[utils_config["column_names"]["label"]].feature.names
        # No need to convert the labels since they are already ints.
        id2label = {k: v for k, v in enumerate(label_list)}
        label2id = {v: k for k, v in enumerate(label_list)}
    else:
        label_list = get_label_list(ds[utils_config["column_names"]["label"]])
        id2label = {k: v for k, v in enumerate(label_list)}
        label2id = {v: k for k, v in enumerate(label_list)}
    return label_list, id2label, label2id


def prepare_examples(examples):
    images = examples[utils_config["column_names"]["image"]]
    words = examples[utils_config["column_names"]["text"]]
    boxes = examples[utils_config["column_names"]["boxes"]]
    word_labels = examples[utils_config["column_names"]["label"]]
    processor = AutoProcessor.from_pretrained(utils_config["pretrained_model"], apply_ocr=False)

    encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
                         truncation=True, padding="max_length")

    return encoding


def make_train_eval_test_ds(ds, cols, features, batched=True, combine_type=None, eval=False,
                            transforms_augmentations=None, transforms_balancing=None, experiment_mode=False, concat_datasets=False):

    # split icpr22 train into train and eval
    icpr22 = ds[0]["train"].train_test_split(shuffle=True, test_size=0.1)

    test_dataset = {"icpr22": ds[0]["test"].map(
        prepare_examples,
        batched=batched,
        remove_columns=cols,
        features=features,
    )}

    # prepare datasets
    if combine_type is None:

        train_dataset = {"icpr22": icpr22["train"].map(
            prepare_examples,
            batched=batched,
            remove_columns=cols,
            features=features,
        )}

        eval_dataset = {"icpr22": icpr22["test"].map(
            prepare_examples,
            batched=batched,
            remove_columns=cols,
            features=features,
        )}

        test_dataset["chimer"] = ds[1]["full"].map(
            prepare_examples,
            batched=batched,
            remove_columns=cols,
            features=features,
        )

        test_dataset["degruyter"] = ds[2]["full"].map(
            prepare_examples,
            batched=batched,
            remove_columns=cols,
            features=features,
        )

        test_dataset["econbiz"] = ds[3]["full"].map(
            prepare_examples,
            batched=batched,
            remove_columns=cols,
            features=features,
        )

        if transforms_augmentations is None and transforms_balancing is None and eval:
            return train_dataset["icpr22"], icpr22["train"], eval_dataset["icpr22"], test_dataset

        elif transforms_augmentations is None and transforms_balancing is None and not eval:
            return train_dataset["icpr22"], icpr22["train"], test_dataset

        # elif augmentations
        train_dataset_dab = get_augmentated_ds(transforms_augmentations, transforms_balancing, icpr22["train"], experiment_mode)
        train_dataset_dab = train_dataset_dab.map(
            prepare_examples,
            batched=batched,
            remove_columns=cols,
            features=features,
        )
        if concat_datasets:
            train_dataset_dab = concatenate_datasets([train_dataset["icpr22"], train_dataset_dab])
            train_dataset_dab.shuffle()
        return train_dataset_dab, icpr22["train"], test_dataset

    # combine

    test_dataset["chimer"] = ds[1]["test"].map(
        prepare_examples,
        batched=batched,
        remove_columns=cols,
        features=features,
    )

    test_dataset["degruyter"] = ds[2]["test"].map(
        prepare_examples,
        batched=batched,
        remove_columns=cols,
        features=features,
    )

    test_dataset["econbiz"] = ds[3]["test"].map(
        prepare_examples,
        batched=batched,
        remove_columns=cols,
        features=features,
    )

    train_dataset_list = [icpr22["train"], ds[1]["train"], ds[2]["train"], ds[3]["train"]]

    if combine_type == "concat":
        train_dataset_combined = concatenate_datasets(train_dataset_list)

    elif combine_type == "interleave":
        train_dataset_combined = interleave_datasets(train_dataset_list)

    train_dataset_combined_mapped = train_dataset_combined.map(
        prepare_examples,
        batched=batched,
        remove_columns=cols,
        features=features,
    )

    if transforms_augmentations is None and transforms_balancing is None:
        return train_dataset_combined_mapped, train_dataset_combined, test_dataset

    # augmentations
    train_dataset_dab = get_augmentated_ds(transforms_augmentations, transforms_balancing, train_dataset_combined, experiment_mode)
    train_dataset_dab = train_dataset_dab.map(
        prepare_examples,
        batched=batched,
        remove_columns=cols,
        features=features,
    )
    if concat_datasets:
        train_dataset_dab = concatenate_datasets([train_dataset_combined_mapped, train_dataset_dab])
        train_dataset_dab.shuffle()
    return train_dataset_dab, train_dataset_combined, test_dataset


def get_augmentated_ds(transforms_augmentations, transforms_balancing, train_ds, experiment_mode):
    ds_list = []
    apply_cutout = Cutout()
    # other augmentations
    if isinstance(transforms_augmentations, dict):
        transforms_augmentation = get_augmentations(transforms_augmentations)
    else:
        transforms_augmentation = None

    if transforms_balancing is not None and "cutout" in transforms_balancing:
        transforms_balancing = [apply_cutout]
    else:
        transforms_balancing = None

    train_dataset_dab = DAB(dataset=train_ds,
                            transforms_balancing=transforms_balancing,
                            transforms_augmentation=transforms_augmentation,
                            experiment_mode=experiment_mode)
    for sample in train_dataset_dab:
        ds_list.append(sample)
    train_dataset_dab = datasets.Dataset.from_list(ds_list)
    return train_dataset_dab


def compute_metrics(p):
    overall_score = {}
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = utils_config["label_list"]

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(
        y_true=true_labels,
        y_pred=true_predictions,
        suffix=False,
        output_dict=True,
        scheme=None,
        mode=None,
        sample_weight=None,
        zero_division="warn",
    )

    overall_score["micro"] = report.pop("micro avg")
    overall_score["macro"] = report.pop("macro avg")
    overall_score["weighted"] = report.pop("weighted avg")

    if utils_config["return_entity_level_metrics"]:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in report.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": overall_score["macro"]["precision"],
            "recall": overall_score["macro"]["recall"],
            "f1_micro": overall_score["micro"]["f1-score"],
            "f1_macro": overall_score["macro"]["f1-score"],
            # "f1_weighted": overall_score["weighted"]["f1-score"],
            "accuracy": accuracy_score(y_true=true_labels, y_pred=true_predictions),
        }


def log(trainer, model_name, config, plot=False):
    if not os.path.exists(f'{config["logging_dir"]}/{model_name}'):
        print(f'{model_name} directory does not exist in {config["logging_dir"]}, creating...')
        os.makedirs(f'{config["logging_dir"]}/{model_name}')
    eval_loss = []
    eval_accuracy = []
    eval_f1_micro = []
    eval_f1_macro = []
    steps = []
    logging_file = f'{config["logging_dir"]}/{model_name}/logs.txt'

    with open(logging_file, 'w') as f:
        f.write(f'The best model checkpoint is: {trainer.state.best_model_checkpoint} \n')
        for l in trainer.state.log_history[1:-1:2]:
            for k, v in l.items():
                if k == 'eval_loss':
                    eval_loss.append(v)
                elif k == 'eval_accuracy':
                    eval_accuracy.append(v)
                elif k == 'eval_f1_micro':
                    eval_f1_micro.append(v)
                elif k == 'eval_f1_macro':
                    eval_f1_macro.append(v)
                elif k == 'step':
                    steps.append(v)
                f.write(f'{k}: {v} \n')

    if plot:
        plotting_file = f'{config["logging_dir"]}/{model_name}/plot.png'
        plt.plot(steps, eval_accuracy, label='accuracy')
        plt.plot(steps, eval_loss, label='loss')
        plt.plot(steps, eval_f1_micro, label='f1_micro')
        plt.plot(steps, eval_f1_macro, label='f1_macro')
        plt.xlabel('number of steps')
        plt.ylabel('score')
        plt.title('model performance on evaluation dataset')
        plt.legend()
        plt.savefig(f'{plotting_file}')
        plt.clf()


def evaluate(trainer, predictions_file, output_file, dataset, label_list):
    # TEST
    logits = trainer.predict(dataset)
    predictions = logits.predictions.argmax(-1).squeeze().tolist()
    labels = dataset['labels']
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    with open(predictions_file, 'w') as f1:
        for i, pred in enumerate(true_predictions):
            f1.write(f'{i}: {pred} \n')
    with open(output_file, 'w') as f2:
        for k, v in logits.metrics.items():
            f2.write(f'{k}: {v} \n')
