import os
import matplotlib.pyplot as plt

import datasets
from datasets import concatenate_datasets, interleave_datasets
from datasets.features import ClassLabel
from seqeval.metrics import accuracy_score, classification_report
from core.common.utils import img_trans_torchvision, get_visual_bbox
from utils_config import utils_config
from dab import *


# adapted from: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv3/Fine_tune_LayoutLMv3_on_FUNSD_(HuggingFace_Trainer).ipynb

EMPTY_BOX = [0, 0, 0, 0]
SEP_BOX = [1, 1, 1, 1]

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


class DataCollatorForT5Chart:
    """
    Data collator used for T5 chart text role classification
    """

    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None):
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id

    def __call__(self, input_ids, bbox_list, labels):
        """
        label_encoding = []

        for label in labels:
            label = self.tokenizer.encode(label, add_special_tokens=True)
            label_encoding.append(label)
        """

        return input_ids, labels, bbox_list


class ChartDataset(Dataset):
    NUM_LABELS = len(utils_config["label_list"])

    def __init__(self, dataset, tokenizer, label_list, id2label,
                 max_seq_length, image_size, type_path=None):

        self.cls_bbox = EMPTY_BOX[:]
        self.pad_bbox = EMPTY_BOX[:]
        self.sep_bbox = SEP_BOX[:]

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_img_embeds = 0

        self.label_list = label_list
        self.label_map = id2label
        self.label_all_tokens = False
        self.n_classes = len(label_list)

        self.words = dataset[type_path]["words"] if type_path is not None else dataset["words"]
        self.bboxes = dataset[type_path]["bboxes"] if type_path is not None else dataset["bboxes"]
        self.labels = dataset[type_path]["labels"] if type_path is not None else dataset["labels"]
        self.images = dataset[type_path]["image"] if type_path is not None else dataset["image"]
        self.image_size = image_size

        self.cls_collator = DataCollatorForT5Chart(
            tokenizer=tokenizer,
        )

        # for example in dataset[type_path]:
        #    self.image_sizes.append(example["image"].size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        rets, n_split = tokenize_and_align_labels(self.images[index],
                                                  self.tokenizer,
                                                  self.image_size,
                                                  self.words[index],
                                                  self.bboxes[index],
                                                  self.labels[index],
                                                  self.label_all_tokens)

        for i in range(n_split):
            input_ids, bbox_inputs, image, org_imgsize, image_size, label_ids = rets[i]

            visual_bbox_input = get_visual_bbox(self.image_size)

            bbox_inputs = adjust_bbox(bbox_inputs, org_imgsize, image_size)

            input_ids, labels, bbox_input = self.cls_collator(input_ids, bbox_inputs, label_ids)

            input_ids, bbox_input, labels = self.pad_tokens(input_ids, bbox_input, labels)

            attention_mask = []
            attention_mask += [1 for i in input_ids if i != 0]
            attention_mask += [0 for i in input_ids if i == 0]

            # char_list = [0]
            # char_bbox_list = [[0,0,0,0]]
            # char_ids = torch.tensor(char_list, dtype=torch.long)
            # char_bbox_input = torch.tensor(char_bbox_list, dtype=torch.float)

            bbox_input = torch.tensor(bbox_input, dtype=torch.float)

            # nested_tensors = [torch.tensor(l) for l in labels]
            # padded_tensor = pad_sequence(nested_tensors, batch_first=True)
            labels = torch.tensor(labels, dtype=torch.int64)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            assert len(bbox_input) == len(input_ids)
            assert len(bbox_input.size()) == 2
            # assert len(char_bbox_input.size()) == 2

            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "seg_data": bbox_input,
                "visual_seg_data": visual_bbox_input,
                "image": image,
                # 'char_ids': char_ids,
                # 'char_seg_data': char_bbox_input
            }
            assert input_ids is not None

            return return_dict

    def get_labels(self):
        return list(map(str, list(range(self.NUM_LABELS))))

    def pad_tokens(self, input_ids, bbox, label):
        # [CLS], sentence, [SEP]
        tokenized_tokens = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        start_token, _, end_token = tokenized_tokens[0], tokenized_tokens[1:-1], tokenized_tokens[-1]

        sentence = [self.tokenizer.eos_token_id] + tokenized_tokens
        bbox = [self.cls_bbox] + bbox + [self.sep_bbox]
        label = [-100] + label + [-100]
        expected_seq_length = self.max_seq_length - self.num_img_embeds
        while len(sentence) < expected_seq_length:
            sentence.append(self.tokenizer.pad_token_id)
            bbox.append(self.pad_bbox)
            label.append(-100)

        assert len(sentence) == len(bbox) and len(sentence) == len(label)
        return sentence, bbox, label


def adjust_bbox(bbox, org_imgsize, new_imgsize):
    unnormalized = []
    adjusted = []
    renormalized = []

    width_scale = new_imgsize / org_imgsize[0]
    height_scale = new_imgsize / org_imgsize[1]

    for box in bbox:
        unnormalized.append([
            org_imgsize[0] * (box[0] / 1000),
            org_imgsize[1] * (box[1] / 1000),
            org_imgsize[0] * (box[2] / 1000),
            org_imgsize[1] * (box[3] / 1000),
        ])

    for box in unnormalized:
        adjusted.append([
            box[0] * width_scale,
            box[1] * height_scale,
            box[2] * width_scale,
            box[3] * height_scale,
        ])

    for box in adjusted:
        renormalized.append([
            box[0] / new_imgsize,
            box[1] / new_imgsize,
            box[2] / new_imgsize,
            box[3] / new_imgsize,
        ])

    return renormalized


def get_index(ids, value):
    for k, v in ids.items():
        if value in v:
            return k


def tokenize_and_align_labels(image, tokenizer, image_size, words, bboxes, labels, label_all_tokens):
    rets = []
    n_split = 0

    text_list = []
    text_dict = {}

    org_imgsize = image.size
    image = img_trans_torchvision(image, image_size)

    count = 0
    for i, word in enumerate(words):
        text_dict[i] = []
        sub_tokens = tokenizer.tokenize(word)
        for sub_token in sub_tokens:
            text_list.append(sub_token)
            text_dict[i].append(count)
            count += 1

    label_ids = []
    bbox_inputs = []
    previous_word_idx = None

    start_ids = [v[0] for k, v in text_dict.items()]

    input_ids = tokenizer.convert_tokens_to_ids(text_list)
    for i, word_id in enumerate(input_ids):
        if word_id is None:
            label_ids.append(-100)
            bbox_inputs.append([0, 0, 0, 0])
        elif word_id != previous_word_idx and i in start_ids:
            idx = get_index(text_dict, i)
            label_ids.append(labels[idx])
            bbox_inputs.append(bboxes[idx])
        else:
            idx = get_index(text_dict, i)
            label_ids.append(labels[idx] if label_all_tokens else -100)
            bbox_inputs.append(bboxes[idx])
        previous_word_idx = word_id

    if len(input_ids) > 0:
        rets.append([input_ids, bbox_inputs, image, org_imgsize, image_size, label_ids])

    assert len(input_ids) == len(bbox_inputs)
    n_split = len(rets)

    return rets, n_split


def make_train_eval_test_ds(ds, tokenizer, id2label, combine_type=None, eval=False, transforms_augmentations=None,
                            transforms_balancing=None, experiment_mode=False, concat_datasets=False):
    # split icpr22 train into train and eval
    icpr22 = ds[0]["train"].train_test_split(shuffle=True, test_size=0.1)

    test_dataset = {"icpr22": ChartDataset(dataset=ds[0]["test"],
                                           tokenizer=tokenizer,
                                           label_list=utils_config["label_list"],
                                           id2label=id2label,
                                           max_seq_length=utils_config["max_seq_length"],
                                           image_size=utils_config["image_size"]
                                           )}

    # prepare datasets
    if combine_type is None:

        train_dataset = ChartDataset(dataset=icpr22,
                                     tokenizer=tokenizer,
                                     label_list=utils_config["label_list"],
                                     id2label=id2label,
                                     max_seq_length=utils_config["max_seq_length"],
                                     image_size=utils_config["image_size"],
                                     type_path='train'
                                     )

        eval_dataset = ChartDataset(dataset=icpr22,
                                    tokenizer=tokenizer,
                                    label_list=utils_config["label_list"],
                                    id2label=id2label,
                                    max_seq_length=utils_config["max_seq_length"],
                                    image_size=utils_config["image_size"],
                                    type_path='test'
                                    )

        test_dataset["chimer"] = ChartDataset(dataset=ds[1],
                                              tokenizer=tokenizer,
                                              label_list=utils_config["label_list"],
                                              id2label=id2label,
                                              max_seq_length=utils_config["max_seq_length"],
                                              image_size=utils_config["image_size"],
                                              type_path='full'
                                              )

        test_dataset["degruyter"] = ChartDataset(dataset=ds[2],
                                                 tokenizer=tokenizer,
                                                 label_list=utils_config["label_list"],
                                                 id2label=id2label,
                                                 max_seq_length=utils_config["max_seq_length"],
                                                 image_size=utils_config["image_size"],
                                                 type_path='full'
                                                 )

        test_dataset["econbiz"] = ChartDataset(dataset=ds[3],
                                               tokenizer=tokenizer,
                                               label_list=utils_config["label_list"],
                                               id2label=id2label,
                                               max_seq_length=utils_config["max_seq_length"],
                                               image_size=utils_config["image_size"],
                                               type_path='full'
                                               )

        if transforms_augmentations is None and transforms_balancing is None:
            if eval:
                return train_dataset, icpr22["train"], eval_dataset, test_dataset
            return train_dataset, icpr22["train"], test_dataset

        # elif augmentations
        if not concat_datasets:
            train_dataset_dab = get_augmentated_ds(transforms_augmentations, transforms_balancing, icpr22["train"],
                                                   experiment_mode)
            train_dataset_dab_mapped = ChartDataset(dataset=train_dataset_dab,
                                                    tokenizer=tokenizer,
                                                    label_list=utils_config["label_list"],
                                                    id2label=id2label,
                                                    max_seq_length=utils_config["max_seq_length"],
                                                    image_size=utils_config["image_size"]
                                                    )
            return train_dataset_dab_mapped, icpr22["train"], test_dataset

        train_dataset_dab = get_augmentated_ds(transforms_augmentations, transforms_balancing, icpr22["train"],
                                               experiment_mode, concat_datasets=True)
        train_dataset_dab.shuffle()
        train_dataset_dab_mapped = ChartDataset(dataset=train_dataset_dab,
                                                tokenizer=tokenizer,
                                                label_list=utils_config["label_list"],
                                                id2label=id2label,
                                                max_seq_length=utils_config["max_seq_length"],
                                                image_size=utils_config["image_size"]
                                                )
        return train_dataset_dab_mapped, icpr22["train"], test_dataset

    # combine

    test_dataset["chimer"] = ChartDataset(dataset=ds[1],
                                          tokenizer=tokenizer,
                                          label_list=utils_config["label_list"],
                                          id2label=id2label,
                                          max_seq_length=utils_config["max_seq_length"],
                                          image_size=utils_config["image_size"],
                                          type_path="test"
                                          )

    test_dataset["degruyter"] = ChartDataset(dataset=ds[2],
                                             tokenizer=tokenizer,
                                             label_list=utils_config["label_list"],
                                             id2label=id2label,
                                             max_seq_length=utils_config["max_seq_length"],
                                             image_size=utils_config["image_size"],
                                             type_path="test"
                                             )

    test_dataset["econbiz"] = ChartDataset(dataset=ds[3],
                                           tokenizer=tokenizer,
                                           label_list=utils_config["label_list"],
                                           id2label=id2label,
                                           max_seq_length=utils_config["max_seq_length"],
                                           image_size=utils_config["image_size"],
                                           type_path="test"
                                           )

    train_dataset_list = [icpr22["train"], ds[1]["train"], ds[2]["train"], ds[3]["train"]]

    if combine_type == "concat":
        train_dataset_combined = concatenate_datasets(train_dataset_list)
    elif combine_type == "interleave":
        train_dataset_combined = interleave_datasets(train_dataset_list)

    train_dataset_combined_mapped = ChartDataset(dataset=train_dataset_combined,
                                                 tokenizer=tokenizer,
                                                 label_list=utils_config["label_list"],
                                                 id2label=id2label,
                                                 max_seq_length=utils_config["max_seq_length"],
                                                 image_size=utils_config["image_size"]
                                                 )

    if transforms_augmentations is None and transforms_balancing is None:
        return train_dataset_combined_mapped, train_dataset_combined, test_dataset

    # augmentations
    if not concat_datasets:
        train_dataset_dab = get_augmentated_ds(transforms_augmentations, transforms_balancing, train_dataset_combined,
                                               experiment_mode)
        train_dataset_dab_mapped = ChartDataset(dataset=train_dataset_dab,
                                                tokenizer=tokenizer,
                                                label_list=utils_config["label_list"],
                                                id2label=id2label,
                                                max_seq_length=utils_config["max_seq_length"],
                                                image_size=utils_config["image_size"]
                                                )
        return train_dataset_dab_mapped, train_dataset_combined, test_dataset

    train_dataset_dab = get_augmentated_ds(transforms_augmentations, transforms_balancing, train_dataset_combined,
                                           experiment_mode, concat_datasets=True)
    train_dataset_dab.shuffle()
    train_dataset_dab_mapped = ChartDataset(dataset=train_dataset_dab,
                                            tokenizer=tokenizer,
                                            label_list=utils_config["label_list"],
                                            id2label=id2label,
                                            max_seq_length=utils_config["max_seq_length"],
                                            image_size=utils_config["image_size"]
                                            )
    return train_dataset_dab_mapped, train_dataset_combined, test_dataset


def get_augmentated_ds(transforms_augmentations, transforms_balancing, train_ds, experiment_mode, concat_datasets=False):
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
    if concat_datasets:
        for sample in train_ds:
            ds_list.append(sample)
    train_dataset_dab = datasets.Dataset.from_list(ds_list)
    return train_dataset_dab


def compute_metrics(p):
    overall_score = {}
    label_list = utils_config["label_list"]
    predictions, labels = p
    # print(f'predictions[1].shape: {predictions[1].shape}') # (10, 512, 1024)
    # predictions = np.argmax(predictions[0], axis=-1)
    predictions = predictions[0]

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    #print(true_predictions)
    #print(true_labels)

    report = classification_report(
        y_true=true_labels,
        y_pred=true_predictions,
        suffix=False,
        output_dict=True,
        scheme=None,
        mode=None,
        sample_weight=None,
        zero_division='warn',
    )

    overall_score['micro'] = report.pop('micro avg')
    overall_score['macro'] = report.pop('macro avg')
    overall_score['weighted'] = report.pop('weighted avg')

    return {
        'precision': overall_score['macro']['precision'],
        'recall': overall_score['macro']['recall'],
        'f1_micro': overall_score['micro']['f1-score'],
        'f1_macro': overall_score['macro']['f1-score'],
        'f1_weighted': overall_score['weighted']['f1-score'],
        'accuracy': accuracy_score(y_true=true_labels, y_pred=true_predictions),
    }


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


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


def evaluate(trainer, predictions_file, output_file, dataset, label_list, test=False):
    # TEST
    logits = trainer.predict(dataset)
    # predictions = logits.predictions.argmax(-1).squeeze().tolist()
    predictions = logits.predictions[0]
    if test:
        labels = dataset.labels
    else:
        labels = [d["labels"] for d in dataset]
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
