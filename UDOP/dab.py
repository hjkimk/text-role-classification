import random
from collections import Counter
import cv2
import numpy as np
from PIL import Image
import string

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_label_weights(ds):
    """
    Gets label weights for weighted cross entropy loss
    """
    # count the number of samples per label
    label_counts = Counter(label for labels in ds["labels"] for label in labels)

    # compute inverse class frequency as weights
    total_samples = sum(label_counts.values())
    label_weights = torch.tensor([total_samples / label_counts[label] for label in label_counts],
                                 dtype=torch.float32)
    label_weights = label_weights / label_weights.sum()
    return label_weights

'''
def unnormalize_box(bbox, width, height):
    return [
        int(width * (bbox[0] / 1000)),
        int(height * (bbox[1] / 1000)),
        int(width * (bbox[2] / 1000)),
        int(height * (bbox[3] / 1000)),
    ]
'''

# cutout augmentation
class Cutout(object):
    # modified from: https://github.com/uoguelph-mlrg/Cutout
    """
    Randomly mask (cutout) one or more patches from an image.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        img = sample["image"]
        w, h = img.size
        # bboxes = [unnormalize_box(box, w, h) for box in sample['bboxes']]
        bboxes = sample["polygon"]
        labels = sample["labels"]

        # Step 1: Calculate probability distribution for the labels

        total_labels = len(labels)
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        probabilities = [label_counts[label] / total_labels for label in label_counts.keys()]

        # Step 2: Randomly select a label for cutout
        mask_label = random.choices(list(label_counts.keys()), weights=probabilities, k=1)[0]

        # Step 3: Get bounding box coordinates for mask
        selected_bbox = []
        mask_indices = [i for i, x in enumerate(labels) if x == mask_label]

        # select random number of masks
        n_holes = random.randint(0, len(mask_indices) - 1)  # some images will not be masked
        mask_indices = random.sample(mask_indices, n_holes)
        for index in mask_indices:
            selected_bbox.append(bboxes[index])

        # mask image

        mask = np.ones((h, w), np.float32)

        for bbox in selected_bbox:
            # x1 = bbox[0]
            # y1 = bbox[1]
            # x2 = bbox[2]
            # y2 = bbox[3]

            # mask[y1: y2, x1: x2] = 0.
            points = np.array([[bbox[0], bbox[4]],
                               [bbox[1], bbox[5]],
                               [bbox[2], bbox[6]],
                               [bbox[3], bbox[7]]], np.int32)

            points = points.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [points], 0)

        mask = torch.from_numpy(mask)
        tensor_img = transforms.ToTensor()(img)
        mask = mask.expand_as(tensor_img)
        tensor_img = tensor_img * mask

        img = transforms.ToPILImage()(tensor_img)

        return {"id": sample["id"],
                "words": sample["words"],
                "bboxes": sample["bboxes"],
                "labels": sample["labels"],
                "image": img,
                "polygon": sample["polygon"]}


# other augmentations
def quad_to_box(quad):
    box = (
        max(0, quad[0]),
        max(0, quad[1]),
        quad[4],
        quad[5]
    )
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box


def normalize_bbox(bbox, width, height):
    left = bbox[0]
    top = bbox[1]
    right = bbox[2]
    bottom = bbox[3]
    return [
        int(1000 * left / width),
        int(1000 * top / height),
        int(1000 * right / width),
        int(1000 * bottom / height)
    ]


class AdjustNoise(object):

    """
    Randomly adjusts noise of image using salt and pepper method and guassian noise.
    """

    def __init__(self, num_pixels_min, num_pixels_max):
        self.num_pixels_min = num_pixels_min
        self.num_pixels_max = num_pixels_max

    def __call__(self, sample):

        img = sample["image"]

        # SALT AND PEPPER NOISE #
        img_array = np.array(img)
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]

        # randomly pick pixels in image and color them white
        number_of_pixels = random.randint(self.num_pixels_min, self.num_pixels_max)
        for i in range(number_of_pixels):
            y_coord = random.randint(0, height - 1)
            x_coord = random.randint(0, width - 1)
            img[y_coord][x_coord] = 225

        # randomly pick pixels in image and color them black
        number_of_pixels = random.randint(self.num_pixels_min, self.num_pixels_max)
        for i in range(number_of_pixels):
            y_coord = random.randint(0, height - 1)
            x_coord = random.randint(0, width - 1)
            img[y_coord][x_coord] = 0

        # GAUSSIAN NOISE #
        mean = 0
        std = 0
        noise = np.zeros(img.shape, np.uint8)
        cv2.randn(noise, mean, std)

        img = cv2.add(img, noise)
        img = Image.fromarray(np.uint8(img))
        if img.mode == "L":
            img = img.convert("RGB")

        return {"id": sample["id"],
                "words": sample["words"],
                "bboxes": sample["bboxes"],
                "labels": sample["labels"],
                "image": img,
                "polygon": sample["polygon"]}


class AdjustColor(object):
    """
    Randomly adjusts the color of the image in terms of brightness, contrast, saturation, and hue.
    """

    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        img = sample["image"]

        transform = transforms.ColorJitter(
            brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue
        )

        img = transform(img)

        return {"id": sample["id"],
                "words": sample["words"],
                "bboxes": sample["bboxes"],
                "labels": sample["labels"],
                "image": img,
                "polygon": sample["polygon"]}


class ApplyRotation(object):

    """
    Randomly applies rotation to image
    """

    def __init__(self, angle, fill_color=(225, 225, 225)):
        self.angle = angle
        self.fill_color = fill_color  # white

    def __call__(self, sample):
        # randomly rotate img
        img = sample["image"]
        polygon = sample["polygon"]
        angle = np.random.randint(-self.angle, self.angle + 1)

        width, height = img.size
        img = np.array(img)
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height), borderValue=self.fill_color)

        # adjust bounding boxes

        bboxes = []
        new_polygons = []

        rotated_img = Image.fromarray(rotated_img)

        for poly in polygon:

            cx, cy = (int(width / 2), int(height / 2))

            bbox_tuple = [
                (poly[0], poly[4]),
                (poly[1], poly[5]),
                (poly[2], poly[6]),
                (poly[3], poly[7]),
            ]  # put x and y coordinates in tuples, we will iterate through the tuples and perform rotation

            rotated_bbox = []

            for i, coord in enumerate(bbox_tuple):
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                v = [coord[0], coord[1], 1]
                adjusted_coord = np.matmul(M, v)
                rotated_bbox.insert(i, (adjusted_coord[0], adjusted_coord[1]))

            result = [int(x) for t in rotated_bbox for x in t]

            # make sure resulting bbox coordinates are within the range of the image
            for i, n in enumerate(result):
                if i % 2 == 0 and n > width:
                    result[i] = width
                elif i % 2 == 1 and n > height:
                    result[i] = height
                elif n < 0:
                    result[i] = 0

            new_polygons.append(result)
            bboxes.append(normalize_bbox(quad_to_box(result), width, height))

        return {"id": sample["id"],
                "words": sample["words"],
                "bboxes": bboxes,
                "labels": sample["labels"],
                "image": rotated_img,
                "polygon": new_polygons}


class InsertChars(object):

    """
    Randomly inserts characters into text
    """

    def __init__(self, num_words):
        self.num_words = num_words

    def __call__(self, sample):
        # check if sample has enough words to apply insertion
        word_count = len(sample["words"])
        if word_count < self.num_words:
            return {"id": sample["id"],
                    "words": sample["words"],
                    "bboxes": sample["bboxes"],
                    "labels": sample["labels"],
                    "image": sample["image"],
                    "polygon": sample["polygon"]}

        # if sample has enough words, apply insertion
        words = sample["words"]
        random_chars = []
        random_word_list_pos = set()
        random_word_pos = []
        new_words = []

        for i in range(self.num_words):
            random_chars.append(random.choice(string.ascii_letters))

        for i in range(self.num_words):
            random_word_list_pos.add(random.randint(0, len(words) - 1))

        for i in random_word_list_pos:
            word = words[self.num_words]
            random_word_pos.append(random.randint(0, len(word) - 1))

        count = 0
        for i, w in enumerate(words):
            if i in random_word_list_pos:
                pos = random_word_pos[count]
                new_word = w[:pos] + random_chars[count] + w[pos:]
                new_words.append(new_word)
                count += 1
            else:
                new_words.append(w)

        return {"id": sample["id"],
                "words": new_words,
                "bboxes": sample["bboxes"],
                "labels": sample["labels"],
                "image": sample["image"],
                "polygon": sample["polygon"]}


class SubstituteChars(object):

    """
    Randomly substitutes characters into text
    """

    def __init__(self, num_words, min_word_len):
        self.num_words = num_words
        self.min_word_len = min_word_len

    def __call__(self, sample):
        # check if sample has enough words to apply substitution
        sub_count = sum(1 for word in list(sample["words"]) if len(word) >= self.min_word_len)
        if (sub_count < self.num_words):
            return {"id": sample["id"],
                    "words": sample["words"],
                    "bboxes": sample["bboxes"],
                    "labels": sample["labels"],
                    "image": sample["image"],
                    "polygon": sample["polygon"]}

        # if sample has enough words, apply substitution
        words = sample["words"]
        random_chars = []
        random_word_list_pos = set()
        random_word_pos = []
        new_words = []

        # get word positions that have at least minimum word length
        word_pos = [i for i, n in enumerate(words) if len(n) >= self.min_word_len]

        # get n random charaters
        for i in range(self.num_words):
            random_chars.append(random.choice(string.ascii_letters))

        # get n random list positions for words in the list
        while len(random_word_list_pos) < self.num_words:
            pos = random.choice(word_pos)
            word_pos.remove(pos)
            random_word_list_pos.add(pos)

        # get n random word positions for each word in the list
        for i in random_word_list_pos:
            word = words[i]
            random_word_pos.append(random.randint(0, len(word) - 1))

        # substitute characters
        count = 0
        for i, w in enumerate(words):
            if i in random_word_list_pos:
                pos = random_word_pos[count]
                if pos == len(w):
                    new_word = w[:pos] + list(random_chars)[count]
                else:
                    new_word = w[:pos] + list(random_chars)[count] + w[pos + 1:]
                new_words.append(new_word)
                count += 1
            else:
                new_words.append(w)

        return {"id": sample["id"],
                "words": new_words,
                "bboxes": sample["bboxes"],
                "labels": sample["labels"],
                "image": sample["image"],
                "polygon": sample["polygon"]}


class DeleteChars(object):

    """
    Randomly deletes characters from text
    """

    def __init__(self, max_del_len, num_words, min_word_len):
        self.max_del_len = max_del_len
        self.num_words = num_words
        self.min_word_len = min_word_len

    def __call__(self, sample):
        # check if sample has enough words for deletion
        del_count = sum(1 for word in list(sample["words"]) if len(word) >= self.min_word_len)
        if (del_count < self.num_words):
            return {"id": sample["id"],
                    "words": sample["words"],
                    "bboxes": sample["bboxes"],
                    "labels": sample["labels"],
                    "image": sample["image"],
                    "polygon": sample["polygon"]}

        # if sample has enough words, apply deletion
        words = sample["words"]
        random_del_len = []
        random_word_list_pos = set()
        new_words = []

        # get word positions that have at least minimum word length
        word_pos = [i for i, n in enumerate(words) if len(n) >= self.min_word_len]

        # get n random deletion lengths
        for i in range(self.num_words):
            random_del_len.append(random.randint(1, self.max_del_len))

        # get n random list positions for words in the list
        while len(random_word_list_pos) < self.num_words:
            pos = random.choice(word_pos)
            word_pos.remove(pos)
            random_word_list_pos.add(pos)

        # substitute characters
        count = 0
        for i, w in enumerate(words):
            if i in random_word_list_pos:
                del_len = random_del_len[count]
                new_word = w[del_len:]
                new_words.append(new_word)
                count += 1
            else:
                new_words.append(w)

        return {"id": sample["id"],
                "words": new_words,
                "bboxes": sample["bboxes"],
                "labels": sample["labels"],
                "image": sample["image"],
                "polygon": sample["polygon"]}


class Identity(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        return {"id": sample["id"],
                "words": sample["words"],
                "bboxes": sample["bboxes"],
                "labels": sample["labels"],
                "image": sample["image"],
                "polygon": sample["polygon"]}


def get_augmentations(augmentation_dict):
    """
    Creates a list of augmentations to apply
    """
    transforms_augmentation = []
    for k, v in augmentation_dict.items():
        if k == "adjust_noise":
            adjust_noise = AdjustNoise(num_pixels_min=augmentation_dict["adjust_noise"]["num_pixels_min"],
                                       num_pixels_max=augmentation_dict["adjust_noise"]["num_pixels_max"])
            transforms_augmentation.append(adjust_noise)
        elif k == "adjust_color":
            adjust_color = AdjustColor(brightness=augmentation_dict["adjust_color"]["brightness"],
                                       contrast=augmentation_dict["adjust_color"]["contrast"],
                                       saturation=augmentation_dict["adjust_color"]["saturation"],
                                       hue=augmentation_dict["adjust_color"]["hue"])
            transforms_augmentation.append(adjust_color)
        elif k == "apply_rotation":
            apply_rotation = ApplyRotation(angle=augmentation_dict["apply_rotation"]["angle"])
            transforms_augmentation.append(apply_rotation)
        elif k == "insert_chars":
            insert_chars = InsertChars(num_words=augmentation_dict["insert_chars"]["num_words"])
            transforms_augmentation.append(insert_chars)
        elif k == "substitute_chars":
            substitute_chars = SubstituteChars(num_words=augmentation_dict["substitute_chars"]["num_words"],
                                               min_word_len=augmentation_dict["substitute_chars"]["min_word_len"])
            transforms_augmentation.append(substitute_chars)
        elif k == "delete_chars":
            delete_chars = DeleteChars(max_del_len=augmentation_dict["delete_chars"]["max_del_len"],
                                       num_words=augmentation_dict["delete_chars"]["num_words"],
                                       min_word_len=augmentation_dict["delete_chars"]["min_word_len"])
            transforms_augmentation.append(delete_chars)
        #elif k == "identity":
        #    identity = Identity()
        #    transforms_augmentation.append(identity)

    return transforms_augmentation


class DAB(Dataset):
    """
    creates augmentated dataset
    """
    def __init__(self, dataset, transforms_balancing=None, transforms_augmentation=None, experiment_mode=False, untouched_percent=30):
        self.dataset = dataset
        self.transforms_balancing = transforms_balancing
        self.transforms_augmentation = transforms_augmentation
        self.experiment_mode = experiment_mode
        self.untouched_percent = untouched_percent
        self.untouched_indices = self._get_untouched_indices()

    def _get_untouched_indices(self):
        num_samples = len(self.dataset)
        num_untouched_samples = int(num_samples * (self.untouched_percent / 100))
        all_indices = list(range(num_samples))
        untouched_indices = random.sample(all_indices, num_untouched_samples)
        return untouched_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        if index in self.untouched_indices:
            # Return untouched sample
            return sample

        if self.transforms_balancing:
            for transform in self.transforms_balancing:
                sample = transform(sample)

        if self.transforms_augmentation:
            if self.experiment_mode:
                for transform in self.transforms_augmentation:
                    sample = transform(sample)
                return sample
            else:
                selected_transform = random.choice(self.transforms_augmentation)
                sample = selected_transform(sample)

        return sample
