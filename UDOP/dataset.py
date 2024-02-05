import json
import os

from PIL import Image
import cv2
import numpy as np

import datasets

from utils_config import utils_config

_CITATION = {
    "ICPR2022Real": """@inproceedings{DBLP:conf/icpr/DavilaXAMSG22,
  author    = {Kenny Davila and
               Fei Xu and
               Saleem Ahmed and
               David A. Mendoza and
               Srirangaraj Setlur and
               Venu Govindaraju},
  title     = {{ICPR} 2022: Challenge on Harvesting Raw Tables from Infographics
               (CHART-Infographics)},
  booktitle = {26th International Conference on Pattern Recognition, {ICPR} 2022,
               Montreal, QC, Canada, August 21-25, 2022},
  pages     = {4995--5001},
  publisher = {{IEEE}},
  year      = {2022},
  url       = {https://doi.org/10.1109/ICPR56361.2022.9956289},
  doi       = {10.1109/ICPR56361.2022.9956289},
  timestamp = {Thu, 01 Dec 2022 15:50:19 +0100},
  biburl    = {https://dblp.org/rec/conf/icpr/DavilaXAMSG22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}""",
    "OTHER": """@article{DBLP:journals/mta/BoschenBS18,
  author       = {Falk B{\"{o}}schen and
                  Tilman Beck and
                  Ansgar Scherp},
  title        = {Survey and empirical comparison of different approaches for text extraction
                  from scholarly figures},
  journal      = {Multim. Tools Appl.},
  volume       = {77},
  number       = {22},
  pages        = {29475--29505},
  year         = {2018},
  url          = {https://doi.org/10.1007/s11042-018-6162-7},
  doi          = {10.1007/S11042-018-6162-7},
  timestamp    = {Mon, 11 May 2020 15:49:46 +0200},
  biburl       = {https://dblp.org/rec/journals/mta/BoschenBS18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}"""
}


# modified from https://huggingface.co/datasets/nielsr/cord-layoutlmv3/blob/main/cord-layoutlmv3.py

def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    return img, (w, h)


def normalize_bbox(bbox, size, type=None):
    if type == "box":
        height = int(bbox["height"])
        width = int(bbox["width"])
        left = max(0, bbox["x0"])
        top = max(0, bbox["y0"])
        right = left + width
        bottom = top + height
    if type == "polygon":
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]
    return [
        int(1000 * left / size[0]),
        int(1000 * top / size[1]),
        int(1000 * right / size[0]),
        int(1000 * bottom / size[1])
    ]


def quad_to_box(quad):
    box = (
        max(0, quad["x0"]),
        max(0, quad["y0"]),
        quad["x2"],
        quad["y2"]
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


def get_quad(bbox, width, height):
    x0 = int(bbox["center_x"] - bbox["width"] / 2)
    x1 = int(bbox["center_x"] + bbox["width"] / 2)
    x2 = int(bbox["center_x"] + bbox["width"] / 2)
    x3 = int(bbox["center_x"] - bbox["width"] / 2)
    y0 = int(bbox["center_y"] - bbox["height"] / 2)
    y1 = int(bbox["center_y"] - bbox["height"] / 2)
    y2 = int(bbox["center_y"] + bbox["height"] / 2)
    y3 = int(bbox["center_y"] + bbox["height"] / 2)

    if bbox["orientation"] == 0:
        return {"x0": x0,
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "y0": y0,
                "y1": y1,
                "y2": y2,
                "y3": y3}

    # rotate coordinates if orientation is not 0

    cx, cy = (int(width / 2), int(height / 2))

    bbox_tuple = [
        (x0, y0),
        (x1, y1),
        (x2, y2),
        (x3, y3),
    ]

    rotated_bbox = []

    for i, coord in enumerate(bbox_tuple):
        M = cv2.getRotationMatrix2D((cx, cy), bbox["orientation"], 1.0)
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

    return {"x0": result[0],
            "x1": result[2],
            "x2": result[4],
            "x3": result[6],
            "y0": result[1],
            "y1": result[3],
            "y2": result[5],
            "y3": result[7]}


class ChartInfoConfig(datasets.BuilderConfig):
    """ Builder Config for Chart Infographics Dataset """

    def __init__(self, data_dir, labels, img_type, citation, **kwargs):
        super(ChartInfoConfig, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.labels = labels
        self.img_type = img_type
        self.citation = citation


class ChartInfo(datasets.GeneratorBasedBuilder):
    """ Chart Infographics Dataset """

    BUILDER_CONFIGS = [
        ChartInfoConfig(name="ICPR2022Real", description="ICPR Dataset 2022 [Real]",
                        data_dir=utils_config["data_dir"]["ICPR2022Real"],
                        labels=["CHART_TITLE", "LEGEND_TITLE", "LEGEND_LABEL", "AXIS_TITLE", "TICK_LABEL",
                                "TICK_GROUPING", "MARK_LABEL", "VALUE_LABEL", "OTHER"],
                        img_type="jpg",
                        citation=_CITATION["ICPR2022Real"]),
        ChartInfoConfig(name="CHIME-R", description="CHIME-R Dataset",
                        data_dir=utils_config["data_dir"]["CHIME-R"],
                        labels=["CHART_TITLE", "LEGEND_TITLE", "LEGEND_LABEL", "AXIS_TITLE", "TICK_LABEL",
                                "TICK_GROUPING", "MARK_LABEL", "VALUE_LABEL", "OTHER"],
                        img_type="bmp",
                        citation=_CITATION["OTHER"]),
        ChartInfoConfig(name="DeGruyter", description="DeGruyter Dataset",
                        data_dir=utils_config["data_dir"]["DeGruyter"],
                        labels=["CHART_TITLE", "LEGEND_TITLE", "LEGEND_LABEL", "AXIS_TITLE", "TICK_LABEL",
                                "TICK_GROUPING", "MARK_LABEL", "VALUE_LABEL", "OTHER"],
                        img_type="png",
                        citation=_CITATION["OTHER"]),
        ChartInfoConfig(name="EconBiz", description="EconBiz Dataset",
                        data_dir=utils_config["data_dir"]["EconBiz"],
                        labels=["CHART_TITLE", "LEGEND_TITLE", "LEGEND_LABEL", "AXIS_TITLE", "TICK_LABEL",
                                "TICK_GROUPING", "MARK_LABEL", "VALUE_LABEL", "OTHER"],
                        img_type="png",
                        citation=_CITATION["OTHER"]),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=self.config.labels
                        )
                    ),
                    "image": datasets.features.Image(),
                    "polygon": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                }
            ),
            citation=self.config.citation,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.extract(self.config.data_dir)
        if self.config.name == "ICPR2022Real":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_files}/ICPR2022/real/train/"}
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_files}/ICPR2022/real/test/"}
                ),
            ]
        if self.config.name == "CHIME-R":
            return [
                datasets.SplitGenerator(
                    name="full", gen_kwargs={"filepath": f"{downloaded_files}/CHIME-R/"}
                ),
            ]
        if self.config.name == "DeGruyter":
            return [
                datasets.SplitGenerator(
                    name="full", gen_kwargs={"filepath": f"{downloaded_files}/DeGruyter/"}
                ),
            ]
        if self.config.name == "EconBiz":
            return [
                datasets.SplitGenerator(
                    name="full", gen_kwargs={"filepath": f"{downloaded_files}/EconBiz/"}
                ),
            ]

    def _generate_examples(self, filepath, filepaths_tar=None):

        if filepaths_tar is not None:
            for tar in filepaths_tar:
                filepath = os.path.join(tar, filepath)
                return self.generate(filepath)

        else:
            return self.generate(filepath)

    def generate(self, filepath):

        dataset_format1 = ["ICPR2022Real"]
        dataset_format2 = ["CHIME-R", "DeGruyter", "EconBiz"]

        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")

        files = [f for f in sorted(os.listdir(ann_dir)) if (not f.startswith('.')) and (f.endswith(".json"))]
        for guid, file in enumerate(files):
            words = []
            bboxes = []
            poly_bboxes = []
            labels = []
            id_role = {}

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            img_path = os.path.join(img_dir, file)
            img_path = img_path.replace("json", self.config.img_type)
            img, size = load_image(img_path)

            if self.config.name not in dataset_format2:
                output = data["task3"]["output"]
                for item in output["text_roles"]:
                    id_role[item["id"]] = item["role"]

                input = data["task3"]["input"]
                # chart_type = input["task1_output"]["chart_type"]

                words_example = input["task2_output"]["text_blocks"]
                words_example = [w for w in words_example if w["text"].strip() != ""]

                for w in words_example:
                    label = id_role[w["id"]]
                    if label.upper() not in self.config.labels:
                        words.append(w["text"])
                        if self.config.name in dataset_format1:
                            bboxes.append(normalize_bbox(quad_to_box(w["polygon"]), size, type="polygon"))
                            poly_bboxes.append([w["polygon"][key] for key in sorted(w["polygon"].keys())])
                        labels.append("OTHER")
                    else:
                        words.append(w["text"])
                        if self.config.name in dataset_format1:
                            bboxes.append(normalize_bbox(quad_to_box(w["polygon"]), size, type="polygon"))
                            poly_bboxes.append([w["polygon"][key] for key in sorted(w["polygon"].keys())])
                        labels.append(label.upper())
            else:
                output = data["textelements"]
                for item in output:
                    id_role[item["id"]] = item["role"]

                for w in output:
                    label = id_role[w["id"]]
                    if label.upper() not in self.config.labels:
                        words.append(w["content"])
                        quad = get_quad(w["boundingbox"], size[0], size[1])
                        bboxes.append(normalize_bbox(quad_to_box(quad), size, type="polygon"))
                        poly_bboxes.append([quad[key] for key in sorted(quad.keys())])
                        labels.append("OTHER")
                    else:
                        words.append(w["content"])
                        quad = get_quad(w["boundingbox"], size[0], size[1])
                        bboxes.append(normalize_bbox(quad_to_box(quad), size, type="polygon"))
                        poly_bboxes.append([quad[key] for key in sorted(quad.keys())])
                        labels.append(label.upper())

            yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "labels": labels, "image": img,
                         "polygon": poly_bboxes}
