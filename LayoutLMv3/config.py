utils_config = {
    "data_dir": {
        "ICPR2022Real": "/datasets/ICPR2022/real/ICPR2022Real.tar.gz",
        "CHIME-R": "/datasets/CHIME-R/CHIME-R.tar.gz",
        "DeGruyter": "/datasets/DeGruyter/DeGruyter.tar.gz",
        "EconBiz": "/datasets/EconBiz/EconBiz.tar.gz",
    },
    "column_names": {"image": "image",
                     "text": "words",
                     "boxes": "bboxes",
                     "label": "labels"},
    "pretrained_model": "microsoft/layoutlmv3-base",
    "label_list": ["CHART_TITLE", "LEGEND_TITLE", "LEGEND_LABEL", "AXIS_TITLE", "TICK_LABEL",
                   "TICK_GROUPING", "MARK_LABEL", "VALUE_LABEL", "OTHER"],
    "return_entity_level_metrics": False,
}
