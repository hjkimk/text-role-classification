# Text Role Classification of Scientific Charts Using Multimodal Transformers

This repository contains the source code for the paper: "Text Role Classification of Scientific Charts Using Multimodal Transformers"

## Abstract: 
Text role classification involves classifying the semantic role of textual elements within scientific charts. For this task, we propose to finetune two pretrained multimodal document layout analysis models, LayoutLMv3 and UDOP, on chart datasets. The transformers utilize the three modalities of text, image, and layout as input. We further investigate whether data augmentation and balancing methods help the performance of the models. The models are evaluated on various chart datasets, and results show that LayoutLMv3 outperforms UDOP in all experiments. LayoutLMv3 achieves the highest F1-macro score of 82.87 on the ICPR22 test dataset, beating the best-performing model from the ICPR22 CHART-Infographics challenge. Moreover, the robustness of the models is tested on a synthetic noisy dataset ICPR22-N. Finally, the generalizability of the models is evaluated on three chart datasets, CHIME-R, DeGruyter, and EconBiz, for which we added labels for the text roles. Findings indicate that even in cases where there is limited training data, transformers can be used with the help of data augmentation and balancing methods.

## Details:
* The pretrained model for LayoutLMv3 can be found here: https://github.com/microsoft/unilm/tree/master/layoutlmv3 (Huggingface model path is used in the paper)
* The pretrained model for UDOP can be found here: https://github.com/microsoft/i-Code/tree/main/i-Code-Doc. After downloading, update path for "pretrained_model" in the corresponding training_config json file for the job script.
* The training configs and job scripts for finetuning the models can be found in the training_config and job_scripts directory for each model
    - ALL = finetuning on all datasets
    - ALL_DAB = finetuning on all datasets with data augmentation and balancing methods
    - ICPR22 = finetuning on only ICPR22 dataset
    - ICPR22_DAB = finetuning on only ICPR22 dataset with data augmentation and balancing methods

## Datasets:
* The datasets directory includes the images and annotations for CHIME-R, EconBiz, and ICPR22-N.
* For DeGruyter, only the annotations are provided. The images can be separately downloaded here: https://www.degruyter.com/ 
* The images and annotations for ICPR22 train and test datasets can be downloaded here: https://chartinfo.github.io/toolsanddata_2022.html

## Example:

Example of finetuning LayoutLMv3 on only ICPR22 dataset:
1. Inside LayoutLMv3 folder, create virtual environment
```
python -m venv venv
```
2. Activate venv environment and install requirements
```
source venv/bin/activate
pip install -r requirements.txt
```
3. Run main file with corresponding training config file
```
python3 main.py training_config/ICPR22.json
```

To run batch job in SLURM environment:
```
sbatch -p gpu_4 job_scripts/ICPR22.sh
```