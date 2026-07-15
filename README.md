# A3Net

This is the implementation of A3Net.

## Download A3Net
You can download the models we trained for each dataset from [here](https://github.com/Vinh-AI/A3Net/blob/main/data/a3net.md).

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic_cxr`.

## Run on IU X-Ray

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.

## Run on MIMIC-CXR

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

## Test on IU X-Ray

Run `bash test_iu_xray.sh` to train a model on the IU X-Ray data.

## Test on MIMIC-CXR

Run `bash test_mimic_cxr.sh` to train a model on the MIMIC-CXR data.
