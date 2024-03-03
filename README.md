# Multiview Contrast Mammography Model (MVCM)

This is the implementation of the MVCM model mentioned in our paper 'Validation of artificial intelligence contrast mammography in diagnosis of breast cancer: Relationship to histopathological results'.

Paper link [here](https://authors.elsevier.com/c/1igk6,GNpjzEe~).

We built a multiview deep learning model (MVCM) to classify and segment malignancy in contrast-enhanced mammography images. The model was trained on the [CDD-CESM Dataset](doi.org/10.1038/s41597-022-01238-0).

![image](https://github.com/omar-mohamed/Birad-Classification-Multiview/assets/6074821/f517816e-5c03-4b38-8ee4-af304b5a1be4)

## Sample Predictions
<img width="677" alt="image" src="https://github.com/omar-mohamed/Birad-Classification-Multiview/assets/6074821/cbfeeaba-77f1-49ed-b3d7-6efdb7155e1c">


## Installation & Usage
*The project was tested on a virtual environment of python 3.7, pip 24.0, and MacOS*

- pip install -r full_requirements.txt (or pip install -r requirements.txt if there are errors because of using a different operating system, as requirements.txt only contains the main dependencies and pip will fetch the compatible sub-dependencies, but it will be slower)
- modify configs.py to point to the training/testing sets & control the training flow
- python train.py
- python test.py (to evaluate the model)

## Related Repositories
- CDD-CESM Dataset [here](https://github.com/omar-mohamed/CDD-CESM-Dataset).

## Citation
To cite this paper, please use:

```
@article{HELAL2024111392,
title = {Validation of artificial intelligence contrast mammography in diagnosis of breast cancer: Relationship to histopathological results},
journal = {European Journal of Radiology},
volume = {173},
pages = {111392},
year = {2024},
issn = {0720-048X},
doi = {https://doi.org/10.1016/j.ejrad.2024.111392},
url = {https://www.sciencedirect.com/science/article/pii/S0720048X24001086},
author = {Maha Helal and Rana Khaled and Omar Alfarghaly and Omnia Mokhtar and Abeer Elkorany and Aly Fahmy and Hebatalla {El Kassas}}
}
```
