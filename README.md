# Semantic matching for long term visual localization
A comprehensive repository to perform visual localization in presence of visual ambiguities and in challenging long-term scenarios.
 
<!-- | **License** | **Language** | **Libraries** |
| ----- | ---- | ---- |
| ![Licence](https://img.shields.io/badge/Licence-MIT-orange) |  ![Python](https://img.shields.io/badge/Python-yellow)| ![Pytorch](https://img.shields.io/badge/Pytorch-1.8.1-brightgreen) ![Flask](https://img.shields.io/badge/Flask-2.0.0-brightgreen) ![Streamlit](https://img.shields.io/badge/Streamlit-0.82.0-brightgreen) ![Pandas](https://img.shields.io/badge/Pandas-1.2.4-brightgreen)  -->
 
### Contents
- [Description](#description)
- [Resources](#resources)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Authors](#authors)

<a name="description"/>

## Description 
The repo includes functionality to perform visual localization, starting from a database of queries and a 3D [COLMAP](https://colmap.github.io/) reconstruction of the same scene. The code offers flexibility to add bias during the matching and RANSAC phases of the localization performance, and to compute the geometric-semantic match consistency score proposed by _Toft et al., Semantic Match Consistency for Long-Term
Visual Localization, 2017_.

<a name="resources"/>

## Resources
A full explanation of the proposed method and performed experiments can be found in ... . The executive summary can be found here ...

<a name="dataset"/>

## Dataset
All experiments are performed on the Extended CMU Seasons dataset available at https://www.visuallocalization.net/datasets/. The dataset consists of several slices (independent image subsets, each consisting of several collections of images of the same route, taken during different seasons and visual conditions). Please refer to that page for a comprehensive explanation of the dataset.

<a name="installation"/>.

## Installation
- download the Extended CMU Seasons dataset from https://www.visuallocalization.net/datasets/, installing it in the folder `/data/Extended-CMU-Seasons/`. You may download all slices or only some of those.
- follow the instructions from https://github.com/maunzzz/fine-grained-segmentation-networks to set up the fine grained segmentation network. We have used their pre-trained network (CMU dataset). Please save the trained weights in the folder `/Models/trained/fgsn/`.
- run the `/experiments/setup.py` algorithm to save segmentations for the slice you intend to analyse. (Warning: large memory consumption)


<a name="usage"/>

## Usage
You may find code for the main experiments of our work in the folder `/experiments`. Run these individually to perform the experiments.

<a name="demo"/>

## Demo

<a name="authors"/>

## Authors
This project was developed by Valentina Sgarbossa [[Email](mailto:valentina.sgarbossa@mail.polimi.it)][[Github](https://github.com/vale9888)] under the supervision of Antonino Maria Rizzo [[Email](mailto:antoninomaria.rizzo@mail.polimi.it)][[Github](https://github.com/rizzoantoninomaria)] and Luca Magri [[Email](mailto:luca.magri@polimi.it)][[Webpage](https://magrilu.github.io/)], in the Department of Electronics, Informatics and Bioengineering at Politecnico di Milano.

The `/fine_grained_segmentation` code was adapted from the repository at https://github.com/maunzzz/fine-grained-segmentation-networks, _Larsson et al., Fine-Grained Segmentation Networks: Self-Supervised Segmentation for Improved Long-Term Visual Localization, 2019_.
