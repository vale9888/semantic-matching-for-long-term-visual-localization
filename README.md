# Semantic matching for long term visual localization
A comprehensive repository to perform visual localization in presence of visual ambiguities and in challenging long-term scenarios.
 
<!-- | **License** | **Language** | **Libraries** |
| ----- | ---- | ---- |
| ![Licence](https://img.shields.io/badge/Licence-MIT-orange) |  ![Python](https://img.shields.io/badge/Python-yellow)| ![Pytorch](https://img.shields.io/badge/Pytorch-1.8.1-brightgreen) ![Flask](https://img.shields.io/badge/Flask-2.0.0-brightgreen) ![Streamlit](https://img.shields.io/badge/Streamlit-0.82.0-brightgreen) ![Pandas](https://img.shields.io/badge/Pandas-1.2.4-brightgreen)  -->

<embed src="Localization.pdf" />
 
### Contents
- [Description](#description)
- [Resources](#resources)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)

<a name="description"/>

## Description 
The repo includes functionality to perform visual localization, starting from a database of queries and a 3D [COLMAP](https://colmap.github.io/) reconstruction of the same scene. The code offers flexibility to add bias during the matching and RANSAC phases of the localization performance, and to compute the geometric-semantic match consistency score proposed by _Toft et al., Semantic Match Consistency for Long-Term
Visual Localization, 2017_.

<a name="resources"/>

## Resources
The executive summary of the proposed method and a detailed explanation of the work and performed experiments can be found on [POLITesi](http://hdl.handle.net/10589/191937).

<a name="dataset"/>

## Dataset
All experiments are performed on the Extended CMU Seasons dataset (see _Badino et al., The CMU Visual Localization Data Set, 2011_ and _Sattler et al., Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions, 2018_ ) available at https://www.visuallocalization.net/datasets/. The dataset consists of several slices (independent image subsets, each consisting of several collections of images of the same route, taken during different seasons and visual conditions). Please refer to the dataset webpage for a comprehensive explanation of the formats and contents.

<a name="installation"/>.

## Installation
- download the Extended CMU Seasons dataset from https://www.visuallocalization.net/datasets/, installing it in the folder `/data/Extended-CMU-Seasons/`. You may download all slices or only some of those.
- follow the instructions from https://github.com/maunzzz/fine-grained-segmentation-networks to set up the fine grained segmentation network. We have used their pre-trained network (CMU dataset). Please save the trained weights in the folder `/Models/trained/fgsn/`.
- run the `/experiments/setup.py` algorithm to save segmentations for the slice you intend to analyse. (Warning: large memory consumption)


<a name="usage"/>

## Usage
You may find code for the main experiments of our work in the folder `/experiments`. Run these individually to perform the experiments.

<a name="authors"/>

## Authors
This project was developed by Valentina Sgarbossa [[Email](mailto:valentina.sgarbossa@mail.polimi.it)][[Github](https://github.com/vale9888)] under the supervision of Antonino Maria Rizzo [[Email](mailto:antoninomaria.rizzo@mail.polimi.it)][[Github](https://github.com/rizzoantoninomaria)] and Luca Magri [[Email](mailto:luca.magri@polimi.it)][[Webpage](https://magrilu.github.io/)], in the Department of Electronics, Informatics and Bioengineering at Politecnico di Milano.

The `/fine_grained_segmentation` code was adapted from the repository at https://github.com/maunzzz/fine-grained-segmentation-networks, _Larsson et al., Fine-Grained Segmentation Networks: Self-Supervised Segmentation for Improved Long-Term Visual Localization, 2019_.
