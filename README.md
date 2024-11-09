# Spam Detection - Natural Language Processing Project

This repository hosts the code, text and data of a spam detection NLP Project followed in M1 MIASHS.

To view the content of this research study, you can either:
- Visit the [hosted version](https://mathisdrn.github.io/M1MIASHS-NLP/) of this project
- Download the [PDF version](https://raw.githubusercontent.com/mathisdrn/M1MIASHS-NLP/master/exports/project.pdf) of this project

### Project description

Train a French spam classifier using TfidfVectorizer features. Compare the performance of Naive Bayes, Logistic Regression, and SVM classifiers

### Usage

To run the code in this repository, you will need to create a conda environment with the dependencies specified in the `environment.yml` file. You can do so by running the following command in your terminal:

```bash
conda env create -f environment.yml
```

Then, you can activate the environment and run the code in the Jupyter notebooks.

You can build the paper as a PDF file by installing [Typst](https://github.com/typst/typst) and running the following command in your terminal:

```bash
myst build Paper.md --pdf
```

You can also serve a static webpage of the paper by running the following command in your terminal:

```bash
myst start
```