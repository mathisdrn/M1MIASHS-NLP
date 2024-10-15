# NLP Project

This repository hosts the code, text and data of a research study on the relation between head coach and change of performance in football teams. 

To view the content of this research study, you can either:
- Visit the [hosted version](https://mathisdrn.github.io/M1MIASHS-NLP/) of this paper
- Download the [PDF version](https://raw.githubusercontent.com/mathisdrn/M1MIASHS-NLP/master/exports/project.pdf) of this project

### Usage

To run the code in this repository, you will need to create a conda environment with the dependencies specified in the `requirements.yml` file. You can do so by running the following command in your terminal:

```bash
conda env create -f requirements.yml
```

Then, you can activate the environment and run the code in the Jupyter notebooks.

You can build the paper by installing [Typst](https://github.com/typst/typst) and running the following command in your terminal:

```bash
myst build Paper.md --pdf
```

You can also serve a static webpage of the paper by running the following command in your terminal:

```bash
myst start
```