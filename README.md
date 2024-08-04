# Principal Component Analysis on Palmer Archipelago Penguins

![Header Image](https://allisonhorst.github.io/palmerpenguins/reference/figures/lter_penguins.png)

Welcome to the GitHub repository for the Principal Component Analysis (PCA) project on the Palmer Archipelago penguins dataset. This repository contains code and documentation related to analyzing the physical characteristics of penguins using PCA.

## Overview

The Palmer Archipelago penguins dataset provides detailed measurements of three penguin species: Adelie, Chinstrap, and Gentoo. The dataset includes key physical measurements such as culmen length, culmen depth, flipper length, body mass, and sex.

In this project, we apply Principal Component Analysis (PCA) to simplify the complexity of these measurements by transforming them into a smaller set of uncorrelated variables known as principal components. This technique helps in understanding the significant patterns in the dataset and visualizing the relationships between different species and measurements.

## Objectives

The primary objectives of this analysis are:

1. **Correlation Analysis**: Identify the most correlated pairs of physical characteristics among the penguins.
2. **Component Selection**: Determine the number of principal components required to capture the variability in the dataset.
3. **PCA Execution**: Perform PCA on the correlation matrix and decide the number of components to retain.
4. **Component Interpretation**: Explain the meaning of each principal component in terms of penguin physical characteristics.
5. **Species Positioning**: Analyze the PCA results to see how different penguin species are positioned relative to each principal component.
6. **Component Contribution**: Evaluate the contributions of principal components to the explained variability of the original variables and vice versa.
7. **Clustering Analysis**: Investigate clustering within the data using hierarchical and non-hierarchical methods to determine the optimal number of clusters.
8. **Cluster Evaluation**: Assess the quality and coherence of the identified clusters.
9. **Summary**: Summarize key findings from the PCA and clustering analyses to highlight insights into species differentiation and physical characteristics.

## Dataset

The dataset used for this analysis is included in the `data` folder:

- `data/penguins_size.csv`: The CSV file containing measurements of the Palmer Archipelago penguins.

## Files in This Repository

- `pca_analysis.ipynb`: The main Jupyter Notebook containing the PCA analysis and results.
- `lib.py`: A Python script containing helper functions used in the analysis.
- `data/penguins_size.csv`: The dataset used for PCA.
- `requirements.txt`: A file listing the Python packages required to run the notebook.

## Installation

To get started with this project, you need to install the required Python packages. You can do this using `pip`:


    pip install -r requirements.txt

## Usage

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/your-username/your-repository-name.git

2. Navigate to the project directory:
    ```bash
    cd palmer-penguins

3. Open the Jupyter Notebook:
    ```bash
    principal-component-analysis-on-palmer-penguins.ipynb

4. Follow the instructions in the notebook to perform the PCA analysis.
5. If needed, review or modify the helper functions in lib.py.

Have fun!
