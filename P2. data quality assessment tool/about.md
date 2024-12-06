# Data Quality Assessment Tool

## Overview

A machine learning data assessment tool is designed to evaluate the quality and suitability of datasets for ML projects. It helps data professionals ensure that the data they are working with is clean, relevant and properly structured for training models.

## Features

- **Data Quality Evaluation** Assess aspects of data quality such as completeness, accuracy, consistency and timeliness; identifying issues such as missing values, duplicates and outliers which could negatively impact the models' performance.
- **Feature Assessment** Look at the features in the dataset to determine its relevance and importance for the task at hand by analyzing distributions, correlations and potential multicollinearity among features.
- **Data Preparation Guidance** Recommend data preprocessing steps like normalization, standardizatino, encoding categorical values or handling missing data
- **Performance metrics** Offer insights into how well a dataset can support model training by simulating model performance based on the data characteristics. he simulation could include running preliminary tests to guage how different features impact the model accuracy.
- **Iterative improvement** Iteratively access data quality and performace metrics to improve both the dataset and the models built on it.


## Getting started

The data is gotten from a spreadsheet of links to data. Data quality checks are made then feature analysis is carried out.  
From the feature analysis, the user would be able to know what features are more important than others; which to use , which to discard and which to build up on.

### Prerequisites

- Python
- Required Libraries
  - NumPy
  - Pandas
  - Scikit-learn
  - PyTorch
  - Matplotlib

### Objectives

The goal of this tool is to assess:

- Completeness (missing values)
- Consistency (data format and types)
- Accuracy (correctness of data)
- Relevance of features

#### Core Functionalities

- **Data quality checks** to identify: missing values ,duplicates ,outliers (z-score)   
  Use statistical methods and visualizations to access data distributions
- **Feature analysis** Evaluate feature importance, correlations and distributions
- **Preprocessing recommendations** Based on the assessment, suggest what to do for data preprocessing **

### Data Source

The data is collected from [InformationIsBeautiful](https://informationisbeautiful.net/data/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request to discuss your changes.

## Acknowledgments

- Credit to the authors of the libraries and tools used in this project.
- Inspiration from various research papers and anomaly detection frameworks.
- Credit to the InformationIsBeautiful team for keeping such detailed logs of their datasets!

## Next steps
