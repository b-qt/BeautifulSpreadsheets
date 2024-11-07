# Anomaly Detection Project

## Overview
This project aims to develop a robust anomaly detection system that identifies unusual patterns or outliers in data taken from projects in the InformationIsBeautiful [website](https://informationisbeautiful.net/data/) to better understand the biases and assumptions in the publication.

## Features
- **Data Preprocessing**: Clean and prepare data for analysis, including handling missing values and normalization.
- **Anomaly Detection Algorithms**: Implementation of scikit-learn's Isolation Forest Machine Learning model
- **Visualization**: Tools to visualize data and detected anomalies for better interpretation.
- **Performance Evaluation**: Metrics to assess model performance, including precision, recall, and F1-score.

## Getting Started

### Prerequisites
- Python 3.6 or higher
- Required libraries:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn

## Contributing
Contributions are welcome! Please open an issue or submit a pull request to discuss your changes.

## Acknowledgments
- Credit to the authors of the libraries and tools used in this project.
- Inspiration from various research papers and anomaly detection frameworks.
- Credit to the InformationIsBeautiful team for keeping such detailed logs of their datasets!


## Next steps
- Use other algorithms: Statistical Methods (Z-Score, IQR) and Deep Learning techniques like AutoEncoders and SVMs to detect the anomalies
- Make the percentage of anomalies to be identified in the IsolationForest interactive and customizable 
- Make the app work for mulitple datasets; not just those from informationisbeautiful