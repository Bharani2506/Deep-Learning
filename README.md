# Vehicle Detection and Classification in Accident Images (Project one)

## Project Overview
This project employs neural networks to tackle vehicle detection and classification in accident scenarios. Using the "Accident Images Analysis" dataset, which comprises 2,946 images of damaged vehicles categorized into three classes—light vehicles, heavy vehicles, and motorcycles—we assess various models for their efficiency and accuracy.

## Data Preprocessing
The images, formatted in RGB and resized to 224x224 pixels, undergo several preprocessing steps:
- Conversion to NumPy arrays for machine learning compatibility.
- Normalization and one-hot encoding for classification.

## Models Overview
- **Multi-Layer Perceptron (MLP)**: Establishes a baseline for performance using simple network architectures.
- **Convolutional Neural Networks (CNN)**: Leverages spatial hierarchies to improve feature extraction and classification accuracy.
- **Transfer Learning**: Implements pre-trained models to enhance learning efficiency and model accuracy on our specific task.

## Model Performance
We evaluated each model's effectiveness, noting that CNNs generally outperformed MLPs in accuracy and adaptability to new data. Transfer learning techniques showed promising results, improving upon base model accuracies.


## Technologies Used
- **Python**: For scripting and model development.
- **TensorFlow and Keras**: For building and training neural network models.
- **NumPy**: For data manipulation and processing.

## Setup and Installation
To replicate and run this project locally:
1. Clone this repository:
2. Install required Python packages:
3. Launch the Jupyter notebooks:


## Usage
After setting up, you can experiment with the models in the Jupyter Notebooks provided within the repository to see how each model performs with the dataset.

## Contributing
Interested in contributing? Great! You can start by:
- Improving existing models or documentation.
- Adding new models or features.
- Reporting or fixing bugs.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments
- Thanks to all the contributors who have helped shape this project.
- Dataset courtesy of the "Accident Images Analysis".



# Predicting Car Park Occupancy Rates in Birmingham (Project two)

## Project Overview
This project focuses on forecasting the occupancy rates of car parks in Birmingham using a dataset sourced from Birmingham City Council. We employ both Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs) to model and predict occupancy rates from historical data.

## Dataset Description
The dataset includes data from car parks operated by NCP in Birmingham, specifically filtered for car parks associated with shopping areas. The data spans from October to December and is used to predict occupancy rates as a time series forecasting task.

## Model Description
- **RNN Models**: Utilize LSTM and GRU architectures to handle sequential data by maintaining internal states.
- **CNN Models**: Adapted for sequence data processing, CNNs analyze local patterns over time using convolutional operations.

## Performance
The best-performing model in terms of RMSE is a CNN model, achieving a lower error rate compared to RNN models. The CNN model demonstrates superior capability in capturing the temporal dynamics without overfitting.

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy

## Installation
To set up the project environment:
1. Clone the repository:
2. Install required dependencies:


## Usage
Navigate to the repository directory and run the Jupyter notebooks to train models and make predictions:

## Visualizations
Visual comparisons of predicted vs. actual values, training, and validation loss for both RNN and CNN models are included in the Jupyter notebooks to illustrate model performance.


## Contributing
Contributions to improve the models or explore new datasets are welcome. Please fork the repository and submit a pull request with your enhancements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References
- Dataset obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/car+evaluation)
- Neural network layers and training procedures are based on implementations from [TensorFlow Keras](https://keras.io/api/).


