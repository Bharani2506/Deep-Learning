# Vehicle Detection and Classification in Accident Images

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
