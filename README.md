# MNIST Handwritten Digit Recognition - CNN Training Notebook

This repository contains the Jupyter Notebook (`CNN.ipynb`) used to build, train, and evaluate a Convolutional Neural Network (CNN) for classifying handwritten digits from the famous MNIST dataset. The final trained model is saved as `mnist_cnn_model.h5`, which is then used by the accompanying FastAPI application.

## Notebook Overview (`CNN.ipynb`)

The notebook is structured to be a clear, step-by-step guide through the process of creating a deep learning model for image classification.

### 1. Setup and Imports

The notebook begins by importing essential libraries for deep learning and data manipulation:
- **TensorFlow & Keras**: For building and training the neural network.
- **NumPy**: For numerical operations.
- **Matplotlib**: For visualizing the data and training results.
- **Scikit-learn**: For splitting the dataset.

### 2. Data Loading and Preprocessing

- **Loading MNIST**: The standard MNIST dataset is loaded directly using `tf.keras.datasets.mnist.load_data()`. This dataset consists of 60,000 training images and 10,000 testing images of 28x28 pixel grayscale handwritten digits (0-9).
- **Normalization**: Pixel values are scaled from the original range of `[0, 255]` to `[0, 1]` by dividing by 255.0. This helps stabilize and accelerate the training process.
- **Data Splitting**: The training data is further split into training and validation sets (80/20 split) to monitor the model's performance on unseen data during training.

### 3. Data Augmentation

To prevent overfitting and improve the model's ability to generalize, `ImageDataGenerator` is used to apply random transformations to the training images in real-time. The augmentations include:
- Random rotations
- Zooming
- Width and height shifts

### 4. Model Architecture

A Sequential Keras model is defined with the following layers, creating a robust CNN for image classification:

1.  **Conv2D** (32 filters, 3x3 kernel, ReLU activation) - *Input Layer*
2.  **MaxPooling2D** (2x2 pool size)
3.  **BatchNormalization**
4.  **Conv2D** (64 filters, 3x3 kernel, ReLU activation)
5.  **MaxPooling2D** (2x2 pool size)
6.  **BatchNormalization**
7.  **Flatten** - To convert the 2D feature maps into a 1D vector.
8.  **Dense** (128 neurons, ReLU activation)
9.  **Dropout** (rate of 0.2) - For regularization.
10. **BatchNormalization**
11. **Dense** (64 neurons, ReLU activation)
12. **Dropout** (rate of 0.2)
13. **BatchNormalization**
14. **Dense** (10 neurons, Softmax activation) - *Output Layer*, producing a probability distribution over the 10 digit classes.

### 5. Model Compilation and Training

- **Compilation**: The model is compiled with:
  - **Optimizer**: `adam`
  - **Loss Function**: `sparse_categorical_crossentropy` (suitable for integer-based multi-class labels).
  - **Metrics**: `accuracy`.
- **Early Stopping**: An `EarlyStopping` callback is used to monitor the validation loss. It stops the training if there is no significant improvement for 20 consecutive epochs, preventing overfitting and saving the best version of the model.
- **Training**: The model is trained for 10 epochs using the augmented data generator (`datagen.flow`).

### 6. Evaluation and Visualization

After training, the notebook plots the training and validation **accuracy** and **loss** over epochs. These plots are crucial for diagnosing the model's learning behavior and checking for signs of overfitting or underfitting.

### 7. Saving the Model

The final and most important step for the API is saving the trained model. The following line of code should be executed at the end of the notebook:

```python
model.save('mnist_cnn_model.h5')
```

This command saves the entire model—architecture, weights, and optimizer state—into a single HDF5 file named `mnist_cnn_model.h5`. This file is then loaded by the FastAPI application to serve predictions.

## How to Run

1.  Ensure you have the necessary libraries installed (`tensorflow`, `matplotlib`, `scikit-learn`, `numpy`).
2.  Open `CNN.ipynb` in a Jupyter environment (like Jupyter Lab, Jupyter Notebook, or Google Colab).
3.  Run the cells in order from top to bottom.
4.  After the final cell runs, the `mnist_cnn_model.h5` file will be created in the same directory.

---

*This notebook serves as the model creation part of the larger FastAPI project.*
