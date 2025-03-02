# Road Traffic Condition Monitoring using Deep Learning

## Overview
This project utilizes Deep Learning to monitor and classify road traffic conditions based on image inputs. A Convolutional Neural Network (CNN) model is trained to classify road conditions into four categories:
- **Accident Occurred**
- **Heavy Traffic Detected**
- **Fire Accident Occurred**
- **Low Traffic**

The system preprocesses images, trains a CNN model, and provides predictions for real-time traffic conditions using a GUI-based application.

## Features
- **Dataset Upload**: Load the dataset for training.
- **Image Preprocessing**: Preprocess images to prepare them for training.
- **CNN Model Training**: Train a deep learning model to classify road traffic conditions.
- **Traffic Prediction**: Upload an image and predict its traffic condition.
- **Accuracy & Loss Graph**: Visualize training performance with accuracy and loss graphs.

## Technologies Used
- **Python**
- **TensorFlow/Keras** (for Deep Learning)
- **OpenCV** (for image processing)
- **Matplotlib** (for visualization)
- **Tkinter** (for GUI)
- **NumPy**
- **Pickle** (for model serialization)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/road-traffic-monitoring.git
   cd road-traffic-monitoring
   ```
2. Install dependencies:
   ```sh
   pip install numpy opencv-python tensorflow keras matplotlib
   ```
3. Run the application:
   ```sh
   python RoadTrafficMonitor.py
   ```

## Usage
1. **Upload Dataset**: Click the "Upload Dataset" button to load images for training.
2. **Image Preprocessing**: Click "Image Preprocessing" to preprocess images.
3. **Train CNN Model**: Click "Generate CNN Traffic Model" to train the model.
4. **Predict Traffic**: Click "Upload Test Image & Predict Traffic" to classify a new image.
5. **View Accuracy & Loss**: Click "Accuracy & Loss Graph" to analyze training performance.

## Model Architecture
- **Convolutional Layers**: Extract image features.
- **MaxPooling Layers**: Reduce spatial dimensions.
- **Fully Connected Layers**: Classify the image into one of four categories.
- **Softmax Activation**: Outputs probabilities for each class.

## Dataset Structure
The dataset consists of labeled images representing different traffic conditions. It is stored in the "model/" directory and used for training.

## Expected Outputs
- A trained CNN model that classifies traffic conditions.
- A GUI-based tool to analyze road traffic images.
- Graphical representation of model accuracy and loss.
