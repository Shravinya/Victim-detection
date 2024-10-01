# Victim Detection in Earthquake Scenarios

This project focuses on developing a machine learning model to detect victims in earthquake scenarios using image classification techniques. The model is trained on a dataset of images to identify the presence of victims effectively, aiding in disaster response efforts.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In the aftermath of a disaster, timely detection of victims is critical for effective rescue operations. This project employs convolutional neural networks (CNN) to automatically identify victims from images captured at disaster sites.

## Dataset

The dataset used for training and testing the model includes images of victims and non-victims in various disaster scenarios. The dataset is organized into separate folders for training and testing.

- **Training Data:** Located in the `victim_data` directory.
- **Testing Data:** Located in the `archive (8)` directory.

## Installation

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- NumPy
- Keras
- Matplotlib (optional, for visualizing results)

You can install the required packages using pip:

```bash
pip install tensorflow numpy keras matplotlib
