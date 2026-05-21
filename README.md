# 🧠 TensorFlow: Complete Notes
### From Absolute Beginner → Industry-Ready Production Engineer

> **Who is this for?** ML/DL students, AI/ML engineers, data scientists, and anyone who wants to go from zero to production with TensorFlow.  
> **How to use:** Read sequentially. Every section builds on the last. Code every example yourself.

---

## 📋 Table of Contents

1. [What is TensorFlow?](#1-what-is-tensorflow)
2. [Installation & Setup](#2-installation--setup)
3. [Core Concepts: Tensors](#3-core-concepts-tensors)
4. [Variables, Operations & Graphs](#4-variables-operations--graphs)
5. [Automatic Differentiation (AutoGrad)](#5-automatic-differentiation-autograd)
6. [Keras: The High-Level API](#6-keras-the-high-level-api)
7. [Building Neural Networks](#7-building-neural-networks)
8. [Loss Functions & Optimizers](#8-loss-functions--optimizers)
9. [Training Pipeline (Fit, Evaluate, Predict)](#9-training-pipeline-fit-evaluate-predict)
10. [Data Pipelines: tf.data API](#10-data-pipelines-tfdata-api)
11. [Convolutional Neural Networks (CNNs)](#11-convolutional-neural-networks-cnns)
12. [Recurrent Neural Networks (RNNs, LSTM, GRU)](#12-recurrent-neural-networks-rnns-lstm-gru)
13. [Transfer Learning & Fine-Tuning](#13-transfer-learning--fine-tuning)
14. [Custom Training Loops](#14-custom-training-loops)
15. [Callbacks & Training Utilities](#15-callbacks--training-utilities)
16. [Regularization Techniques](#16-regularization-techniques)
17. [Saving & Loading Models](#17-saving--loading-models)
18. [TensorBoard: Visualization](#18-tensorboard-visualization)
19. [TensorFlow Datasets (TFDS)](#19-tensorflow-datasets-tfds)
20. [Distributed Training](#20-distributed-training)
21. [TensorFlow Lite (On-Device ML)](#21-tensorflow-lite-on-device-ml)
22. [TensorFlow Serving (Production Deployment)](#22-tensorflow-serving-production-deployment)
23. [TF SavedModel & TF Serving REST API](#23-tf-savedmodel--tf-serving-rest-api)
24. [Performance Optimization](#24-performance-optimization)
25. [Debugging & Best Practices](#25-debugging--best-practices)
26. [Real-World Project Checklist](#26-real-world-project-checklist)
27. [Quick Reference Cheat Sheet](#27-quick-reference-cheat-sheet)

---

## 1. What is TensorFlow?

TensorFlow is an **open-source machine learning framework** developed by Google Brain, released in 2015. It allows you to build, train, and deploy machine learning and deep learning models at scale.

### Key Features
| Feature | Description |
|---|---|
| **Eager Execution** | Operations execute immediately (like NumPy) |
| **Keras API** | High-level, user-friendly neural network API |
| **tf.data** | Efficient data pipeline building |
| **TF Lite** | Deploy models on mobile/embedded devices |
| **TF Serving** | Production model serving infrastructure |
| **TF.js** | Run models in browsers with JavaScript |
| **Distributed Training** | Scale across multiple GPUs/TPUs |
| **AutoGraph** | Convert Python to optimized TF graphs |

### TensorFlow vs PyTorch (Quick Comparison)
| Aspect | TensorFlow 2.x | PyTorch |
|---|---|---|
| Execution | Eager by default | Eager by default |
| Deployment | Excellent (TF Serving, TFLite) | Good (TorchServe) |
| Mobile | TFLite (mature) | ExecuTorch (newer) |
| Industry Use | Google, Airbnb, Twitter | Meta, Tesla, OpenAI |
| Learning Curve | Medium | Slightly easier |

### TensorFlow 1.x vs 2.x
- **TF 1.x**: Session-based, define-then-run, verbose boilerplate
- **TF 2.x**: Eager execution by default, Keras as the main API, much cleaner code

> **Always use TensorFlow 2.x.** TF 1.x is deprecated.

---

## 2. Installation & Setup

### System Requirements
- Python 3.8 – 3.11
- pip 19.0+
- (Optional) NVIDIA GPU with CUDA support

### Installation

```bash
# CPU-only (recommended for beginners)
pip install tensorflow

# GPU support (requires CUDA + cuDNN installed)
pip install tensorflow[and-cuda]

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Conda Environment (Recommended)

```bash
conda create -n tf_env python=3.10
conda activate tf_env
pip install tensorflow jupyter matplotlib numpy pandas scikit-learn
```

### Google Colab (Zero Setup)

```python
# In Colab, TensorFlow is pre-installed
import tensorflow as tf
print(tf.__version__)  # Usually latest stable

# Enable GPU in Colab: Runtime > Change runtime type > GPU
```

### Standard Imports

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, metrics
import numpy as np
import matplotlib.pyplot as plt

# Suppress excessive warnings (optional)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

---

## 3. Core Concepts: Tensors

A **tensor** is the fundamental data structure in TensorFlow — a multi-dimensional array with a uniform data type.

### Tensor Ranks
| Rank | Shape Example | Name |
|---|---|---|
| 0 | `()` | Scalar |
| 1 | `(5,)` | Vector |
| 2 | `(3, 4)` | Matrix |
| 3 | `(2, 3, 4)` | 3D Tensor |
| N | `(d1, d2, ..., dN)` | N-D Tensor |

### Creating Tensors

```python
# From Python values
a = tf.constant(42)                        # Scalar
b = tf.constant([1, 2, 3])                 # 1D tensor (vector)
c = tf.constant([[1, 2], [3, 4]])          # 2D tensor (matrix)
d = tf.constant([[[1, 2], [3, 4]],
                 [[5, 6], [7, 8]]])        # 3D tensor

# Inspect tensors
print(a.numpy())           # 42
print(b.shape)             # (3,)
print(c.dtype)             # tf.int32
print(d.ndim)              # 3

# Special tensors
tf.zeros([3, 4])           # All zeros, shape (3,4)
tf.ones([2, 3])            # All ones
tf.eye(4)                  # 4x4 identity matrix
tf.random.normal([3, 3])   # Random normal distribution
tf.random.uniform([2, 5], minval=0, maxval=10)  # Random uniform
tf.range(0, 10, delta=2)   # [0, 2, 4, 6, 8]

# From NumPy
arr = np.array([1.0, 2.0, 3.0])
t = tf.constant(arr)
back_to_numpy = t.numpy()  # Convert back
```

### Data Types (dtypes)

```python
# Common dtypes
tf.float16, tf.float32, tf.float64
tf.int8, tf.int16, tf.int32, tf.int64
tf.bool, tf.string, tf.complex64

# Cast between types
x = tf.constant([1, 2, 3], dtype=tf.int32)
y = tf.cast(x, dtype=tf.float32)
```

### Tensor Operations

```python
a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

# Arithmetic
tf.add(a, b)         # or a + b
tf.subtract(a, b)    # or a - b
tf.multiply(a, b)    # or a * b  — element-wise!
tf.divide(a, b)      # or a / b

# Matrix multiplication
tf.matmul(a, b)      # or a @ b

# Reduction operations
tf.reduce_sum(a)              # Sum all elements
tf.reduce_mean(a)             # Mean of all elements
tf.reduce_max(a, axis=0)      # Max along axis 0
tf.reduce_min(a, axis=1)      # Min along axis 1

# Reshaping
x = tf.constant([1,2,3,4,5,6])
tf.reshape(x, [2, 3])         # Shape (2,3)
tf.reshape(x, [3, -1])        # Shape (3,2), -1 inferred
tf.expand_dims(x, axis=0)     # Shape (1,6)
tf.squeeze(tf.expand_dims(x, 0))  # Remove dim-1 axes

# Transpose
tf.transpose(a)               # Shape (2,2) -> (2,2) here
tf.transpose(a, perm=[1, 0])  # Explicit axes

# Indexing & Slicing (NumPy-like)
t = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
t[0]          # First row: [1, 2, 3]
t[:, 1]       # Second column: [2, 5, 8]
t[1:, :2]     # Rows 1-end, columns 0-1
```

### Broadcasting

```python
# TensorFlow broadcasts like NumPy
a = tf.constant([[1, 2, 3]])    # shape (1, 3)
b = tf.constant([[1], [2], [3]])  # shape (3, 1)
a + b  # shape (3, 3) — broadcast!
```

---

## 4. Variables, Operations & Graphs

### tf.Variable

Variables are **mutable tensors** used to store model weights. Unlike `tf.constant`, they can be updated during training.

```python
# Create a variable
w = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
b = tf.Variable(0.0)

# Read value
print(w.numpy())

# Assign new value
w.assign([[5.0, 6.0], [7.0, 8.0]])

# Modify in-place
w.assign_add(tf.ones([2,2]))   # w += 1
w.assign_sub(tf.ones([2,2]))   # w -= 1

# Variable properties
w.shape       # TensorShape([2, 2])
w.dtype       # tf.float32
w.trainable   # True by default

# Non-trainable variable (e.g., step counter)
step = tf.Variable(0, trainable=False)
```

### Computation Graphs & tf.function

TF 2.x runs in **eager mode** by default (execute immediately), but `@tf.function` compiles code to a fast static graph.

```python
# Eager execution (default)
x = tf.constant(3.0)
y = x ** 2 + 2 * x + 1
print(y.numpy())   # 16.0 — runs immediately

# Graph execution with @tf.function
@tf.function
def compute(x):
    return x ** 2 + 2 * x + 1

result = compute(tf.constant(3.0))
print(result.numpy())  # 16.0 — compiled to graph on first call

# Why use @tf.function?
# 1. Much faster for repeated calls
# 2. Enables deployment to TF Serving, TFLite
# 3. Enables XLA compilation
```

### When to Use @tf.function
- Custom training loops
- Any function called millions of times
- Production inference functions
- **Don't use** during debugging (harder to step through)

---

## 5. Automatic Differentiation (AutoGrad)

**Backpropagation** is automated in TensorFlow via `tf.GradientTape`. This is the engine behind all neural network training.

### Basic Gradient Computation

```python
# Record operations for differentiation
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2  # y = x²

# Compute dy/dx
dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())   # 6.0 (derivative of x² is 2x, at x=3 → 6)
```

### Gradient with Respect to Multiple Variables

```python
W = tf.Variable(tf.random.normal([3, 2]))
b = tf.Variable(tf.zeros([2]))
x = tf.constant([[1.0, 2.0, 3.0]])

with tf.GradientTape() as tape:
    y = x @ W + b
    loss = tf.reduce_sum(y ** 2)

gradients = tape.gradient(loss, [W, b])
print("dL/dW:", gradients[0])
print("dL/db:", gradients[1])
```

### Persistent Tape (Multiple Gradient Calls)

```python
x = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
    y = x ** 2
    z = x ** 3

dy_dx = tape.gradient(y, x)   # 6.0
dz_dx = tape.gradient(z, x)   # 27.0
del tape  # Free memory when done
```

### Higher-Order Derivatives

```python
x = tf.Variable(2.0)

with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = x ** 3       # y = x³
    dy_dx = tape1.gradient(y, x)    # 3x² = 12

d2y_dx2 = tape2.gradient(dy_dx, x)  # 6x = 12
print(d2y_dx2.numpy())   # 12.0
```

### Manual Training Step (Building Block)

```python
# This is exactly how Keras does it internally
def train_step(model, x, y_true, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_fn(y_true, y_pred)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

---

## 6. Keras: The High-Level API

Keras is TensorFlow's official high-level neural networks API. It's the recommended way to build and train models.

### Three Ways to Build Models

#### 1. Sequential API (Simplest)

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()
```

#### 2. Functional API (Flexible, for complex architectures)

```python
# Define inputs
inputs = keras.Input(shape=(784,), name='input')

# Define the forward pass
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax', name='output')(x)

# Create model
model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
model.summary()

# Multi-input / Multi-output with Functional API
input_a = keras.Input(shape=(64,), name='input_a')
input_b = keras.Input(shape=(32,), name='input_b')

x = layers.Dense(32, activation='relu')(input_a)
y = layers.Dense(16, activation='relu')(input_b)

merged = layers.concatenate([x, y])
out1 = layers.Dense(1, name='output_1')(merged)
out2 = layers.Dense(5, activation='softmax', name='output_2')(merged)

model = keras.Model(inputs=[input_a, input_b], outputs=[out1, out2])
```

#### 3. Subclassing API (Full Control, PyTorch-like)

```python
class MyModel(keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)  # Only drop during training
        x = self.dense2(x)
        return self.output_layer(x)

model = MyModel(num_classes=10)
```

### Common Keras Layers

```python
# Core layers
layers.Dense(units, activation=None)             # Fully connected
layers.Flatten()                                  # Flatten N-D to 1D
layers.Reshape(target_shape)                      # Reshape tensor

# Activation layers (can also pass as string to Dense)
layers.Activation('relu')
layers.ReLU()
layers.Softmax()
layers.Sigmoid()

# Normalization
layers.BatchNormalization()                       # Normalize activations
layers.LayerNormalization()                       # Layer-wise normalization

# Regularization
layers.Dropout(rate)                              # Randomly zero activations
layers.GaussianNoise(stddev)                      # Add Gaussian noise

# Convolutional
layers.Conv2D(filters, kernel_size, activation)
layers.MaxPooling2D(pool_size)
layers.AveragePooling2D(pool_size)
layers.GlobalAveragePooling2D()
layers.DepthwiseConv2D(kernel_size)

# Recurrent
layers.LSTM(units, return_sequences=False)
layers.GRU(units, return_sequences=False)
layers.SimpleRNN(units)
layers.Bidirectional(layers.LSTM(units))

# Embedding (NLP)
layers.Embedding(input_dim, output_dim, input_length)

# Merging layers
layers.Add()
layers.Multiply()
layers.Concatenate()
layers.Average()
```

---

## 7. Building Neural Networks

### Anatomy of a Neuron

```
Input (x) → Weighted Sum (w·x + b) → Activation Function → Output
```

### Activation Functions

```python
# Common activations and when to use them
activations = {
    'relu':    "Hidden layers — most common, fast",
    'sigmoid': "Binary classification output — outputs (0,1)",
    'softmax': "Multi-class classification output — outputs sum to 1",
    'tanh':    "Recurrent networks — outputs (-1,1)",
    'leaky_relu': "When dying ReLU is a problem",
    'elu':     "Smoother than ReLU, can go slightly negative",
    'swish':   "Google's activation — often beats ReLU",
    'gelu':    "Used in Transformers (BERT, GPT)",
    'linear':  "Regression output layer",
}

# Usage
layers.Dense(64, activation='relu')
layers.Dense(64, activation=tf.nn.relu)
layers.Dense(64, activation=keras.activations.relu)

# Custom activation
def custom_activation(x):
    return tf.nn.relu(x) ** 2

layers.Dense(64, activation=custom_activation)
```

### Complete Classification Example (MNIST)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Preprocess
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 784).astype("float32") / 255.0

# 3. Build model
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=20,
    validation_split=0.1,
    verbose=1
)

# 6. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
```

### Complete Regression Example

```python
# Boston Housing / California Housing style regression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # No activation for regression
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',              # Mean Squared Error
    metrics=['mae']          # Mean Absolute Error
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)
```

---

## 8. Loss Functions & Optimizers

### Loss Functions

```python
# Classification losses
keras.losses.BinaryCrossentropy()                 # Binary classification
keras.losses.CategoricalCrossentropy()            # Multi-class (one-hot labels)
keras.losses.SparseCategoricalCrossentropy()      # Multi-class (integer labels)
keras.losses.KLDivergence()                       # Distribution comparison

# Regression losses
keras.losses.MeanSquaredError()          # MSE — penalizes outliers heavily
keras.losses.MeanAbsoluteError()         # MAE — more robust to outliers
keras.losses.Huber(delta=1.0)            # Combines MSE + MAE (robust)
keras.losses.LogCosh()                   # Smooth approximation of MAE

# String shorthand (equivalent)
model.compile(loss='mse')
model.compile(loss='binary_crossentropy')
model.compile(loss='sparse_categorical_crossentropy')

# When to use which loss?
"""
Binary classification (0/1 output):     binary_crossentropy + sigmoid
Multi-class (one label per sample):     sparse_categorical_crossentropy + softmax
Multi-label (multiple labels):          binary_crossentropy + sigmoid (per class)
Regression (continuous output):         mse or mae or huber
"""

# Custom loss function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred) ** 1.5)

model.compile(loss=custom_loss)

# Class-based custom loss
class WeightedMSE(keras.losses.Loss):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
    
    def call(self, y_true, y_pred):
        return tf.reduce_mean(self.weights * (y_true - y_pred) ** 2)
```

### Optimizers

```python
# SGD variants
keras.optimizers.SGD(learning_rate=0.01)
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Adaptive learning rate optimizers
keras.optimizers.Adam(learning_rate=0.001)           # Most commonly used
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)  # Adam + weight decay
keras.optimizers.RMSprop(learning_rate=0.001)
keras.optimizers.Adagrad(learning_rate=0.01)
keras.optimizers.Adadelta()
keras.optimizers.Nadam()   # Adam + Nesterov

# Learning rate schedules
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

cosine_lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=10000
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# Gradient clipping (prevent exploding gradients)
optimizer = keras.optimizers.Adam(clipnorm=1.0)      # Clip gradient norm
optimizer = keras.optimizers.Adam(clipvalue=0.5)     # Clip gradient value
```

### Optimizer Selection Guide

| Problem | Recommended Optimizer |
|---|---|
| General purpose | Adam |
| Computer vision | SGD + Momentum + LR schedule |
| NLP / Transformers | AdamW |
| Fine-tuning | Adam with small LR (1e-5) |
| Simple problems | SGD |

---

## 9. Training Pipeline (Fit, Evaluate, Predict)

### Model Compilation

```python
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc'),
    ]
)
```

### Training: model.fit()

```python
history = model.fit(
    x_train,           # Training data (numpy array, tf.Tensor, or tf.data.Dataset)
    y_train,           # Training labels
    batch_size=32,     # Samples per gradient update
    epochs=50,         # Full passes through training data
    verbose=1,         # 0=silent, 1=progress bar, 2=one line per epoch
    validation_data=(x_val, y_val),   # Or validation_split=0.2
    shuffle=True,      # Shuffle data each epoch
    class_weight={0: 1.0, 1: 5.0},   # Handle class imbalance
    sample_weight=weights,            # Per-sample weights
    initial_epoch=0,                  # Resume from specific epoch
)

# History object
print(history.history.keys())  # ['loss', 'accuracy', 'val_loss', 'val_accuracy']

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title('Accuracy')

plt.tight_layout()
plt.show()
```

### Evaluation & Prediction

```python
# Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Get predictions
y_pred_proba = model.predict(x_test)             # Raw probabilities (softmax output)
y_pred_class = np.argmax(y_pred_proba, axis=1)  # Class indices

# Single prediction
single = x_test[:1]
pred = model.predict(single)  # Shape: (1, num_classes)

# Predict in batches for large datasets
predictions = model.predict(x_test, batch_size=256)
```

---

## 10. Data Pipelines: tf.data API

The `tf.data` API is TensorFlow's powerful, scalable data pipeline. It handles data loading, shuffling, batching, and preprocessing efficiently — including GPU prefetching.

### Basic tf.data Usage

```python
# From numpy arrays
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Build a standard pipeline
dataset = (
    dataset
    .shuffle(buffer_size=10000)        # Shuffle with buffer
    .batch(32)                          # Create batches of 32
    .prefetch(tf.data.AUTOTUNE)         # Prefetch for GPU
)

# From files
file_dataset = tf.data.Dataset.list_files("data/*.csv")

# Iterate
for x_batch, y_batch in dataset:
    print(x_batch.shape, y_batch.shape)
    break
```

### Preprocessing with .map()

```python
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(5000)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

# Use with model.fit directly
model.fit(train_dataset, epochs=20, validation_data=val_dataset)
```

### Loading Images from Directory

```python
import tensorflow as tf
import pathlib

# Directory structure: data/class_a/img1.jpg, data/class_b/img2.jpg
data_dir = pathlib.Path("data/")

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',          # Use subdirectory names as labels
    label_mode='int',           # 'int', 'categorical', or 'binary'
    image_size=(224, 224),
    batch_size=32,
    seed=42,
    validation_split=0.2,
    subset='training'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(224, 224),
    batch_size=32,
    seed=42,
    validation_split=0.2,
    subset='validation'
)

# Optimize
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

### TFRecord Format (Production Standard)

```python
# Writing TFRecords
def serialize_example(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

with tf.io.TFRecordWriter('train.tfrecord') as writer:
    for img, lbl in zip(images, labels):
        writer.write(serialize_example(img.tobytes(), lbl))

# Reading TFRecords
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def parse_record(serialized_example):
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_raw(parsed['image'], tf.uint8)
    image = tf.cast(image, tf.float32) / 255.0
    label = parsed['label']
    return image, label

raw_dataset = tf.data.TFRecordDataset('train.tfrecord')
dataset = raw_dataset.map(parse_record).batch(32).prefetch(tf.data.AUTOTUNE)
```

---

## 11. Convolutional Neural Networks (CNNs)

CNNs are the go-to architecture for **image data**. They learn spatial hierarchies of features.

### CNN Building Blocks

```
Input Image → [Conv2D → ReLU → Pool] × N → Flatten → Dense → Output
```

| Layer | Purpose |
|---|---|
| Conv2D | Detect local features (edges, textures, shapes) |
| MaxPooling2D | Reduce spatial size, keep dominant features |
| BatchNormalization | Stabilize training |
| GlobalAveragePooling2D | Compact feature representation |
| Dense | Classification/Regression head |

### Complete CNN for CIFAR-10

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

# Build CNN
inputs = keras.Input(shape=(32, 32, 3))
x = data_augmentation(inputs)

# Block 1
x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.2)(x)

# Block 2
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.3)(x)

# Block 3
x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)

# Classifier
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs, name='cifar10_cnn')
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=50,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)
```

### ResNet-style Skip Connections

```python
def residual_block(x, filters, stride=1):
    shortcut = x
    
    x = layers.Conv2D(filters, 3, stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Match dimensions if needed
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, stride)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x
```

---

## 12. Recurrent Neural Networks (RNNs, LSTM, GRU)

RNNs process **sequential data** like text, time series, speech, and video.

### Why LSTM/GRU over Simple RNN?

| Model | Handles Long Dependencies | Speed |
|---|---|---|
| SimpleRNN | ❌ (vanishing gradient) | Fast |
| LSTM | ✅ | Medium |
| GRU | ✅ | Faster than LSTM |

### Text Classification with LSTM

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Example: IMDB sentiment analysis
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# Pad sequences to same length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test  = keras.preprocessing.sequence.pad_sequences(x_test,  maxlen=200)

# Build LSTM model
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=200),
    layers.LSTM(64, return_sequences=True),    # return_sequences=True → output at each step
    layers.LSTM(32),                           # return_sequences=False → only last output
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')      # Binary sentiment
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
```

### Bidirectional LSTM

```python
model = keras.Sequential([
    layers.Embedding(10000, 128),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(1, activation='sigmoid')
])
```

### Time Series Forecasting

```python
# Sequence-to-one prediction
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Model
model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
    layers.LSTM(32),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Predict next value
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

### GRU (Often preferred for speed)

```python
model = keras.Sequential([
    layers.Embedding(vocab_size, embed_dim),
    layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    layers.GRU(64, dropout=0.2),
    layers.Dense(num_classes, activation='softmax')
])
```

---

## 13. Transfer Learning & Fine-Tuning

Transfer learning uses **pre-trained models** (trained on ImageNet, etc.) as a starting point. This gives you state-of-the-art performance with little data.

### Strategy 1: Feature Extraction (Freeze base)

```python
# Load pretrained base model
base_model = keras.applications.EfficientNetB0(
    include_top=False,          # Remove original classifier
    weights='imagenet',         # Use pretrained ImageNet weights
    input_shape=(224, 224, 3)
)

# Freeze the base model
base_model.trainable = False

# Add custom head
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # training=False keeps BN frozen
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train only the head
model.fit(train_ds, epochs=10, validation_data=val_ds)
```

### Strategy 2: Fine-Tuning (Unfreeze top layers)

```python
# After feature extraction training, fine-tune by unfreezing top layers
base_model.trainable = True

# Freeze everything except the last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with much smaller learning rate!
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),   # 100x smaller than before
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds, epochs=10, validation_data=val_ds)
```

### Popular Pretrained Models

```python
# Image classification
keras.applications.VGG16(weights='imagenet')
keras.applications.ResNet50(weights='imagenet')
keras.applications.InceptionV3(weights='imagenet')
keras.applications.EfficientNetB0(weights='imagenet')   # Best accuracy/speed
keras.applications.EfficientNetB7(weights='imagenet')   # More accurate, heavier
keras.applications.MobileNetV2(weights='imagenet')      # Mobile-friendly

# Check preprocessing required
from tensorflow.keras.applications.efficientnet import preprocess_input
x = preprocess_input(x)  # Model-specific normalization
```

---

## 14. Custom Training Loops

For maximum control over training — useful for GANs, RL, custom architectures.

### Basic Custom Loop

```python
# Setup
model = MyModel()
optimizer = keras.optimizers.Adam(0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy()
train_acc = keras.metrics.SparseCategoricalAccuracy()
val_acc   = keras.metrics.SparseCategoricalAccuracy()

@tf.function  # Compile to graph for speed
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
        loss += sum(model.losses)  # Add regularization losses
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc.update_state(y, logits)
    return loss

@tf.function
def val_step(x, y):
    logits = model(x, training=False)
    val_acc.update_state(y, logits)

# Training loop
EPOCHS = 20
for epoch in range(EPOCHS):
    train_acc.reset_state()
    val_acc.reset_state()
    
    # Training
    epoch_loss = 0
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch)
        epoch_loss += loss
    
    # Validation
    for x_batch, y_batch in val_dataset:
        val_step(x_batch, y_batch)
    
    print(f"Epoch {epoch+1}/{EPOCHS} — "
          f"Loss: {epoch_loss/step:.4f} — "
          f"Acc: {train_acc.result():.4f} — "
          f"Val Acc: {val_acc.result():.4f}")
```

### GAN Training Loop Example

```python
class GAN:
    def __init__(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.g_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.d_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.loss_fn = keras.losses.BinaryCrossentropy()
    
    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, 100])
        
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            fake_images = self.generator(noise, training=True)
            
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            
            d_loss = (self.loss_fn(tf.ones_like(real_output), real_output) +
                      self.loss_fn(tf.zeros_like(fake_output), fake_output))
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        
        self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.g_opt.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        return d_loss, g_loss
```

---

## 15. Callbacks & Training Utilities

Callbacks are **hooks** that execute at specific points during training.

```python
callbacks = [

    # Stop training when val_loss stops improving
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,  # Load best weights at end
        verbose=1
    ),

    # Reduce learning rate when plateau detected
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,         # New LR = LR * factor
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),

    # Save checkpoints during training
    keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/model_{epoch:02d}_{val_accuracy:.4f}.keras',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),

    # TensorBoard logging
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,         # Log weight histograms every epoch
        write_graph=True,
        update_freq='epoch'
    ),

    # CSV logging
    keras.callbacks.CSVLogger('training_log.csv', append=True),

    # Learning rate scheduler
    keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-3 * 0.95 ** epoch
    ),
]

model.fit(x_train, y_train, callbacks=callbacks, epochs=100)
```

### Custom Callback

```python
class TrainingMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)
        print(f"\n[Monitor] Epoch {epoch+1}: LR={lr:.6f}, Loss={logs['loss']:.4f}")
    
    def on_train_begin(self, logs=None):
        print("Training started!")
    
    def on_train_end(self, logs=None):
        print("Training complete!")
    
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print(f"  Batch {batch}: loss={logs['loss']:.4f}")
```

---

## 16. Regularization Techniques

Regularization prevents **overfitting** — when a model performs well on training data but poorly on new data.

```python
# 1. Dropout — randomly zero out neurons
layers.Dropout(rate=0.5)    # 50% of neurons set to 0 during training

# 2. L1/L2 Weight Regularization
from tensorflow.keras import regularizers

layers.Dense(64, activation='relu',
             kernel_regularizer=regularizers.l2(0.001))   # L2 (weight decay)
             
layers.Dense(64, activation='relu',
             kernel_regularizer=regularizers.l1(0.001))   # L1 (sparse weights)
             
layers.Dense(64, activation='relu',
             kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))  # Elastic Net

# 3. Batch Normalization — normalize activations per batch
layers.BatchNormalization()   # Place AFTER linear layer, BEFORE activation

# 4. Data Augmentation (best regularizer for images)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.GaussianNoise(0.05),
])

# 5. Label Smoothing (for classification)
keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# 6. Early Stopping
keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# 7. Learning Rate Warmup + Decay
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def __call__(self, step):
        warmup_lr = self.peak_lr * step / self.warmup_steps
        cosine_lr = 0.5 * self.peak_lr * (1 + tf.cos(
            np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
```

---

## 17. Saving & Loading Models

### SavedModel Format (Recommended)

```python
# Save entire model (architecture + weights + optimizer state)
model.save('saved_model/my_model')           # SavedModel format (directory)
model.save('my_model.keras')                 # Keras native format (recommended)

# Load
loaded_model = keras.models.load_model('my_model.keras')
loaded_model = keras.models.load_model('saved_model/my_model')

# Save weights only
model.save_weights('weights.h5')
model.load_weights('weights.h5')

# Save architecture only (JSON)
json_config = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(json_config)

# Load architecture then weights
with open('model_architecture.json') as f:
    json_config = f.read()
new_model = keras.models.model_from_json(json_config)
new_model.load_weights('weights.h5')
```

### Model Checkpointing During Training

```python
# Best practice: save best model during training
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

model.fit(x_train, y_train, callbacks=[checkpoint_cb], validation_split=0.2)

# Load the best model after training
best_model = keras.models.load_model('best_model.keras')
```

### Custom Objects

```python
# If model uses custom layers/losses, register them
@keras.utils.register_keras_serializable()
class MyCustomLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units))
        self.b = self.add_weight(shape=(self.units,))
    
    def call(self, inputs):
        return tf.matmul(inputs, self.W) + self.b
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

# Save and load works automatically with registered classes
model.save('model_with_custom.keras')
loaded = keras.models.load_model('model_with_custom.keras')
```

---

## 18. TensorBoard: Visualization

TensorBoard is a powerful visualization toolkit to monitor training.

### Setup & Launch

```bash
# In terminal
tensorboard --logdir=./logs

# In Jupyter/Colab
%load_ext tensorboard
%tensorboard --logdir logs
```

### Logging with Callbacks

```python
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,        # Weight/bias histograms every epoch
    write_graph=True,        # Log model graph
    write_images=False,      # Log model weights as images
    profile_batch='5,10',    # Profile batches 5-10 for performance
    update_freq='epoch'
)

model.fit(x_train, y_train, callbacks=[tensorboard_cb], epochs=50)
```

### Custom Logging

```python
# Create summary writer
train_writer = tf.summary.create_file_writer('./logs/train')
val_writer   = tf.summary.create_file_writer('./logs/val')

# Log scalar
with train_writer.as_default():
    tf.summary.scalar('custom_metric', value=0.95, step=epoch)

# Log images
with train_writer.as_default():
    tf.summary.image('sample_images', images[:5], step=epoch, max_outputs=5)

# Log histograms
with train_writer.as_default():
    tf.summary.histogram('layer_weights', model.layers[1].weights[0], step=epoch)

# Log text
with train_writer.as_default():
    tf.summary.text('hyperparams', 'lr=0.001, batch=32', step=0)
```

---

## 19. TensorFlow Datasets (TFDS)

TFDS provides ready-to-use datasets for ML.

```python
import tensorflow_datasets as tfds

# List available datasets
tfds.list_builders()

# Load a dataset
(train_ds, val_ds, test_ds), info = tfds.load(
    'cifar10',
    split=['train[:80%]', 'train[80%:]', 'test'],
    shuffle_files=True,
    as_supervised=True,     # Returns (image, label) tuples
    with_info=True
)

# Explore dataset info
print(info)
print(info.features)
print(info.splits['train'].num_examples)

# Preprocess
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Visualize samples
tfds.show_examples(train_ds, info)

# Popular TFDS datasets
"""
Image: 'mnist', 'cifar10', 'cifar100', 'imagenet2012', 'oxford_flowers102'
Text:  'imdb_reviews', 'glue/sst2', 'squad'
Audio: 'speech_commands', 'librispeech'
Video: 'ucf101'
"""
```

---

## 20. Distributed Training

Scale training across multiple GPUs or machines.

### MirroredStrategy (Multiple GPUs, Single Machine)

```python
# Automatically uses all available GPUs
strategy = tf.distribute.MirroredStrategy()
print(f"Number of GPUs: {strategy.num_replicas_in_sync}")

# Build and compile model inside strategy scope
with strategy.scope():
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Training is the same — strategy handles distribution
BATCH_SIZE = 64 * strategy.num_replicas_in_sync   # Scale batch size
model.fit(train_dataset, epochs=10)
```

### TPU Strategy (Google TPU / Colab TPU)

```python
# In Colab with TPU runtime
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.TPUStrategy(resolver)
print(f"Number of TPU cores: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

### MultiWorkerMirroredStrategy (Multiple Machines)

```python
# Set TF_CONFIG environment variable for each worker
import json, os

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['worker0.example.com:12345', 'worker1.example.com:12345']
    },
    'task': {'type': 'worker', 'index': 0}  # Change index per worker
})

strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

---

## 21. TensorFlow Lite (On-Device ML)

Deploy models on **mobile phones, microcontrollers, and edge devices**.

### Convert to TFLite

```python
# Convert a SavedModel or Keras model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Basic conversion
tflite_model = converter.convert()

# With optimization (smaller, faster)
converter.optimizations = [tf.lite.Optimize.DEFAULT]    # Dynamic range quantization
tflite_quantized = converter.convert()

# Full integer quantization (fastest on edge devices)
def representative_dataset():
    for data in train_ds.take(100):
        yield [tf.cast(data[0], tf.float32)]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
tflite_int8_model = converter.convert()

# Save model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"Model size: {len(tflite_model)/1024:.1f} KB")
```

### Run TFLite Inference

```python
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
input_data = np.expand_dims(test_image, axis=0).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)
```

---

## 22. TensorFlow Serving (Production Deployment)

TF Serving is a **production-grade model serving system** built for high throughput.

### Export Model for Serving

```python
# Save model in SavedModel format
model.save('models/my_model/1')  # Version number in path!

# Inspect saved model
# Terminal:
# saved_model_cli show --dir models/my_model/1 --tag_set serve --signature_def serving_default
```

### Deploy with Docker

```bash
# Pull TF Serving Docker image
docker pull tensorflow/serving

# Run the server
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/models/my_model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  -t tensorflow/serving

# Test the server
curl http://localhost:8501/v1/models/my_model
```

### Make REST API Predictions

```python
import requests
import json
import numpy as np

# Prepare input
input_data = x_test[:5].tolist()

# REST API request
payload = {"instances": input_data}
response = requests.post(
    'http://localhost:8501/v1/models/my_model:predict',
    data=json.dumps(payload)
)

predictions = response.json()['predictions']
print(np.argmax(predictions, axis=1))
```

### gRPC Interface (faster for production)

```python
import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import tensorflow as tf

# Connect
channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Build request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'my_model'
request.model_spec.signature_name = 'serving_default'
request.inputs['input'].CopyFrom(
    tf.make_tensor_proto(x_test[:1], dtype=tf.float32)
)

# Send request
result = stub.Predict(request, 10.0)  # 10 second timeout
```

---

## 23. TF SavedModel & TF Serving REST API

### SavedModel Signatures

```python
# Add custom serving signature
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
def serving_fn(inputs):
    return {'predictions': model(inputs)}

# Save with custom signature
model.save(
    'saved_model/my_model',
    signatures={'serving_default': serving_fn}
)

# Load and use
loaded = tf.saved_model.load('saved_model/my_model')
infer = loaded.signatures['serving_default']
result = infer(tf.constant(x_test[:1], dtype=tf.float32))
print(result['predictions'])
```

---

## 24. Performance Optimization

### GPU Memory Management

```python
# Prevent TF from grabbing all GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Limit GPU memory
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4 GB
)
```

### Mixed Precision Training (Speed + Memory)

```python
# Use float16 for computations, float32 for variables
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Build model normally — TF handles dtype automatically
model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(10, dtype='float32', activation='softmax')  # Keep output float32
])

# Use loss scaling to prevent underflow
optimizer = keras.optimizers.Adam(1e-3)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
```

### XLA Compilation

```python
# Enable XLA for all tf.functions
tf.config.optimizer.set_jit(True)

# Or selectively
@tf.function(jit_compile=True)
def fast_matmul(a, b):
    return tf.matmul(a, b)
```

### Profile & Benchmark

```python
# Profile training with TensorBoard
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir='./logs',
    profile_batch='10,20'   # Profile batches 10 to 20
)

# Benchmark inference
import time

@tf.function
def predict_batch(x):
    return model(x, training=False)

# Warmup
predict_batch(x_test[:32])

# Benchmark
N = 100
start = time.time()
for _ in range(N):
    predict_batch(x_test[:32])
end = time.time()
print(f"Average inference time: {(end-start)/N*1000:.2f}ms")
```

### tf.data Pipeline Optimization

```python
# Optimal pipeline order:
dataset = (
    tf.data.Dataset.from_tensor_slices(data)
    .cache()                                  # Cache in memory (if fits)
    .shuffle(buffer_size=1000)               # Shuffle
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # Parallel map
    .batch(batch_size, drop_remainder=True)   # Batch (drop_remainder for TPUs)
    .prefetch(tf.data.AUTOTUNE)              # Prefetch to GPU
)
```

---

## 25. Debugging & Best Practices

### Debugging Techniques

```python
# 1. Use eager execution for debugging (default in TF2)
tf.config.run_functions_eagerly(True)  # Disable @tf.function compilation

# 2. Gradient checking — verify gradients aren't zero or exploding
with tf.GradientTape() as tape:
    output = model(x_batch, training=True)
    loss = loss_fn(y_batch, output)

grads = tape.gradient(loss, model.trainable_variables)
for var, grad in zip(model.trainable_variables, grads):
    if grad is None:
        print(f"WARNING: None gradient for {var.name}")
    else:
        print(f"{var.name}: grad norm = {tf.norm(grad):.4f}")

# 3. Check for NaN/Inf
tf.debugging.check_numerics(tensor, "NaN/Inf detected!")

# 4. Print shapes during forward pass
@tf.function
def forward(x):
    tf.print("Input shape:", tf.shape(x))
    x = layer1(x)
    tf.print("After layer1:", tf.shape(x))
    return x

# 5. Assert tensor properties
tf.debugging.assert_shapes([(x, ('N', 784)), (y, ('N', 10))])
tf.debugging.assert_greater_equal(x, 0.0)
```

### Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `InvalidArgumentError: Input to reshape is a tensor with N values, but the requested shape has M` | Wrong reshape | Check input/output shapes |
| `NaN loss` | Too high LR, missing normalization | Lower LR, add BatchNorm, check data |
| `ResourceExhaustedError: OOM` | GPU out of memory | Reduce batch size, use mixed precision |
| `ValueError: A merge layer should be called on a list of inputs` | Wrong layer usage | Check layer input format |
| Gradient = None | Layer not in tape scope | Ensure variables are `tf.Variable` |
| Slow training | No GPU / bad pipeline | Check GPU, add `.prefetch()` |

### Best Practices Checklist

```
✅ Normalize input data (zero mean, unit variance)
✅ Use validation set to monitor overfitting
✅ Start with a small learning rate (1e-3 for Adam)
✅ Use EarlyStopping + ModelCheckpoint callbacks
✅ Use @tf.function for production training loops
✅ Profile GPU utilization — aim for >90%
✅ Use tf.data with .prefetch() and .cache()
✅ Save models in .keras or SavedModel format
✅ Log experiments to TensorBoard
✅ Version your models and datasets
✅ Set random seeds for reproducibility
✅ Use mixed precision on modern GPUs (2x speedup)
```

### Reproducibility

```python
# Set all random seeds
import random, numpy as np, tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Note: full reproducibility on GPU is not always guaranteed
# due to non-deterministic CUDA operations
```

---

## 26. Real-World Project Checklist

### End-to-End ML Pipeline

```
1. PROBLEM DEFINITION
   □ Define problem type (classification/regression/etc.)
   □ Define success metric (accuracy, F1, AUC, RMSE...)
   □ Establish baseline (random, simple model)

2. DATA
   □ Collect and label data
   □ Exploratory data analysis (EDA)
   □ Handle missing values, outliers
   □ Feature engineering
   □ Train/Val/Test split (e.g., 70/15/15)
   □ Check for data leakage

3. MODEL
   □ Start simple (Logistic Regression, shallow NN)
   □ Gradually increase complexity
   □ Use pretrained models when possible
   □ Apply appropriate regularization

4. TRAINING
   □ Set up reproducible experiments
   □ Monitor loss curves (overfitting/underfitting)
   □ Tune hyperparameters (LR, batch size, architecture)
   □ Use callbacks (EarlyStopping, ReduceLROnPlateau)
   □ Log to TensorBoard

5. EVALUATION
   □ Evaluate on held-out test set (once!)
   □ Analyze confusion matrix / error cases
   □ Check model fairness and bias

6. DEPLOYMENT
   □ Convert to TFLite (edge) or SavedModel (server)
   □ Test with TF Serving locally
   □ Set up monitoring (data drift, model degradation)
   □ A/B testing strategy
   □ Rollback plan
```

### Hyperparameter Tuning with Keras Tuner

```python
import keras_tuner as kt

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    
    # Tune number of layers
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', 32, 512, step=32),
            activation='relu'
        ))
        model.add(layers.Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))
    
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    directory='tuning',
    project_name='mnist'
)

tuner.search(x_train, y_train, epochs=20, validation_split=0.2,
             callbacks=[keras.callbacks.EarlyStopping(patience=5)])

best_model = tuner.get_best_models()[0]
best_hps    = tuner.get_best_hyperparameters()[0]
```

---

## 27. Quick Reference Cheat Sheet

### Model Architecture by Problem Type

| Problem | Input | Architecture | Output Layer |
|---|---|---|---|
| Binary Classification | Tabular | Dense → Dense | Dense(1, sigmoid) |
| Multi-class Classification | Tabular | Dense → Dense | Dense(N, softmax) |
| Image Classification | Images | CNN + Dense | Dense(N, softmax) |
| Object Detection | Images | YOLO/SSD/Faster-RCNN | Boxes + Classes |
| Image Segmentation | Images | U-Net | Pixel classes |
| Text Classification | Text | Embedding + LSTM/GRU | Dense(N, softmax) |
| Sequence-to-Sequence | Sequences | Encoder-Decoder | Token probabilities |
| Time Series Forecast | Sequences | LSTM/GRU/CNN | Dense(forecast_horizon) |
| Regression | Tabular | Dense → Dense | Dense(1, linear) |
| Anomaly Detection | Any | Autoencoder | Reconstructed input |
| Generative | Noise | GAN / VAE | Generated samples |

### Loss & Metric Cheat Sheet

```python
# Binary classification
model.compile(loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

# Multi-class (integer labels)
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Multi-class (one-hot labels)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Regression
model.compile(loss='mse', metrics=['mae', 'mse'])

# Multi-label classification
model.compile(loss='binary_crossentropy', metrics=['accuracy'])
```

### Common Tensor Shapes

```python
# Image: (batch, height, width, channels)
# Example CIFAR-10 batch: (32, 32, 32, 3)

# Sequence: (batch, timesteps, features)
# Example LSTM batch: (32, 100, 128)

# Text (embedded): (batch, sequence_length, embedding_dim)

# Dense layer input: (batch, features)
```

### The Golden Rules

```
1. 🔑 Always normalize your inputs
2. 🔑 Use validation data — always
3. 🔑 Start simple, add complexity only when needed
4. 🔑 If loss is NaN: lower LR, check data, add gradient clipping
5. 🔑 High train acc, low val acc = overfitting → more dropout/data/regularization
6. 🔑 Low train acc, low val acc = underfitting → bigger model, more epochs, lower LR
7. 🔑 Use @tf.function for production, eager for debugging
8. 🔑 tf.data.AUTOTUNE is your friend — always prefetch
9. 🔑 Save your best model, not just the last one
10. 🔑 Document your experiments — you will forget what you tried
```

---

## 📚 Resources & Next Steps

### Official Documentation
- [TensorFlow Docs](https://www.tensorflow.org/api_docs)
- [Keras Docs](https://keras.io/api/)
- [TF Tutorials](https://www.tensorflow.org/tutorials)
- [TF Guide](https://www.tensorflow.org/guide)

### Recommended Learning Path
1. ✅ Complete this guide
2. 📖 Read *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* — Aurélien Géron
3. 🧪 Build 5 projects (MNIST → CIFAR → NLP → Time Series → Deployment)
4. 📜 Read TensorFlow paper + Keras documentation
5. 🏆 Compete on Kaggle with TensorFlow
6. 🚀 Deploy a model to production using TF Serving

### Important Libraries in the TF Ecosystem
```
tensorflow-datasets    — Ready-made datasets
tensorflow-hub         — Pretrained models
keras-tuner            — Hyperparameter tuning  
tensorflow-addons      — Community extensions
tensorflow-text        — NLP preprocessing
tensorflow-probability — Probabilistic models
tensorflow-recommenders— Recommendation systems
tensorflow-federated   — Federated learning
```

---

*These notes cover TensorFlow from first principles to production deployment.*  
*Every concept links to the next. Master the fundamentals — everything else follows.*

> **"The most important thing is to keep learning, keep coding, and keep building."**

---

*Last updated: 2025 | TensorFlow 2.x | Python 3.10+*
