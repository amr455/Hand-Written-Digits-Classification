# Hand-Written-Digits-Classification

# **Digits Classification Using Neural Networks**

## 🚀 Project Highlights

I'm excited to share a recent project where I built and trained a neural network model to classify handwritten digits using the MNIST dataset. This project involved several key steps:

### ⭐ **Data Preprocessing**

- Loaded the MNIST dataset, which consists of 60,000 training images and 10,000 test images.
- Normalized and flattened the images to prepare them for model training.

### ⭐ **Model Building**

- Developed a sequential neural network model using TensorFlow and Keras.
- The model includes:
  - A dense layer with ReLU activation and L2 regularization.
  - A dropout layer for regularization.
  - A final softmax layer for classification.

### ⭐ **Model Compilation**

- Used the Adam optimizer.
- Sparse categorical crossentropy as the loss function.
- Accuracy as the evaluation metric.

### ⭐ **Visualization**

- Visualized sample images from the dataset and their corresponding labels to understand the data better before training the model.

### ⭐ **Performance Visualization**

- Plotted training and validation loss and accuracy over epochs.
- The improved model demonstrated better convergence and more stable performance compared to the initial configuration.

---

## 📈 **Key Takeaways**

### ⭐ **Model Improvement with Increased Neurons**

To enhance the model, I expanded its architecture:

- Increased the first hidden layer to **128 neurons** (previously 64) with ReLU activation.
- Added a second hidden layer with **64 neurons** to capture more complex patterns in the data.

### ⭐ **Impact of Neuron Expansion**

- Increasing the number of neurons in the hidden layers provided the model with greater capacity to learn from the data.
- The enhanced model achieved a **higher accuracy of 96.97%** on the test set, with a **slightly improved loss of 0.1867**.

---

## 📌 **Technologies Used**

- Python 🐍
- TensorFlow & Keras 🤖
- NumPy & Pandas 📊
- Matplotlib & Seaborn 📉

## 📂 **Project Structure**

```bash
📦 Digits-Classification-NN
├── 📜 model.py  # Neural Network Implementation
├── 📜 README.md  # Project Documentation
```



