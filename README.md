# 🎯 Real-Time Object Classification (ResNet18)

This project implements a real-time object classification system using a webcam and a deep learning model based on ResNet18.

## 🚀 Features

* Real-time object detection using webcam (OpenCV)
* Deep learning model (ResNet18 - PyTorch)
* Confidence score display
* Unknown object detection using entropy
* Smooth predictions using frame averaging
* Training visualization (loss & accuracy graphs)

## 🧠 Model

* Pretrained ResNet18 (transfer learning)
* Fine-tuning on custom dataset
* Data augmentation:

  * Rotation
  * Flip
  * Color jitter

## 📂 Dataset

Dataset is organized using ImageFolder format:
dataset/
├── class1/
├── class2/
├── class3/

## ⚙️ Installation

```bash
pip install torch torchvision opencv-python matplotlib
```

## ▶️ Training

```bash
python train.py
```

## 🎥 Testing (Camera)

```bash
python test.py
```

## 📊 Output

* model.pth → trained model
* training_results.png → training graphs

## 💡 Technologies

* Python
* PyTorch
* OpenCV
* NumPy

## 📌 Author

GTARI Aymen 
Electrical Engineer specialized in Embedded Systems
