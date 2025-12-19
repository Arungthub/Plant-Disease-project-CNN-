# **Plant Disease Detection Using Convolutional Neural Networks**

## **1. Introduction**

Agriculture plays a vital role in the economy, and plant diseases significantly affect crop yield and quality. Early detection of plant diseases helps farmers take timely preventive measures and reduce losses. With recent advancements in Artificial Intelligence, especially in **Neural Networks and Deep Learning**, image-based disease detection has become an effective solution.

This project titled **“Plant Disease Detection System”** is an **academic deep learning project** that uses a **Convolutional Neural Network (CNN)** to automatically identify plant diseases from leaf images. The system classifies plant type and disease and provides suitable treatment recommendations.

This project is developed purely for **academic and educational purposes** as part of coursework in **Neural Networks / Deep Learning**.

---

## **2. Objective of the Project**

The main objectives of this project are:

* To design and implement a **Convolutional Neural Network (CNN)** for image classification.
* To identify plant diseases from leaf images.
* To classify multiple plant types such as **Tomato, Potato, and Pepper**.
* To provide disease-specific solutions based on model predictions.
* To deploy the trained model as a simple web application for demonstration purposes.

---

## **3. Technology Stack Used**

* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow & Keras
* **Neural Network Type:** Convolutional Neural Network (CNN)
* **Web Framework:** Streamlit
* **Image Processing:** PIL, NumPy
* **Visualization:** Matplotlib
* **Development Tool:** Visual Studio Code

---

## **4. Dataset Description**

The project uses the **PlantVillage dataset**, which is a publicly available and widely used dataset for plant disease classification research.

### **Dataset Structure**

The dataset is organized into three folders:

```
PlantVillage_Split/
├── train/
├── validation/
└── test/
```

Each folder contains subfolders representing different plant disease classes.

### **Dataset Path Used in the Project**

```text

```

Each subfolder contains leaf images belonging to a specific disease category, such as:

* Tomato_Late_blight
* Tomato_Early_blight
* Tomato_Target_Spot
* Potato_Late_blight
* Pepper_bell_Bacterial_spot
* Healthy plant classes

This directory-based structure allows the CNN model to automatically learn class labels from folder names.

---

## **5. Methodology**

### **5.1 Data Preprocessing**

* Images are resized to **64×64 pixels**
* Pixel values are normalized between **0 and 1**
* Data augmentation techniques such as rotation, zoom, and flipping are applied to improve generalization

### **5.2 Model Architecture**

A **Convolutional Neural Network (CNN)** is designed with the following layers:

* Convolution layers for feature extraction
* Max pooling layers for dimensionality reduction
* Fully connected (Dense) layers for classification
* Softmax activation for multi-class output

The model is trained using the **Adam optimizer** and **categorical cross-entropy loss**.

---

## **6. Model Training and Evaluation**

* The dataset is split into training, validation, and test sets
* Model performance is evaluated using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix
* Training and validation accuracy and loss graphs are generated to analyze learning behavior

The trained model is saved as:

```
plant_disease_model.h5
class_names.npy
```

## **Terminal Code: **
```
cd Plant-Disease-project-CNN
python dataset_download.py
python train.py
streamlit run app.py
```
---

## **7. Deployment**

For demonstration purposes, the trained CNN model is deployed using a **Streamlit web application**. The application allows users to:

* Upload a plant leaf image
* Automatically detect plant type
* Identify the disease
* Display confidence score
* Suggest appropriate treatment solutions

This deployment is intended **only for academic demonstration** and not for commercial or medical use.

---

## **8. Results and Discussion**

The system achieves **reasonable classification accuracy**. However, some misclassifications occur due to:

* Visual similarity between plant diseases
* Low-quality or unclear images
* Limited training epochs

Such limitations are common in real-world deep learning applications and are acknowledged as part of this academic study.

---

## **9. Limitations**

* Performance depends on image quality
* Similar disease patterns can cause confusion
* The model is trained on a limited dataset
* Not intended for real agricultural decision-making

---

## **10. Future Scope**

This project can be extended in the future by:

* Training on larger and more diverse datasets
* Using advanced architectures like **ResNet or MobileNet**
* Improving image resolution
* Adding mobile application support
* Integrating real-time camera input
* Deploying the system on cloud platforms

---

## **11. Conclusion**

This project successfully demonstrates the application of **Neural Networks and Deep Learning** in solving real-world problems. A CNN-based model was developed to detect plant diseases from leaf images and deployed as a simple web application. The project fulfills academic objectives by combining theory with practical implementation.

---

## **12. Academic Declaration**

> This project is developed strictly for **academic and educational purposes** as part of coursework in **Neural Networks and Deep Learning**. The system is not intended for commercial or professional agricultural diagnosis.

---
