# Akbank Deep Learning Bootcamp: Architectural Heritage Classification
This project, developed for the Akbank Deep Learning Bootcamp, builds a robust deep learning model to classify images of ten different architectural heritage elements. The model successfully distinguishes between features like domes, columns, and gargoyles with high accuracy.

Final Kaggle Notebook: [Notebook]([https://www.kaggle.com/datasets/ikobzev/architectural-heritage-elements-image64-dataset](https://www.kaggle.com/code/eminesunaylmaz/akbk-bootcamp-architectural-heritage-project))


## Proje Notebook'u
Projeye ait tüm kod ve analizleri içeren dosyaya aşağıdaki linklerden erişebilirsiniz:

* **[GitHub üzerinde görüntüle](./akbk-bootcamp-architectural-heritage-project.ipynb)**
* **[Kaggle üzerinde interaktif olarak çalıştır](https://www.kaggle.com/code/eminesunaylmaz/akbk-bootcamp-architectural-heritage-project)**



# 1. Project Goal
The main goal of this project was to build an accurate and interpretable deep learning model for image classification. The project involved using Convolutional Neural Networks (CNNs), leveraging Transfer Learning with the VGG16 architecture, and fine tuning the model for optimal performance. A key part of the project was also model interpretability, using tools like Grad CAM to understand and visualize the model's decision making process.


# 2. Dataset
This project uses the "Architectural Heritage Elements" dataset from Kaggle. It contains approximately 10,000 training images spread across 10 distinct classes.
Dataset Link: [Architectural Heritage Elements on Kaggle](https://www.kaggle.com/datasets/ikobzev/architectural-heritage-elements-image64-dataset)


# 3. Methodology
The project followed a systematic approach, progressing from a simple baseline to a highly optimized model.

Data Preprocessing and Augmentation: The image data was prepared for training by resizing and normalizing the pixels. For the VGG16 model, a specific preprocessing function was used. Data augmentation techniques (rotation, zoom, horizontal flip) were applied to the training set to prevent overfitting and improve generalization.

Baseline Model: A simple CNN was built from scratch to establish a baseline performance. This model consisted of three convolutional layers.

Transfer Learning: The pre trained VGG16 model, with weights from ImageNet, was used as a feature extractor. Its convolutional base was frozen, and a custom classifier was added on top.

Fine Tuning: To further improve accuracy, the top layers of the VGG16 model were unfrozen and trained with a very low learning rate. This step allowed the model to adapt its learned features more closely to our specific dataset.

Model Evaluation: All models were evaluated using accuracy, a confusion matrix, and a detailed classification report (with precision, recall, and F1 score).

Model Interpretation: Grad CAM was used to visualize the model's attention, confirming that it focused on relevant features within the images to make its predictions.


# 4. Results
The project successfully demonstrated the power of transfer learning and fine tuning, with each step yielding significant improvements in accuracy. The final model achieved an excellent accuracy of 94.97% on the unseen test set.

Performance Progression:
Baseline Model (Simple CNN): Achieved a validation accuracy of around 75%. This served as a solid starting point.

Transfer Learning Model (VGG16): Significantly improved performance, reaching a validation accuracy of 91%.

Fine Tuned Model (VGG16): Reached a final test accuracy of 94.97%, showing strong generalization to new data.

Final Model Performance Highlights:
The final model showed outstanding performance, particularly for classes like stained_glass (100% F1 score) and dome(inner) (98% F1 score). The initial confusion between apse and vault was also noticeably reduced after fine tuning.

Final Confusion Matrix
Grad CAM Visualization


# 5. Technologies Used
> TensorFlow & Keras
> Scikit learn
> Pandas & NumPy
> Matplotlib & Seaborn
> Kaggle Notebooks







