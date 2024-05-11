# Kidney CT Dataset Classification with VGG16

![Front page](https://github.com/nahidkawsar/Performance-Visualization-of-VGG16-based-Model-for-CT-Kidney-Dataset/assets/149723828/7d57fa8f-8fce-4679-9cbe-934dd61053d1)


## Introduction
This project aims to classify kidney CT images into four categories: Normal, Cyst, Tumor, and Stone. We'll employ a pre-trained VGG16 model for this task.

## Dataset
The dataset used for this project can be found on Kaggle [here](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone). It consists of kidney CT images categorized into Normal, Cyst, Tumor, and Stone classes.

## Process
1. **Data Preparation:**
   - Created a Kaggle API token and downloaded the dataset.
   - Extracted the dataset and organized it into appropriate directories.
   - Loaded the dataset using TensorFlow's `image_dataset_from_directory` function, splitting it into training and validation sets.

2. **Model Building:**
   - Imported necessary libraries and pre-trained VGG16 model without the top dense layers.
   - Built a Sequential model with VGG16 base, followed by Flatten and Dense layers for classification.
   - Froze the convolutional base layers to prevent their weights from being updated during training.

3. **Data Augmentation and Normalization:**
   - Used TensorFlow's `ImageDataGenerator` for data augmentation, although it's commented out in the code.
   - Normalized the pixel values to the range [0, 1].

4. **Model Compilation and Training:**
   - Compiled the model with Adam optimizer and sparse categorical crossentropy loss function.
   - Trained the model for 10 epochs using the training and validation datasets.

5. **Evaluation and Visualization:**
   - Visualized the training and validation accuracy and loss using Matplotlib.
   - Saved the trained model as "Model.h5".

## Model Performance Visualization

### Training Accuracy validation Accuracy

![Training and validation accuracy](https://github.com/nahidkawsar/Performance-Visualization-of-VGG16-based-Model-for-CT-Kidney-Dataset/assets/149723828/af0dbd6f-d7ff-4ad2-bb03-14535723745a)

### Training and Validation Loss

![Training and validation loss](https://github.com/nahidkawsar/Performance-Visualization-of-VGG16-based-Model-for-CT-Kidney-Dataset/assets/149723828/d18657bd-0105-49ec-ab4b-e19ffbd66bdb)

## Conclusion
The model achieved decent accuracy in classifying kidney CT images into Normal, Cyst, Tumor, and Stone categories. Further optimization and fine-tuning can be explored for better performance. 

## GitHub Repository
You can find the GitHub repository containing the code and relevant files [here](https://github.com/nahidkawsar/Performance-Visualization-of-VGG16-based-Model-for-CT-Kidney-Dataset/blob/main/Performance_Visualization_of_VGG16_based_Model_for_CT_Kidney_Dataset.ipynb.)

## Find me in linked:
[H.M Nahid Kawsar](linkedin.com/in/h-m-nahid-kawsar-232a86266)




