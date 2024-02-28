# Diabetes Prediction Project

<img src="https://github.com/sakethyalamanchili/Diabetes-Prediction-Project/blob/main/diabetes_icon.webp" alt="Diabetes Prediction" width="800"/>

*Image Source: [Unsplash](https://unsplash.com/photos/a-woman-holding-a-pen-and-a-cell-phone-o1N2-pSIhbw)*

## Introduction

Welcome to the Diabetes Prediction Project! This project aims to help predict if a person might have diabetes by looking at their health information. The project is managed by **Saketh Yalamanchili**, who is learning about machine learning and how it can be used to help people.

## Dataset

The dataset used in this project contains diagnostic measurements from female individuals of Pima Indian heritage, aged 21 years or older. It comprises essential features such as pregnancies, glucose concentration, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age. My thorough exploration of the dataset revealed missing data represented as zero values in certain features. To ensure data integrity, I meticulously preprocessed the dataset by replacing these zero values with the mean of each respective feature. https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data

## Model Selection and Evaluation

To identify the most effective predictive model, I evaluated the performance of seven machine learning algorithms:

- **Logistic Regression**
- **KNeighbors Classifier**
- **Random Forest Classifier**
- **Support Vector Classifier (SVC)**
- **Gaussian Naive Bayes**
- **XGBoost**
- **CatBoost**

Through rigorous model selection and evaluation, I conducted baseline modeling and hyperparameter tuning to optimize model performance. CatBoost emerged as the top-performing model, achieving an impressive accuracy of **0.82** on both scaled and unscaled data.

## Model Comparison

![Accuracy Comparison](https://github.com/sakethyalamanchili/Diabetes-Prediction-Project/blob/main/accuracies_img.png)

The bar plot above illustrates the comparison of accuracy scores obtained from different machine learning models. CatBoost outperformed other models, demonstrating its superior predictive capability.

## Web Application Deployment

To make our predictive model accessible to a wider audience, I developed a user-friendly web application using Streamlit. The application allows users to input their health parameters and receive instant predictions regarding their diabetes status. With its intuitive interface and real-time feedback, this web application empowers individuals to take proactive steps towards better health management.

## Conclusion

The Diabetes Prediction Project showcases the transformative potential of data-driven healthcare solutions. By harnessing the power of machine learning and deploying user-friendly applications, I aim to revolutionize diabetes risk assessment and early detection. Together, we can pave the way for a healthier future.
