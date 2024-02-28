# Diabetes Prediction Project

![Diabetes Prediction](https://github.com/sakethyalamanchili/Diabetes-Prediction-Project/blob/main/diabetes_icon.webp)

*Image Source: [Unsplash](https://unsplash.com/photos/a-woman-holding-a-pen-and-a-cell-phone-o1N2-pSIhbw)*

## Introduction

Welcome to the Diabetes Prediction Project! This project aims to help predict if a person might have diabetes by looking at their health information. The project is managed by **Saketh Yalamanchili**, who is learning about machine learning and how it can be used to help people.

## Dataset

The dataset used in this project contains diagnostic measurements from female individuals of Pima Indian heritage, aged 21 years or older. It comprises essential features such as pregnancies, glucose concentration, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age. My thorough exploration of the dataset revealed missing data represented as zero values in certain features. To ensure data integrity, I meticulously preprocessed the dataset by replacing these zero values with the mean of each respective feature. [Dataset Source](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)

## Model Selection and Evaluation

To identify the most effective predictive model, I evaluated the performance of seven machine learning algorithms:

- **Logistic Regression**
- **KNeighbors Classifier**
- **Random Forest Classifier**
- **Support Vector Classifier (SVC)**
- **Gaussian Naive Bayes**
- **XGBoost**
- **CatBoost**

Through rigorous model selection and evaluation, I conducted baseline modeling and hyperparameter tuning to optimize model performance. CatBoost emerged as the top-performing model, achieving an impressive accuracy of **82%** on both scaled and unscaled data.

## Model Comparison

![Accuracy Comparison](https://github.com/sakethyalamanchili/Diabetes-Prediction-Project/blob/main/accuracies_img.png)

The bar plot above illustrates the comparison of accuracy scores obtained from different machine learning models. CatBoost outperformed other models, demonstrating its superior predictive capability.

## Web Application Deployment

To make our predictive model accessible to a wider audience, I developed a user-friendly web application using Streamlit. The application allows users to input their health parameters and receive instant predictions regarding their diabetes status. With its intuitive interface and real-time feedback, this web application empowers individuals to take proactive steps towards better health management.

**Public Deployment Link:** [Diabetes Predictor Web App](https://saketh-diabetes-predictor.streamlit.app/)

**Interface Screenshots:**

![Screenshot 1](https://github.com/sakethyalamanchili/Diabetes-Prediction-Project/blob/main/1.%20PNG.png)
![Screenshot 2](https://github.com/sakethyalamanchili/Diabetes-Prediction-Project/blob/main/2.%20PNG.png)

In this web app, users need to provide all the required information. If any field is left blank, the application will display an error message, prompting the user to fill in all the fields before proceeding. This model predicts with an accuracy of 82%, which is good. By using this web app, individuals can assess their risk of diabetes and take proactive measures towards better health.

## Conclusion

The Diabetes Prediction Project showcases the transformative potential of data-driven healthcare solutions. By harnessing the power of machine learning and deploying user-friendly applications, I aim to revolutionize diabetes risk assessment and early detection. Together, we can pave the way for a healthier future.
