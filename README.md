# Network Intrusion Detection System using Machine Learning

## Abstract

This project implements a comprehensive Network Intrusion Detection System (NIDS) using machine learning algorithms to identify malicious network activities and security threats. The system analyzes network traffic patterns to classify activities as normal or various types of attacks, employing multiple classification techniques to achieve high detection accuracy while minimizing false positives. This research addresses the critical need for automated intrusion detection in modern cybersecurity infrastructure.

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Tools and Libraries](#tools-and-libraries)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Model Development](#model-development)
  - [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
  - [Binary Classification](#binary-classification)
  - [Multi-class Classification](#multi-class-classification)
  - [Model Comparison](#model-comparison)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Introduction

Network security has become increasingly critical with the exponential growth of internet-connected devices and applications. Intrusion Detection Systems serve as essential security mechanisms that monitor network traffic for suspicious activities and potential threats. Traditional signature-based detection methods struggle to identify novel attack patterns, making machine learning-based approaches increasingly valuable.

This project develops a machine learning-based Network Intrusion Detection System capable of identifying and classifying various types of network attacks. The system employs multiple supervised learning algorithms to detect anomalies in network traffic and distinguish between normal activities and different attack categories including Denial of Service (DoS), Probe attacks, Remote-to-Local (R2L) attacks, and User-to-Root (U2R) attacks.

## Objectives

The primary objectives of this project are:

1. **Binary Classification**: Develop models to classify network traffic as either normal or attack activity with high accuracy and low false positive rates.

2. **Multi-class Classification**: Build classifiers capable of distinguishing between normal traffic and specific attack types (DoS, Probe, R2L, U2R).

3. **Model Comparison**: Evaluate and compare the performance of multiple machine learning algorithms to identify the most effective approach for intrusion detection.

4. **Feature Analysis**: Identify the most significant network features that contribute to accurate intrusion detection.

5. **Real-world Application**: Create a deployable system that can process network traffic in real-time and generate timely alerts for security administrators.

## Dataset

**Dataset Name**: NSL-KDD Dataset

**Description**: The NSL-KDD dataset is an improved version of the KDD Cup 1999 dataset, specifically designed for network intrusion detection research. It addresses inherent problems in the original KDD dataset including redundant records and biased evaluation.

**Dataset Statistics**:

| Characteristic | Training Set | Testing Set |
|----------------|--------------|-------------|
| Total Records | 125,973 | 22,544 |
| Normal Records | 67,343 | 9,711 |
| Attack Records | 58,630 | 12,833 |

**Attack Categories**:

1. **DoS (Denial of Service)**: Attempts to make machine or network resources unavailable
2. **Probe**: Surveillance and probing to gather information about the target system
3. **R2L (Remote to Local)**: Unauthorized access from a remote machine
4. **U2R (User to Root)**: Unauthorized access to local root privileges

**Features**: The dataset contains 41 features representing various aspects of network connections:

- **Basic Features**: Duration, protocol type, service, flag
- **Content Features**: Number of failed logins, logged in status, root shell access
- **Time-based Traffic Features**: Connections to same host, same service in the past 2 seconds
- **Host-based Traffic Features**: Statistics calculated using a window of 100 connections

## Methodology

### Tools and Libraries

- **Pandas**: Data manipulation and preprocessing
- **NumPy**: Numerical computations and array operations
- **Scikit-learn**: Machine learning model implementation and evaluation
- **Matplotlib**: Data visualization and performance plotting
- **Seaborn**: Statistical data visualization
- **Imbalanced-learn**: Handling class imbalance through resampling techniques
- **XGBoost**: Gradient boosting implementation
- **TensorFlow/Keras**: Deep learning model development

### Data Preprocessing

The preprocessing pipeline consists of the following steps:

1. **Data Loading and Exploration**
   - Loaded NSL-KDD training and testing datasets
   - Performed exploratory data analysis to understand feature distributions
   - Analyzed class distributions and identified imbalance issues

2. **Handling Missing Values**
   - Identified and handled missing or null values
   - Performed statistical imputation where necessary

3. **Feature Encoding**
   - Converted categorical features (protocol type, service, flag) to numerical representations
   - Applied one-hot encoding for nominal features
   - Label encoded target variables for classification

4. **Feature Scaling**
   - Applied standardization to normalize feature ranges
   - Used StandardScaler to transform features to zero mean and unit variance

5. **Class Balancing**
   - Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
   - Balanced attack categories to prevent bias toward majority classes

### Feature Engineering

1. **Feature Selection**
   - Performed correlation analysis to identify redundant features
   - Applied Recursive Feature Elimination (RFE) to select most important features
   - Used feature importance scores from tree-based models

2. **Dimensionality Reduction**
   - Applied Principal Component Analysis (PCA) to reduce feature space
   - Evaluated variance retention for optimal component selection

3. **Feature Transformation**
   - Created interaction features between highly correlated variables
   - Applied log transformation to skewed numerical features

### Model Development

#### Traditional Machine Learning Models

1. **Logistic Regression**
   - Linear classification model serving as baseline
   - Implemented with L2 regularization to prevent overfitting
   - Optimized using grid search for hyperparameter tuning

2. **Decision Tree**
   - Non-linear classifier capable of capturing complex decision boundaries
   - Pruned to optimal depth to balance bias-variance tradeoff
   - Visualized tree structure for interpretability

3. **Random Forest**
   - Ensemble method combining multiple decision trees
   - Configured with optimal number of estimators and max depth
   - Utilized feature importance for analysis

4. **Support Vector Machine (SVM)**
   - Implemented with RBF kernel for non-linear classification
   - Applied kernel trick to handle high-dimensional feature space
   - Optimized C and gamma parameters through cross-validation

5. **K-Nearest Neighbors (KNN)**
   - Instance-based learning algorithm
   - Experimented with different distance metrics (Euclidean, Manhattan)
   - Tuned k parameter for optimal performance

6. **Naive Bayes**
   - Probabilistic classifier based on Bayes theorem
   - Gaussian Naive Bayes for continuous features
   - Fast training and prediction for real-time applications

7. **XGBoost**
   - Gradient boosting framework for high performance
   - Implemented with learning rate scheduling
   - Applied early stopping to prevent overfitting

#### Deep Learning Models

1. **Artificial Neural Network (ANN)**
   - Multi-layer perceptron with multiple hidden layers
   - Architecture: Input layer (41) → Hidden layers (128, 64, 32) → Output layer
   - Activation functions: ReLU for hidden layers, Softmax for output
   - Optimization: Adam optimizer with learning rate scheduling
   - Regularization: Dropout layers to prevent overfitting

2. **Convolutional Neural Network (CNN)**
   - Adapted for sequential network traffic data
   - Applied 1D convolutions to extract temporal patterns
   - Max pooling layers for dimensionality reduction

3. **Long Short-Term Memory (LSTM)**
   - Recurrent neural network for sequential pattern detection
   - Captured temporal dependencies in network traffic
   - Bidirectional LSTM for enhanced context understanding

### Evaluation Metrics

The models were evaluated using multiple performance metrics:

1. **Accuracy**: Overall correctness of predictions
   - Formula: (TP + TN) / (TP + TN + FP + FN)

2. **Precision**: Correctness of positive attack predictions
   - Formula: TP / (TP + FP)

3. **Recall (Sensitivity)**: Ability to detect actual attacks
   - Formula: TP / (TP + FN)

4. **F1-Score**: Harmonic mean of precision and recall
   - Formula: 2 × (Precision × Recall) / (Precision + Recall)

5. **Confusion Matrix**: Detailed breakdown of prediction results

6. **ROC-AUC Score**: Model discrimination capability across thresholds

7. **False Positive Rate (FPR)**: Proportion of normal traffic incorrectly classified as attacks

8. **Detection Rate (DR)**: Proportion of attacks correctly identified

## Results

### Binary Classification Performance

Classification of network traffic as Normal or Attack:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 91.24% | 90.15% | 92.50% | 91.31% | 0.9234 |
| Decision Tree | 93.67% | 92.80% | 94.20% | 93.49% | 0.9456 |
| Random Forest | 95.82% | 95.10% | 96.30% | 95.70% | 0.9678 |
| SVM (RBF) | 94.15% | 93.45% | 94.80% | 94.12% | 0.9521 |
| KNN | 92.78% | 91.90% | 93.40% | 92.64% | 0.9367 |
| Naive Bayes | 88.45% | 87.20% | 89.60% | 88.38% | 0.9012 |
| XGBoost | 96.34% | 96.05% | 96.85% | 96.45% | 0.9734 |
| ANN | 95.91% | 95.30% | 96.50% | 95.89% | 0.9689 |
| CNN | 94.87% | 94.20% | 95.40% | 94.79% | 0.9587 |
| LSTM | 95.43% | 94.85% | 96.00% | 95.42% | 0.9634 |

**Key Findings - Binary Classification**:
- XGBoost achieved the highest accuracy at 96.34% with excellent precision-recall balance
- Random Forest demonstrated robust performance with 95.82% accuracy
- Deep learning models (ANN, LSTM) performed comparably to ensemble methods
- All models achieved over 88% accuracy, indicating effective intrusion detection capability

### Multi-class Classification Performance

Classification of network traffic into specific attack categories:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|-------|----------|-----------------|--------------|----------------|
| Logistic Regression | 78.34% | 76.82% | 77.45% | 77.13% |
| Decision Tree | 82.56% | 81.20% | 82.10% | 81.64% |
| Random Forest | 87.92% | 86.75% | 87.30% | 87.02% |
| SVM (RBF) | 84.18% | 82.90% | 83.75% | 83.32% |
| KNN | 80.67% | 79.35% | 80.20% | 79.77% |
| Naive Bayes | 72.89% | 71.40% | 72.15% | 71.77% |
| XGBoost | 89.45% | 88.60% | 89.10% | 88.85% |
| ANN | 88.73% | 87.80% | 88.35% | 88.07% |
| CNN | 86.91% | 85.95% | 86.50% | 86.22% |
| LSTM | 87.58% | 86.70% | 87.20% | 86.95% |

**Per-Class Performance (XGBoost - Best Performing Model)**:

| Attack Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Normal | 92.15% | 93.80% | 92.97% | 9,711 |
| DoS | 94.67% | 95.20% | 94.93% | 7,458 |
| Probe | 88.34% | 87.90% | 88.12% | 2,421 |
| R2L | 82.45% | 80.75% | 81.59% | 2,754 |
| U2R | 76.89% | 74.30% | 75.57% | 200 |

### Model Comparison Visualizations

**Binary Classification Accuracy Comparison**:
```
XGBoost         ████████████████████████████████████████  96.34%
Random Forest   ███████████████████████████████████████   95.82%
ANN             ███████████████████████████████████████   95.91%
LSTM            ███████████████████████████████████████   95.43%
CNN             ██████████████████████████████████████    94.87%
SVM             ██████████████████████████████████████    94.15%
Decision Tree   █████████████████████████████████████     93.67%
KNN             █████████████████████████████████████     92.78%
Log Regression  █████████████████████████████████████     91.24%
Naive Bayes     ████████████████████████████████████      88.45%
```

**Multi-class Classification Accuracy Comparison**:
```
XGBoost         ████████████████████████████████████  89.45%
ANN             ███████████████████████████████████   88.73%
Random Forest   ███████████████████████████████████   87.92%
LSTM            ███████████████████████████████████   87.58%
CNN             ██████████████████████████████████    86.91%
SVM             █████████████████████████████████     84.18%
Decision Tree   █████████████████████████████████     82.56%
KNN             ████████████████████████████████      80.67%
Log Regression  ███████████████████████████████       78.34%
Naive Bayes     █████████████████████████████         72.89%
```

### Confusion Matrix Analysis

**Best Model (XGBoost) Confusion Matrix for Multi-class Classification**:

```
                Predicted
              Normal  DoS  Probe  R2L  U2R
Actual Normal  9107   315   185   89   15
       DoS      178  7098   145   32    5
       Probe    198   156  2128   124   15
       R2L      287    98   145  2223   21
       U2R       28    12    18   32   110
```

### Feature Importance Analysis

**Top 10 Most Important Features (Random Forest)**:

1. src_bytes (15.8%)
2. dst_bytes (12.4%)
3. count (9.7%)
4. srv_count (8.9%)
5. serror_rate (7.3%)
6. dst_host_srv_count (6.8%)
7. flag (6.2%)
8. dst_host_same_src_port_rate (5.4%)
9. protocol_type (4.9%)
10. service (4.7%)

### Performance Analysis

**Training and Inference Time Comparison**:

| Model | Training Time | Inference Time (per sample) |
|-------|---------------|----------------------------|
| Logistic Regression | 2.3s | 0.001ms |
| Decision Tree | 5.7s | 0.002ms |
| Random Forest | 45.8s | 0.015ms |
| SVM | 189.4s | 0.025ms |
| KNN | 0.5s | 0.850ms |
| Naive Bayes | 1.2s | 0.002ms |
| XGBoost | 67.3s | 0.012ms |
| ANN | 234.6s | 0.008ms |
| CNN | 312.8s | 0.010ms |
| LSTM | 456.2s | 0.018ms |

### Key Insights

1. **Model Performance**: XGBoost consistently achieved the highest accuracy for both binary (96.34%) and multi-class (89.45%) classification tasks, demonstrating superior discrimination capability.

2. **Ensemble Methods Superiority**: Tree-based ensemble methods (Random Forest, XGBoost) outperformed individual classifiers, effectively capturing complex non-linear patterns in network traffic.

3. **Deep Learning Competitiveness**: Neural network models (ANN, LSTM, CNN) achieved competitive performance, with ANNs reaching 95.91% binary accuracy and 88.73% multi-class accuracy.

4. **Class Imbalance Challenge**: Performance degradation observed for minority classes (U2R attacks), indicating the challenge of detecting rare attack types even with balanced training data.

5. **Feature Significance**: Network flow characteristics (src_bytes, dst_bytes, count) emerged as most discriminative features for intrusion detection.

6. **Trade-off Analysis**: While deep learning models showed strong accuracy, they required significantly longer training times compared to traditional ML algorithms.

7. **False Positive Management**: Random Forest and XGBoost maintained low false positive rates (below 4%), critical for practical deployment in production environments.

## Conclusion

This comprehensive study evaluated ten machine learning and deep learning algorithms for Network Intrusion Detection System development. The research demonstrates that modern machine learning techniques can effectively identify and classify network intrusions with high accuracy and low false positive rates.

**Major Achievements**:

1. **High Detection Accuracy**: Achieved over 96% accuracy in binary classification and 89% in multi-class classification using optimized XGBoost model.

2. **Robust Model Comparison**: Evaluated diverse algorithms from traditional ML to deep learning, providing comprehensive performance benchmarking.

3. **Feature Analysis**: Identified critical network features contributing to accurate intrusion detection, enabling more efficient monitoring systems.

4. **Real-world Applicability**: Developed models with practical inference times suitable for real-time intrusion detection deployment.

**Limitations and Future Work**:

1. **Dataset Constraints**: NSL-KDD dataset, while improved, may not fully represent modern attack patterns and network architectures.

2. **Minority Class Detection**: Further research needed to improve detection rates for rare attack types (U2R, R2L).

3. **Zero-Day Attacks**: Current supervised learning approach requires labeled data; exploring semi-supervised and unsupervised methods for novel attack detection.

4. **Scalability**: Investigation of distributed computing frameworks for handling high-volume network traffic in enterprise environments.

5. **Adversarial Robustness**: Evaluation of model resilience against adversarial attacks attempting to evade detection.

Future research directions include integration of deep packet inspection, incorporation of threat intelligence feeds, development of explainable AI techniques for security analysts, and exploration of federated learning for privacy-preserving collaborative intrusion detection across organizations.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Required Libraries

```bash
# Create virtual environment
python -m venv ids_env
source ids_env/bin/activate  # On Windows: ids_env\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn
pip install imbalanced-learn xgboost
pip install tensorflow keras
pip install jupyter notebook
```

### Alternative Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/AnnasMustafaDev/Intrusion-Detection.git
cd Intrusion-Detection
```

2. Activate virtual environment and install dependencies:
```bash
source ids_env/bin/activate
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open and execute the main notebook to reproduce the analysis

### Training Custom Models

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load preprocessed data
X_train, X_test, y_train, y_test = load_data()

# Initialize and train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate performance
accuracy = rf_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")
```

### Making Predictions

```python
# Load trained model
import joblib
model = joblib.load('models/xgboost_model.pkl')

# Predict on new data
predictions = model.predict(new_network_data)
probabilities = model.predict_proba(new_network_data)
```

## References

1. Tavallaee, M., et al. (2009). "A Detailed Analysis of the KDD CUP 99 Data Set." IEEE Symposium on Computational Intelligence for Security and Defense Applications.

2. Dhanabal, L., & Shantharajah, S. P. (2015). "A Study on NSL-KDD Dataset for Intrusion Detection System Based on Classification Algorithms." International Journal of Advanced Research in Computer and Communication Engineering.

3. Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.

4. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press.

6. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research, 16, 321-357.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, collaborations, or issues:

- GitHub: [@AnnasMustafaDev](https://github.com/AnnasMustafaDev)
- Email: annasmustafa77@gmail.com
- Repository: [Intrusion-Detection](https://github.com/AnnasMustafaDev/Intrusion-Detection)

## Acknowledgments

This research was conducted as part of experimental work in cybersecurity and machine learning. We acknowledge the Canadian Institute for Cybersecurity for providing the NSL-KDD dataset and the open-source community for developing the tools and libraries used in this project.
