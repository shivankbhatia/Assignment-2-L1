# Data Sampling Techniques & Model Evaluation

_This is a submission to the (Assignment-1)[https://github.com/AnjulaMehto/Sampling_Assignment/tree/main]_

## 📋 Project Overview

This project demonstrates various **data sampling techniques** applied to an imbalanced credit card fraud detection dataset. We implement and compare multiple sampling strategies to handle class imbalance, then train 5 different machine learning models to evaluate their performance across these sampling techniques.

---

## 🎯 Objectives

1. **Understand Data Imbalance**: Identify and visualize the class distribution problem in the credit card dataset
2. **Balance the Data using SMOTE**: Used SMOTE to balance the data  and match the classes distribution.
3. **Implement Sampling Techniques**: Apply 6 different sampling methods to create balanced samples
4. **Train Multiple Models**: Build and train 5 different classification models
5. **Compare Performance**: Evaluate model accuracy across all sampling techniques

---

## 📊 Dataset

- **Source**: Credit Card Fraud Detection Dataset (`Creditcard_data.csv`)
- **Problem**: Highly imbalanced dataset with significant class imbalance
- **Classes**: 
  - Class 0: Legitimate transactions (Majority)
  - Class 1: Fraudulent transactions (Minority)
- **SMOTE**:
    Synthetically generate data to balance the dataset, here we match the number of ('0') which was earlier 9 to ('1') as 763 each.
---

## 🛠️ Sampling Techniques Implemented

### 1. **Original Data**
- The raw, unsampled dataset with its natural class imbalance
- Baseline for comparison

### 2. **Simple Random Sampling**
- **Method**: Randomly select samples without replacement
- **Sample Size**: 100 records
- **Use Case**: Quick prototyping, exploratory analysis
- **Pros**: Simple and unbiased
- **Cons**: May not preserve class distribution

### 3. **Systematic Sampling**
- **Method**: Select every kth sample at regular intervals
- **Use Case**: When data has no inherent ordering
- **Pros**: Ensures spread across entire dataset
- **Cons**: Can introduce bias if data has patterns

### 4. **Stratified Sampling**
- **Method**: Sample proportionally from each class (50% from each stratum)
- **Use Case**: **BEST for imbalanced data** - maintains class distribution
- **Pros**: Preserves class ratios, reduces bias
- **Cons**: Requires knowledge of strata

### 5. **Cluster Sampling**
- **Method**: Divide data into 10 clusters using K-Means, then select 3 random clusters
- **Use Case**: When data naturally groups together
- **Pros**: Reduces data collection costs
- **Cons**: May introduce within-cluster bias

### 6. **Bootstrap Sampling**
- **Method**: Random sampling with replacement
- **Sample Size**: 100 records
- **Use Case**: Confidence interval estimation, ensemble methods
- **Pros**: Allows same record multiple times, good for variance estimation
- **Cons**: May reduce effective sample size
---

## 🤖 Machine Learning Models

Five different classification models are trained on each sampling technique:

1. **Logistic Regression** 
2. **Random Forest** 
3. **Support Vector Machine (SVM)** 
4. **K-Nearest Neighbors (KNN)**
5. **Decision Tree** 

---

## 📈 Evaluation Metric

- **Primary Metric**: **Accuracy**
- **Definition**: Percentage of correct predictions out of total predictions
- **Formula**: `(TP + TN) / (TP + TN + FP + FN)`

> **Note**: For imbalanced datasets, accuracy alone may be misleading. Additional metrics (precision, recall, F1-score) should be considered in production systems. It is better to use Recall and Precision than Accuracy, but I have sticked to the instructions as mentioned in the assignment.

---

## 🚀 How to Use

### Prerequisites
```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

### Running the Notebook
1. Open `main.ipynb` in Jupyter Notebook or VS Code
2. Run cells sequentially.

---

## 📊 Results Summary

The results are stored in [results.csv].
| Model | Original Data | Random Sampling | Systematic Sampling | Stratified Sampling | Cluster Sampling | Bootstrap Sampling | SMOTE Oversampling |
|--------|-------------|----------------|---------------------|---------------------|-----------------|--------------------|--------------------|
| Logistic Regression | 0.9871 | 0.8500 | 0.7619 | 0.9150 | 0.9496 | 0.9500 | 0.9085 |
| Random Forest | 0.9871 | 1.0000 | 0.8571 | 0.9935 | 1.0000 | 1.0000 | 0.9902 |
| Support Vector Machine | 0.9871 | 0.9000 | 0.9524 | 0.9869 | 0.9784 | 0.9500 | 0.9804 |
| K-Nearest Neighbors | 0.9871 | 0.9000 | 0.6667 | 0.8889 | 0.9353 | 0.9000 | 0.9379 |
| Decision Tree | 0.9677 | 0.8500 | 0.7619 | 0.9608 | 0.9928 | 0.8000 | 0.9706 |

---

## 💡 Key Recommendations

### For Imbalanced Classification:
1. **Use SMOTE Oversampling** ✓
   - Preserves original data integrity
   - Creates realistic synthetic minority samples
   - Prevents information loss

2. **Use Stratified Sampling** ✓
   - Maintains class distribution in samples
   - Simple and interpretable
   - Good baseline approach

3. **Avoid Simple Random Sampling** ✗
   - For imbalanced data, likely to create even more imbalance
   - May miss minority class in small samples

### Model Selection:
- **Random Forest**: Often performs well without tuning
- **Logistic Regression**: Interpretable, good baseline
- **SVM**: Effective for binary classification
- **KNN**: Sensitive to feature scaling (handled in code)
- **Decision Tree**: Fast inference, interpretable

### Important Considerations:
⚠️ **Always validate on original unsampled test data** - Sampled data results may not reflect real-world performance

---

## 📚 Learning Outcomes

After completing this project, you will understand:
- ✓ What class imbalance is and why it matters
- ✓ How different sampling techniques work
- ✓ Pros and cons of each sampling approach
- ✓ How to train and evaluate ML models
- ✓ How to compare model performance across techniques
- ✓ Best practices for handling imbalanced datasets

---

## 📖 References

- **Imbalanced Learning**: SMOTE technique (Chawla et al., 2002)
- **Sampling Methods**: Stratified vs Random sampling trade-offs
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score
- **Scikit-learn Documentation**: [https://scikit-learn.org](https://scikit-learn.org)
- **Imbalanced-learn**: [https://imbalanced-learn.org](https://imbalanced-learn.org)

---

## 👨‍💻 Author & Acknowledgments

**Course**: Data Science - Lecture Work  
**Semester**: 6th  
**Topic**: Data Sampling & Class Imbalance Handling

---

## 📝 License

This project is for educational purposes.

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2026 | Initial implementation with 7 sampling techniques and 5 models |

---

**Last Updated**: February 3, 2026