# 🎯 Acceleration-Based User Authentication System

This repository contains the implementation, and results for an acceleration-based user authentication system using neural networks and optimization techniques. The system leverages motion data from wearable devices to distinguish between legitimate users and imposters, with applications in secure authentication systems.

## 🌟 Final Report

### 📢 *Extended version of the report is available* 🛑 [**CLICK HERE TO VIEW**](./Extended%20Version%20of%20AI%20&%20ML%20Report%20_%20GROUP%20-%2034.pdf) 🛑
</br>

## 🌟 Project Overview

This project explores a novel method for continuous user authentication using motion data collected from wearable devices like smartwatches. The primary goals include:

1. Developing a robust machine learning model for user authentication.
2. Optimizing model performance using advanced feature selection and genetic algorithms.
3. Evaluating system performance with metrics like accuracy, precision, recall, FAR, FRR, and EER.

## 🌟 Data Description

The data consists of accelerometer readings collected over two separate days from users. Features are extracted in both time and frequency domains:

- **Time Domain:** 88 statistical features (e.g., mean, standard deviation).
- **Frequency Domain:** 43 features generated via Fast Fourier Transform (FFT).

## 🌟 Key Features

- **Neural Networks:** Designed and tuned for binary classification with optimized architectures.
- **Optimization Techniques:** Leveraging ANOVA, Mutual Information, Steepest Gradient, and Genetic Algorithms.
- **Evaluation Metrics:** FAR, FRR, EER, precision, recall, and accuracy are utilized for performance assessment.

## 🌟 Implementation

### 🔴 Testing and Validation

**1. Data Splitting:**

- Experiments with different training/testing combinations (e.g., day-wise splitting).
- Optimal ratio of legitimate to imposter samples found to be 1:5.

![Train Ratio benchmarking & Dataset Contribution for authentication](G3%20All%20Other%20Figures/Train%20Test%20Ratio%20selection%20along%20with%20domain%20combinations/ratio_featureSet_benchmarks.svg)

**2. Cross-validation:**
  - Leave-One-User-Out (LOUO) for generalization.
  - DTW (Dynamic Time Warping) to select Leave-out user: Measures the distance between temporal sequences with varying lengths or speeds.

**3. Cosine Similarity Analysis**

Cosine similarity was employed to identify similarities and anomalies in walking patterns. For example:

- **User 7:** Displayed identical samples across days, flagged as a potential anomaly.

![Cosine Similarity Anomaly for User 7](G3%20All%20Other%20Figures/cosine%20simillarities%20for%20day%20wise%20samples/u7.jpg)

### 🔴 Dimensionality Reduction and PCA

Principal Component Analysis (PCA) was used to reduce dataset complexity while preserving variance. The PCA revealed overlapping user clusters, highlighting challenges in separating similar patterns.

![PCA Visualization](G3%20All%20Other%20Figures/PCA%20User%20mapping/pca.jpg)

### 🔴 Results and Optimization

- Initial models achieved an average accuracy of 93.14%, with FAR of 7.59% and EER of 3.93%.
- Optimized models using Genetic Algorithms and SVMs reached lower EERs (e.g., 0.97%) and improved robustness.

#### 1. Similarity Score Comparisons

Similarity scores provided key insights:

- **Legitimate Users:** High scores for their own data.
- **Imposters:** Consistently low scores.

## 🌟 Key Findings

- **Feature Selection:** Combining ANOVA, Mutual Information, and Steepest Gradient techniques yielded the most discriminative features.
- **Dimensionality Reduction:** PCA successfully identified clusters and reduced redundancy.
- **Advanced Optimization:** Techniques like GA+SVM improved precision and recall.
- **Robust Validation:** LOUO cross-validation ensured generalization to unseen users.

**🟢 Initial Model Score:**

![Initial Model Score](<G3%20All%20Other%20Figures/NN%20intial%20simillarity%20scores/intial_similarity_score(selected_leave_out).png>)

**🟢 Optimized Model Scores:**

|                  | GA+SVM model                                                                                                                             | ANOVA+MI+SG model                                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **with LOUO**    | ![with LOUO GA+SVM](G3%20All%20Other%20Figures/NN%20optimized%20%20SVM+GA%20simillarity%20scores/svm_fs_similarity_score.svg)            | ![with LOUO ANOVA+MI+SG](G3%20All%20Other%20Figures/NN%20optimized%20%20Anova+Mi+sgm%20simillarity%20scores/ANOVA_MI_SGM_LOUO.svg)           |
| **without LOUO** | ![without LOUO GA+SVM](G3%20All%20Other%20Figures/NN%20optimized%20%20SVM+GA%20simillarity%20scores/svm_fs_similarity_score_no_louo.svg) | ![without LOUO ANOVA+MI+SG](G3%20All%20Other%20Figures/NN%20optimized%20%20Anova+Mi+sgm%20simillarity%20scores/AONVA_MI_SGM_SS_NO__LUOU.svg) |

## 🚀 How to Run

1. Clone the repository.  
2. Install MATLAB and required toolboxes.  
3. Run the scripts in sequence, starting with ☛ [`model_initial.m`](./model_initial.m)

## 📑 References

1. Full reference list available in ☛ [`Extended Version of AI & ML Report _ GROUP - 34.pdf`](./Extended%20Version%20of%20AI%20&%20ML%20Report%20_%20GROUP%20-%2034.pdf)

## 😊 Contributors

- ![@AVDiv's Avatar](https://github.com/AVDiv.png?size=50) [**@AVDiv**](https://github.com/https://github.com/AVDiv)  
- ![@Pathfinder1152's Avatar](https://github.com/Pathfinder1152.png?size=50) [**@Pathfinder1152**](https://github.com/Pathfinder1152)  
- ![@kasrsu's Avatar](https://github.com/kasrsu.png?size=50) [**@kasrsu**](https://github.com/kasrsu)
- ![@GDJinasena's Avatar](https://github.com/GDJinasena.png?size=50) [**@GDJinasena**](https://github.com/GDJinasena)
- ![@scssandanayake's Avatar](https://github.com/scssandanayake.png?size=50) [**@scssandanayake**](https://github.com/scssandanayake)


