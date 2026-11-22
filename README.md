# Predicting Hospital Readmission of Diabetic Patients

**Machine Learning - Assignment 2 | Classification models compared on the Diabetes 130-US Hospitals dataset**


## 1. Problem Statement

When a diabetic patient is sent home and ends up back in hospital soon afterwards, it is usually bad news for the patient and
an expensive one for the hospital. If we could tell from the details of a hospital stay which patients are likely to return,
hospitals could give those patients extra follow-up care.

This project treats that as a classification problem. Using the records of a hospital stay (who the patient is, how long they
stayed, which tests and medicines they received, what they were diagnosed with), the goal is to predict whether the patient was
**readmitted within 30 days**, **readmitted after more than 30 days**, or **not readmitted at all**. Five different classification
models are trained on exactly the same prepared data and compared using six metrics: Accuracy, AUC, Precision, Recall, F1 and MCC.

---

## 2. Dataset Description

I used the **Diabetes 130-US Hospitals for Years 1999-2008** dataset from the UCI Machine Learning Repository. It holds hospital
records of diabetic patients collected across 130 US hospitals over ten years, with 101,766 encounters in total.

- Dataset page: https://archive.ics.uci.edu/dataset/296/diabetes+130-us-hospitals+for+years+1999-2008
- Licence: CC BY 4.0
- Citation: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999-2008* [Dataset].
  UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

### Why I chose this dataset

I chose it for a few reasons. Predicting whether a patient comes back to hospital is a problem hospitals care about in practice, since
repeat admissions are expensive and often point to gaps in the care given during the first stay. The dataset is also well above the
size requirements, with 47 input features and over 100,000 rows. It has real missing values, which meant I had to make actual
decisions about how to handle them instead of just ticking a box. Finally, the classes are clearly imbalanced, which makes the
choice of evaluation metrics matter.

### Target variable

The target variable is `readmitted`, which has three classes: `<30` (readmitted within 30 days), `>30` (readmitted after more than
30 days) and `NO` (never readmitted). This makes the task a multi-class classification problem.

### Feature types

The raw file has 49 columns besides the target. Two of them are patient and encounter identifiers, which carry no medical meaning,
so I removed them. I also dropped `weight` because almost all of its values are missing. That leaves 46 model inputs, of which 8 are
numerical and 38 are categorical.

The numerical ones are counts such as days in hospital, number of lab procedures, number of medications and the number of earlier
outpatient, emergency and inpatient visits. The categorical ones cover demographics (race, gender, age band), admission and discharge
details, payer and medical specialty, three diagnosis columns with hundreds of different ICD-9 codes each, two lab-result columns,
and 25 columns about diabetes medication. Three columns look numeric but are really integer codes for admission type, discharge
disposition and admission source, so I treated them as categories.

### Class distribution

The classes are uneven. Of the 101,766 encounters in the raw file, 54,864 were never readmitted, 35,545 were readmitted after 30 days,
and 11,357 were readmitted within 30 days.

| Class | Meaning | Count | Share |
| --- | --- | --- | --- |
| `NO` | not readmitted | 54,864 | 53.9% |
| `>30` | readmitted after 30 days | 35,545 | 34.9% |
| `<30` | readmitted within 30 days | 11,357 | 11.2% |

The largest class is almost five times the size of the smallest. A model that always predicted "not readmitted" would already be right
about 54% of the time without learning anything. Because of this, I did not rely on accuracy alone. I also looked at AUC, MCC and how
well each model finds the small early-readmission group.


### Data preparation

- **Missing values.** The file marks missing values with `?`. I dropped `weight` (about 97% missing). For the remaining numerical
  columns I filled gaps with the median. For categorical columns such as `race`, `payer_code` and `medical_specialty` I gave missing
  values their own "Missing" category, because for columns with so many gaps the fact that a value is missing can itself be informative.
- **Duplicates and unusable rows.** I checked for duplicate rows and rows without a target value and removed them.
- **Encoding.** Categorical features were one-hot encoded. Categories used by fewer than 1% of rows were grouped together, so the
  diagnosis columns did not blow up into thousands of columns.
- **Scaling.** Numerical features were standardised, which matters for Logistic Regression and kNN.
- **Train-test split.** 80% training and 20% test, stratified on the target so both parts keep the same class proportions, with a fixed
  random seed of 42.
- **No data leakage.** All of the steps above are part of a scikit-learn `Pipeline`, so the medians, scaling values and category lists
  are learned from the training data only and then applied to the test data.

---

## 3. Repository and How to Run

GitHub repository: **https://github.com/2026ct05069/ML-Assignment-2**

```
project-folder/
|-- README.md
|-- requirements.txt
|-- test_data.csv                      (the 20% test set, created by the notebook)
|-- main.py                            (same pipeline as the notebook, runs from the command line)
|-- data/                              (diabetic_data.csv goes here)
|-- results/                           (comparison table, observations and graphs)
|-- model/
    |-- ml_assignment_2_diabetes.ipynb (the full notebook: data prep, five models, metrics, observations)
    |-- logistic_regression.py
    |-- decision_tree.py
    |-- knn.py
    |-- naive_bayes.py
    |-- random_forest.py
    |-- data_utils.py
    |-- evaluate.py
    |-- saved/                         (trained models, *.joblib)
```

To run it yourself:

```bash
pip install -r requirements.txt
```

Then open `model/ml_assignment_2_diabetes.ipynb` and run all cells. If `data/diabetic_data.csv` is not there, the notebook downloads the
dataset from UCI directly. Or, from the command line:

```bash
python main.py --data data/diabetic_data.csv --target readmitted \
  --drop-cols encounter_id patient_nbr weight \
  --as-categorical admission_type_id discharge_disposition_id admission_source_id
```

---

## 4. Models Used

All five models were trained on the same training set and evaluated on the same test set. Since the target has three classes,
Precision, Recall and F1 are weighted averages over the classes, AUC is the one-vs-rest macro average, and MCC is the multi-class
version of the Matthews correlation coefficient.

| Model | Settings used |
| --- | --- |
| Logistic Regression | `max_iter=1000`, `C=1.0`, `random_state=42` |
| Decision Tree | `criterion="gini"`, `max_depth=8`, `min_samples_leaf=5`, `random_state=42` |
| kNN | `n_neighbors=7`, uniform weights, Euclidean distance |
| Naive Bayes | Gaussian Naive Bayes, default settings |
| Random Forest (ensemble) | `n_estimators=200`, `min_samples_leaf=2`, `random_state=42` |

### Comparison of the models (test set)

| **ML Model Name** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |
| --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.5830 | 0.6725 | 0.5462 | 0.5830 | 0.5292 | 0.1974 |
| **Decision Tree** | 0.5728 | 0.6430 | 0.5320 | 0.5728 | 0.5163 | 0.1732 |
| **kNN** | 0.5330 | 0.5962 | 0.5007 | 0.5330 | 0.5108 | 0.1373 |
| **Naive Bayes** | 0.1649 | 0.5871 | 0.5835 | 0.1649 | 0.1300 | 0.0673 |
| **Random Forest (Ensemble)** | 0.5911 | 0.6793 | 0.5842 | 0.5911 | 0.5375 | 0.2143 |

### Graphs of all metrics

**[After running the notebook, keep the images below. They are created in the `results/` folder.]**

!(metrics_bar_chart-1.png)(results/metrics_bar_chart.png)

![One panel per metric](results/metrics_per_metric.png)

![Metric heatmap](results/metrics_heatmap.png)

![Confusion matrices](results/confusion_matrices.png)

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| **Logistic Regression** | A linear model on scaled and one-hot inputs: fast, stable and well calibrated, but it cannot model non-linear interactions between features. Test accuracy 0.583 (rank 2 of 5; majority-class baseline 0.539), AUC 0.672, F1 0.529, MCC 0.197 (rank 2). Fit check: consistent generalisation (train-test accuracy gap 0.003). Recall for the minority class '<30' is 0.021. |
| **Decision Tree** | A single depth-limited tree: easy to interpret and able to capture non-linear splits, but a single tree is high-variance. Test accuracy 0.573 (rank 3 of 5; majority-class baseline 0.539), AUC 0.643, F1 0.516, MCC 0.173 (rank 3). Fit check: consistent generalisation (train-test accuracy gap 0.011). Recall for the minority class '<30' is 0.011. |
| **kNN** | Distance based on 205 encoded features: distances become less informative in high dimensions, and prediction is slow on a large test set. Test accuracy 0.533 (rank 4 of 5; majority-class baseline 0.539), AUC 0.596, F1 0.511, MCC 0.137 (rank 4). Fit check: clear overfitting (train-test accuracy gap 0.123). Recall for the minority class '<30' is 0.076. |
| **Naive Bayes** | Assumes features are independent and Gaussian; the one-hot and medication features are correlated and not Gaussian, so the assumptions are violated. Test accuracy 0.165 (rank 5 of 5; majority-class baseline 0.539), AUC 0.587, F1 0.130, MCC 0.067 (rank 5). Fit check: consistent generalisation (train-test accuracy gap 0.003). Recall for the minority class '<30' is 0.948. |
| **Random Forest (Ensemble)** | Averages many de-correlated trees, which reduces variance and captures non-linear interactions. Test accuracy 0.591 (rank 1 of 5; majority-class baseline 0.539), AUC 0.679, F1 0.537, MCC 0.214 (rank 1). Fit check: clear overfitting (train-test accuracy gap 0.310). Recall for the minority class '<30' is 0.009. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** is the overall winner: it has the highest MCC (0.214), which is the most reliable single score on an imbalanced multi-class target, and it is best on 6 of the 6 metrics (best per metric: Accuracy: Random Forest (Ensemble), AUC: Random Forest (Ensemble), Precision: Random Forest (Ensemble), Recall: Random Forest (Ensemble), F1: Random Forest (Ensemble), MCC: Random Forest (Ensemble)). Its AUC is 0.679 and its train-test accuracy gap is 0.310. |

### Things to keep in mind

All models were run once on a single train-test split with fairly standard settings, so models with very close scores could swap places
with a different seed or with tuning. Also, the same patient can appear in several rows (several hospital stays), so a patient may end up
in both the training and the test set, which can make the scores slightly optimistic.
