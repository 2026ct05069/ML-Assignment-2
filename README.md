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
ML-Assignment-2/
│
├── Model/
│   ├── 2026CT05069_ML_Assignment_...ipynb
│   └── hi
│
├── Results/
│   ├── class_distribution.png
│   ├── confusion_matrices.png
│   ├── metrics_bar_chart.png
│   ├── metrics_heatmap.png
│   └── metrics_per_metric.png
│
├── README.md
├── requirements.txt
└── test_data.csv
```

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

![All metrics for all models](Results/metrics_bar_chart.png)

![One panel per metric](Results/metrics_per_metric.png)

![Metric heatmap](Results/metrics_heatmap.png)

![Confusion matrices](Results/confusion_matrices.png)

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Logistic Regression came second on accuracy (0.583), AUC (0.672) and MCC (0.197). What I like about it is how steady it is: training and test accuracy are almost the same (gap of 0.003), so it is not memorising the data. The downside is that it finds only 2.1% of the early readmissions (`<30`), probably because a straight-line boundary cannot capture the more complicated patterns. |
| Decision Tree | The decision tree was a little behind (accuracy 0.573, AUC 0.643, MCC 0.173). Limiting its depth worked, since the train-test gap is only 0.011, but it still spots just 1.1% of the `<30` patients. |
| kNN | kNN did poorly. Its accuracy of 0.533 is lower than the 0.539 I would get by always guessing "not readmitted", and the gap between training and test accuracy (0.123) shows that it overfits. After encoding there are 205 features, so distances between patients probably stop meaning much. On the positive side, it finds 7.6% of the `<30` cases, which is more than Logistic Regression, the Decision Tree or the Random Forest manage. |
| Naive Bayes | Naive Bayes was last on accuracy (0.165), AUC (0.587), F1 (0.130) and MCC (0.067). It labels almost every patient as `<30`, which is why its recall for that class is a high 0.948 while its overall accuracy collapses, so that high number is not a good sign. The model assumes the features are independent and bell-shaped, and one-hot and medication columns do not fit that. |
| Random Forest (ensemble) | Random Forest came first on all six metrics (accuracy 0.591, AUC 0.679, F1 0.537, MCC 0.214). Its training accuracy is far above its test accuracy (gap of 0.310), which is common for forests because they nearly memorise the training data, but it still beats every other model on unseen data, and that is what matters. It finds only 0.9% of the `<30` cases, though. |
| **Overall winner for this dataset** | **Random Forest** is my overall winner. It ranks first on every metric, including MCC, which I trust most for an imbalanced three-class problem like this one. Logistic Regression is close behind and much steadier, but the forest copes better with non-linear patterns. I should be honest that the lead is small, and that none of the models is good at catching early readmissions. |

### Things to keep in mind

All models were run once on a single train-test split with fairly standard settings, so models with very close scores could swap places
with a different seed or with tuning. Also, the same patient can appear in several rows (several hospital stays), so a patient may end up
in both the training and the test set, which can make the scores slightly optimistic.
