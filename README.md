# Assignment 1: High-Level Design for a Machine Learning Project

I used [template](https://github.com/eugeneyan/ml-design-docs) for design docs for machine learning systems.

---
## 1. Overview

Automated Social Media Moderation. The objective is to create a system that detects and moderates harmful content on social media while adapting to new forms of harmful behavior and changes in usage patterns. The system will output a confidence score for each prediction, messages with low confidence will be forwarded to human moderators for review.


## 2. Motivation
Moderating harmful content can have a negative psychological impact on human moderators. By creating a system that automatically classifies whether content is harmful, we can significantly reduce human exposure to toxic material and improve the overall health of the social media environment.

## 3. Success metrics
Reduce harmful content exposure while requiring fewer human moderators.
Decrease the total number of user reports.
Increase user engagement, measured by Daily Active Users (DAU).

## 4. Requirements & Constraints

Requirements for product:
* High recall > 80%;
* Low amount of false positives < 15%;
* Low amount of messages with low confidense score < 10%;
* System needs to be stable, availability > 99%;
* avarage latency < 1 sec;
* 99th percentile latency < 10 sec.

Constraints:
* System need to be very cheap to deploy, train, run. < 50$ for whole project


### 4.1 What's in-scope & out-of-scope?

In-scope:
* Text clasification into classes
* API for inference
* English language support
* MLflow-based experiment tracking
* CI/CD with GitHub Actions
* Cloud deployment

Out-of-scope:
* Image, video, sound clasification
* User history based analysis
* High interpretability

## 5. Methodology

### 5.1. Problem statement

The task is to classify a text message into one of three classes:
* hate speech
* offensive language
* neither

For each request, the system will return the predicted class along with a confidence score. Supervised learning with labeled data will be used. Initially, a baseline model will be built, followed by more advanced models that aim to outperform the baseline.

### 5.2. Data

The model will use only the text content of messages as input. The dataset used is the [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset).

The dataset is based on Twitter data and contains 24,783 unique English messages. Each message is labeled as hate speech, offensive language, or neither. Labels are derived from votes by CrowdFlower users.


### 5.3. Techniques

The data will be cleaned and prepared before training using:
* Removal of special characters
* Removal of non-meaningful links
* Removal of excesive punctuation
* Replacement usernames with \<USER\>
* Replacement links with \<LINK\>

From messages will be extracted such information as:
* TF-IDF features
* Capitalization ratio

For baseline model i will use logistic regression or in case of a bad perfomance some tree based model with TF-IDF features.

For adavnced model i will try to use LSTM or some small transformer.

Because the dataset is imbalanced, techniques such as class weighting during training will be applied.

### 5.4. Experimentation & Validation

The dataset will be split into training, validation and test
Evaluation metrics will include recall, false positive rate, F1.

A/B testing will be used for champion and chellenger models.
Metrics for it is:
* Accuracy of harmfull vs not-harmfull content
* Recall
* Amount of low confidence results

### 5.5. Human-in-the-loop

Human-in-the-loop will be used for clasification of new data if needed. Also moderation team will deside what to do with low-confidence predictions.

## 6. Implementation

### 6.1. High-level design

![](images/1.drawio.png)

### 6.2. Infra

While i develop i will use local machine with Docker Desktop, Python 3.14.2.

For production i am planing to deploy core services into AWS EC2 with database in S3.

### 6.3. Performance (Throughput, Latency)

Model service will scale horizontaly.

### 6.4. Security

For production it will use AWS security measures security groups/VPC.

### 6.5. Data privacy


### 6.6. Monitoring & Alarms

I will use AWS load balancer functionality for scailing services if needed. Core metricas for this will be latency and CPU/GPU utilization. 

For model perfomance monitoring I will be using Airflow which will periodicly compare model outputs with results of user interactions with post, from that data we will get if model performance changed and allow us to set up alarms or actions for it.

### 6.7. Cost


### 6.8. Integration points

System integrates with upstream data sources through a REST API. Text content is sent to the moderation service.
Downstream consumers is automated baning system, human moderators, MLflow monitoring, analytics.

### 6.9. Risks & Uncertainties

Risks:
* biased training data
* costs

Uncertainties:
* harmful language evolution
* human moderators bias
* evolving behaviour of users to evade automatic flagging

## 7. Appendix

### 7.6. References

[Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

[MLflow docs](https://mlflow.org/docs/latest/)

[Original template](https://github.com/eugeneyan/ml-design-docs)