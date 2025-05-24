### Project Demo



https://github.com/user-attachments/assets/c4684246-2181-4ea0-824a-40814c965564



### Introduction

<br>

* Cricket is a globally celebrated sport with profound economic impacts, involving billions in revenue and extensive fan engagement.
* AI-driven data analytics is rapidly transforming cricket, offering new avenues for player development and strategic planning.
* This project focuses on classifying cricket shots from videos into distinct categories and exploring the similarities between these shots.
* By leveraging these insights, players can enhance their skills, and coaches can identify promising new talent more effectively.


<br>

### Dataset Preparation

![329913659-f9fd9c8f-925c-41ec-957a-306bae1fbdb4](https://github.com/user-attachments/assets/6f056192-980b-4926-b5d1-68e79fd0dea8)

### Model Training

<br>



| Model               | Training Accuracy | Validation Accuracy |
|---------------------|-------------------|---------------------|
| EfficientNetB0      | 100%              | 85.80%              |
| EfficientNetV2B0    | 100%              | 77.01%              |
| EfficientNetB4      | 100%              | 72.86%              |


* Built three model variants, each with a distinct feature extractor head to evaluate performance variations.
* Trained all models for 20 epochs using batch sizes of 16, processing 30 frames per video to capture temporal dynamics.
* Utilized the Adam optimizer, configured with a learning rate of 0.001, to efficiently converge to optimal weights.
* Employed sparse categorical crossentropy as the loss function, for handling class labels as integers.


### Model Evaluation


<br>



| Model            | Testing Accuracy | Precision | Recall | F1 Score |
|------------------|------------------|-----------|--------|----------|
| EfficientNetB0   | 94%              | 94%       | 94%    | 94%      |
| EfficientNetV2B0 | 81%              | 82%       | 81%    | 81%      |
| EfficientNetB4   | 74%              | 75%       | 74%    | 74%      |


* All three models were evaluated on the test set.
* Accuracy, Precision, Recall, F1-score were the metrics used for evaluation.
* Model with EffiecientNet B0 backbone outperformed the other two models.


<br>
