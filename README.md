## Leveraging Whisper ASR with LLM Fine-Tuning for End-to-End Multi-Lingual Speech Dialect Classification in Dravidian Languages

This repository provides detailed information about the ablation study conducted with our proposed approach.

We introduced a modified Whisper architecture, incorporating large language models (LLMs) with parameter-efficient fine-tuning (PEFT) techniques such as LoRA and QLoRA, for multi-dialect speech classification in Dravidian languages, including Tamil, Malayalam, and Kannada. To harness the efficiency of these PEFT techniques, we conducted various experiments by varying the rank parameter from 24 to 2. Detailed information regarding the training process and results is provided below.

In common fine-tuning, the entire pre-trained model is updated during the training process, meaning every parameter in the model is adjusted based on the new task-specific data. In contrast, PEFT focuses on fine-tuning a smaller subset of the model’s parameters, often by introducing additional task-specific parameters while keeping most of the pre-trained model frozen. Techniques such as LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) are commonly used in PEFT. In both LoRA and QLoRA, the total number of trainable parameters is controlled by a hyperparameter called "Rank." In our paper, we reported the results of the Whisper model using LoRA and QLoRA, both with a rank value of 32. The results for other rank values (24, 16, 8, 4, and 2) were investigated in our ablation study and are presented in Figure 4 of the paper. The detailed table of training parameters and macro-average F1 scores is available in this repository for reference.

#### Table 1: Model Size Estimates for Different Quantization Techniques 

|      Whisper-large-v2 (Total Trainable Parameters) | Rank | Reduced Trainable  Parameters | Reduced Trainable  Parameters (in percentage) | Trainable Parameter Size (GB) | LoRA (8-bit Quantization) Size (GB) | QLoRA (4-bit Quantization) Size (GB) |
|---------------------------------------------------|------|-------------------------------|-----------------------------------------------|---------------------------|--------------------------------------|---------------------------------------|
| 637,776,160                                       | 32   | 5,242,880                       | 80                                            | 2.551                     | 0.654                               | 0.337                               |
| 637,776,160                                       | 24   | 3,932,160                     | 61                                            | 2.551                     | 0.645                                | 0.332                                 |
| 637,776,160                                       | 16   | 2,621,440                     | 40                                            | 2.551                     | 0.645                                | 0.328                                 |
| 637,776,160                                       | 8    | 1,310,720                     | 20                                            | 2.551                     | 0.642                                | 0.324                                 |
| 637,776,160                                       | 4    | 655,360                       | 10                                            | 2.551                     | 0.6396                               | 0.3216                                |
| 637,776,160                                       | 2    | 327,680                       | 5                                             | 2.551                     | 0.6383                               | 0.3203                                |


#### Table 2: Detailed results for the Whisper classification model with LLM's LoRA and QLoRA adapters, showing varying rank values and trainable parameters.

|      |                                        |           |    LoRA   |  LoRA  |   LoRA   |   QLoRA   |  QLoRA |   QLoRA  |
|------|:--------------------------------------:|:---------:|:---------:|:------:|:--------:|:---------:|:------:|:--------:|
| Rank | Trainable  Parameters  (in percentage) | Languages | Precision | Recall | F1-Score | Precision | Recall | F1-Score |
| 24   | 61%                                    | Tamil     | 0.84      | 0.83   | 0.83     | 0.84      | 0.83   | 0.83     |
| 24   | 61%                                    | Malayalam | 0.86      | 0.75   | 0.76     | 0.74      | 0.74   | 0.74     |
| 24   | 61%                                    | Kannada   | 0.99      | 0.99   | 0.99     | 0.99      | 0.99   | 0.99     |
| 16   | 40%                                    | Tamil     | 0.86      | 0.85   | 0.85     | 0.88      | 0.88   | 0.88     |
| 16   | 40%                                    | Malayalam | 0.87      | 0.84   | 0.85     | 0.78      | 0.75   | 0.76     |
| 16   | 40%                                    | Kannada   | 0.99      | 0.99   | 0.99     | 0.99      | 0.99   | 0.99     |
| 8    | 20%                                    | Tamil     | 0.83      | 0.83   | 0.83     | 0.78      | 0.79   | 0.78     |
| 8    | 20%                                    | Malayalam | 0.75      | 0.75   | 0.75     | 0.80      | 0.78   | 0.79     |
| 8    | 20%                                    | Kannada   | 0.99      | 0.99   | 0.99     | 0.98      | 0.98   | 0.98     |
| 4    | 10%                                    | Tamil     | 0.82      | 0.81   | 0.81     | 0.87      | 0.86   | 0.86     |
| 4    | 10%                                    | Malayalam | 0.81      | 0.75   | 0.74     | 0.91      | 0.89   | 0.90     |
| 4    | 10%                                    | Kannada   | 0.99      | 0.99   | 0.99     | 0.99      | 0.99   | 0.99     |
| 2    | 5%                                     | Tamil     | 0.85      | 0.84   | 0.84     | 0.81      | 0.80   | 0.80     |
| 2    | 5%                                     | Malayalam | 0.87      | 0.83   | 0.84     | 0.90      | 0.87   | 0.88     |
| 2    | 5%                                     | Kannada   | 0.98      | 0.97   | 0.97     | 0.97      | 0.97   | 0.97     |


#### As part of an ablation study, we evaluated the Whisper model's performance on Kannada language data using 5-fold cross-validation to ensure that the model is not overfitting. The classification reports for all five folds, along with their average, are presented in the table below.

#### Table 3: 5-Fold Cross-Validation results for Kannada language. C-1, C-2, C-3, and C-4 refer to Class 1 (Coastal), Class 2 (North), Class 3 (South), and Class 4 (Uttara-Kannada), respectively. P, R, F1, and Acc denote Precision, Recall, F1 Score, and Accuracy.

| Fold    | Accuracy | C-1 P | C-1 R  | C-1 F1 | C-2 P | C-2 R | C-2 F1 | C-3 P | C-3 R | C-3 F1 | C-4 P | C-4 R | C-4 F1 |
|---------|----------|-------|--------|--------|--------|-------|--------|-------|-------|--------|-------|-------|--------|
| Fold 1  | 0.99     | 1.00  | 1.00   | 1.00   | 0.98   | 0.97  | 0.97   | 0.96  | 0.97  | 0.97   | 1.00  | 1.00  | 1.00   |
| Fold 2  | 0.99     | 1.00  | 1.00   | 1.00   | 0.99   | 0.95  | 0.97   | 0.95  | 0.99  | 0.97   | 1.00  | 1.00  | 1.00   |
| Fold 3  | 0.99     | 1.00  | 1.00   | 1.00   | 0.96   | 0.98  | 0.97   | 0.98  | 0.95  | 0.97   | 1.00  | 1.00  | 1.00   |
| Fold 4  | 0.99     | 1.00  | 1.00   | 1.00   | 0.98   | 0.97  | 0.98   | 0.97  | 0.98  | 0.97   | 1.00  | 1.00  | 1.00   |
| Fold 5  | 0.99     | 1.00  | 1.00   | 1.00   | 0.97   | 0.97  | 0.97   | 0.97  | 0.97  | 0.97   | 1.00  | 1.00  | 1.00   |
| Average | 0.99     | 1.00  | 1.00   | 1.00   | 0.984  | 0.972 | 0.972  | 0.964 | 0.972 | 0.97   | 1.00  | 1.00  | 1.00   |


##### **Interpretation:** The cross-validation results show that the model achieves a consistent accuracy of 99% across all five folds, with no significant variation in performance metrics. The perfect precision, recall, and F1 scores for Class 1 and Class 4, as well as very high scores for Class 2 and Class 3, across each fold further indicate that the model is not overfitting. The uniform performance across different folds confirms that the model generalizes well and maintains its effectiveness regardless of the specific subset of data used for training and validation.


#### In addition to these approaches, we also conducted experiments on our custom dataset using various standard models, including support vector machines (SVM), 1-dimensional convolutional neural networks (1D-CNN), 2-dimensional CNN (2D-CNN), long short-term memory (LSTM) networks, and a pre-trained automatic speech recognition (ASR) model for feature extraction and classification using the Wav2Vec2-large-XLS-R-53 model.

#### Table 4: Performance of standard models and traditional feature-based appraoches on our custom dataset across Tamil, Malayalam, and Kannada languages, including Macro Average Precision, Recall, and F1-Score for each language.

|                   | Tamil     | Tamil  | Tamil    | Malayalam | Malayalam | Malayalam | Kannada   | Kannada | Kannada  |
|-------------------|-----------|--------|----------|-----------|-----------|-----------|-----------|---------|----------|
| Model Names       | Precision | Recall | F1-Score | Precision | Recall    | F1-Score  | Precision | Recall  | F1-Score |
| MFCCs+DNN         | 0.84      | 0.83   | 0.83     | 0.84      | 0.81      | 0.82      | 0.95      | 0.95    | 0.95     |
| eGeMAPS+DNN       | 0.68      | 0.68   | 0.66     | 0.69      | 0.65      | 0.66      | 0.86      | 0.86    | 0.86     |
| SVM+MFCCs         | 0.73      | 0.72   | 0.72     | 0.81      | 0.80      | 0.80      | 0.91      | 0.91    | 0.91     |
| LSTM+MFCCs        | 0.84      | 0.83   | 0.83     | 0.86      | 0.84      | 0.85      | 0.97      | 0.97    | 0.97     |
| 1D-CNN+MFCCs      | 0.81      | 0.81   | 0.81     | 0.88      | 0.90      | 0.88      | 0.96      | 0.96    | 0.96     |
| 2D-CNN+MFCCs      | 0.78      | 0.77   | 0.76     | 0.88      | 0.82      | 0.83      | 0.97      | 0.97    | 0.97     |

