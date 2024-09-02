## Leveraging Whisper ASR with LLM Fine-Tuning for End-to-End Multi-Lingual Speech Dialect Classification in Dravidian Languages

This repository provides detailed information about the ablation study conducted with our proposed approach.

We introduced a modified Whisper architecture, incorporating large language models (LLMs) with parameter-efficient fine-tuning (PEFT) techniques such as LoRA and QLoRA, for multi-dialect speech classification in Dravidian languages, including Tamil, Malayalam, and Kannada. To harness the efficiency of these PEFT techniques, we conducted various experiments by varying the rank parameter from 24 to 2. Detailed information regarding the training process and results is provided below.

In common fine-tuning, the entire pre-trained model is updated during the training process, meaning every parameter in the model is adjusted based on the new task-specific data. In contrast, PEFT focuses on fine-tuning a smaller subset of the model’s parameters, often by introducing additional task-specific parameters while keeping most of the pre-trained model frozen. Techniques such as LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) are commonly used in PEFT. In both LoRA and QLoRA, the total number of trainable parameters is controlled by a hyperparameter called "Rank." In our paper, we reported the results of the Whisper model using LoRA and QLoRA, both with a rank value of 32. The results for other rank values (24, 16, 8, 4, and 2) were investigated in our ablation study and are presented in Figure 4 of the paper. The detailed table of training parameters and macro-average F1 scores is available in this repository for reference.

In addition to these approaches, we also conducted experiments on our custom dataset using various standard models, including support vector machines (SVM), 1-dimensional convolutional neural networks (1D-CNN), 2-dimensional CNN (2D-CNN), long short-term memory (LSTM) networks, and a pre-trained automatic speech recognition (ASR) model for feature extraction and classification using the Wav2Vec2-large-XLS-R-53 model.

|    Rank  & Trainable Parameters    |  Languages |    LoRA   |        |          |   QLoRA   |        |          |
|:----------------------------------:|:----------:|:---------:|:------:|:--------:|:---------:|:------:|:--------:|
|                                    |            | Precision | Recall | F1-Score | Precision | Recall | F1-Score |
|  Rank-24, Trainable Parameters-61% | Tamil      | 0.84      | 0.83   | 0.83     | 0.84      | 0.83   | 0.83     |
|                                    | Malayalam  | 0.86      | 0.75   | 0.76     | 0.74      | 0.74   | 0.74     |
|                                    | Kannada    | 0.99      | 0.99   | 0.99     | 0.99      | 0.99   | 0.99     |
|  Rank-16, Trainable Parameters-40% | Tamil      | 0.86      | 0.85   | 0.85     | 0.88      | 0.88   |          |
| Trainable                          | Malayalam  | 0.87      | 0.84   | 0.85     | 0.78      | 0.75   | 0.76     |
| Parameters-61%                     | Kannada    | 0.99      | 0.99   | 0.99     | 0.99      | 0.99   | 0.99     |
|  Rank-8, Trainable Parameters-20%  | Tamil      | 0.83      | 0.83   | 0.83     | 0.78      | 0.79   | 0.78     |
|                                    | Malayalam  | 0.75      | 0.75   | 0.75     | 0.80      | 0.78   | 0.79     |
|                                    | Kannada    | 0.99      | 0.99   | 0.99     | 0.98      | 0.98   | 0.98     |
|  Rank-4, Trainable Parameters-10%  | Tamil      | 0.82      | 0.81   | 0.81     | 0.87      | 0.86   | 0.86     |
| Trainable                          | Malayalam  | 0.81      | 0.75   | 0.74     | 0.91      | 0.89   | 0.90     |
| Parameters-20%                     | Kannada    | 0.99      | 0.99   | 0.99     | 0.99      | 0.99   | 0.99     |
|  Rank-2, Trainable Parameters-5%   | Tamil      | 0.85      | 0.84   | 0.84     | 0.81      | 0.80   | 0.80     |
|                                    | Malayalam  | 0.87      | 0.83   | 0.84     | 0.90      | 0.87   | 0.88     |
|                                    | Kannada    | 0.98      | 0.97   | 0.97     | 0.97      | 0.97   | 0.97     |
