## Leveraging Whisper ASR with LLM Fine-Tuning for End-to-End Multi-Lingual Speech Dialect Classification in Dravidian Languages

This repository provides detailed information about the ablation study conducted with our proposed approach.

We introduced a modified Whisper architecture, incorporating large language models (LLMs) with parameter-efficient fine-tuning (PEFT) techniques such as LoRA and QLoRA, for multi-dialect speech classification in Dravidian languages, including Tamil, Malayalam, and Kannada. To harness the efficiency of these PEFT techniques, we conducted various experiments by varying the rank parameter from 24 to 2. Detailed information regarding the training process and results is provided below.

In common fine-tuning, the entire pre-trained model is updated during the training process, meaning every parameter in the model is adjusted based on the new task-specific data. In contrast, PEFT focuses on fine-tuning a smaller subset of the model’s parameters, often by introducing additional task-specific parameters while keeping most of the pre-trained model frozen. Techniques such as LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) are commonly used in PEFT. In both LoRA and QLoRA, the total number of trainable parameters is controlled by a hyperparameter called "Rank." In our paper, we reported the results of the Whisper model using LoRA and QLoRA, both with a rank value of 32. The results for other rank values (24, 16, 8, 4, and 2) were investigated in our ablation study and are presented in Figure 4 of the paper. The detailed table of training parameters and macro-average F1 scores is available in this repository for reference.

#### Effectiveness of using LoRA and QLoRA with whisper

|      Whisper-large-v2 (Total Trainable Parameters | Rank | Reduced Trainable  Parameters | Reduced Trainable  Parameters (in percentage) | Original Model  Size (GB) | LoRA (8-bit Quantization)  Size (GB) | QLoRA (4-bit Quantization)  Size (GB) |
|---------------------------------------------------|------|-------------------------------|-----------------------------------------------|---------------------------|--------------------------------------|---------------------------------------|
| 637,776,160                                       | 32   | 5,242,880                       | 80                                            | 2.551                     | 0.654                               | 0.337                               |
| 637,776,160                                       | 24   | 3,932,160                     | 61                                            | 2.551                     | 0.645                                | 0.332                                 |
| 637,776,160                                       | 16   | 2,621,440                     | 40                                            | 2.551                     | 0.645                                | 0.328                                 |
| 637,776,160                                       | 8    | 1,310,720                     | 20                                            | 2.551                     | 0.642                                | 0.324                                 |
| 637,776,160                                       | 4    | 655,360                       | 10                                            | 2.551                     | 0.6396                               | 0.3216                                |
| 637,776,160                                       | 2    | 327,680                       | 5                                             | 2.551                     | 0.6383                               | 0.3203                                |

#### The detailed results for Whisper classification model with LLM's LoRA and QLoRA Adapters with varying rank values and trainable parameter are reported below in the table

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


In addition to these approaches, we also conducted experiments on our custom dataset using various standard models, including support vector machines (SVM), 1-dimensional convolutional neural networks (1D-CNN), 2-dimensional CNN (2D-CNN), long short-term memory (LSTM) networks, and a pre-trained automatic speech recognition (ASR) model for feature extraction and classification using the Wav2Vec2-large-XLS-R-53 model.

