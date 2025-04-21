# Multi Label Classification with LSTM with Attention and BERT

## Overview
This project implements a multi-label text classification model using a Bidirectional LSTM with Attention and BERT. The models are trained on custom-labeled datasets and optimized for better generalization with label smoothing, dropout, and regularization.

## Features
LSTM APPROACH
Uses BERT tokenizer for text preprocessing.
Implements a Bidirectional LSTM + Attention model.
Optimized with dropout, L2 regularization, and learning rate scheduling.
Supports multi-label classification with sigmoid activation.
Early stopping & model checkpointing for efficient training.

BERT APPROACH
Uses BERT tokenizer for text preprocessing.
Implements a BERT model.
Optimized with dropout and learning rate scheduling.
Optimized with dropout, L2 regularization, and learning rate scheduling.
Early stopping & model checkpointing for efficient training.

## Dataset
The model expects CSV files with the following format:
| text | label_1 | label_2 | label_3 | ... |
|------|---------|---------|---------|
| "Sample text" | 1 | 0 | 0 |
| "Another example" | 0 | 1 | 1 |

## Model Architecture For LSTM
Input: Tokenized text (BERT-based embeddings)
Embedding Layer: 128-dimensional word embeddings
Bidirectional LSTM: 128 units, dropout (0.3), recurrent dropout (0.3)
Attention Mechanism: Improves focus on relevant words
Global Average Pooling: Reduces dimensionality
Fully Connected Layer: 256 neurons, ReLU activation (L2 regularization: 0.0005)
Output Layer: Multi-label classification (sigmoid activation)
Hyperparameter Tuning
Label smoothing: 0.1 (can be adjusted in train())
Optimizer: Adam (learning_rate=0.001)
Learning Rate Scheduling: Reduces LR when validation loss plateaus
Dropout Rate: 0.5 for improved generalization

## Evaluation Metrics
The LSTM and BERT model is evaluated using:

F1 Score (micro-averaged, threshold=0.5)
Binary Crossentropy Loss
Current F1 Score for LSTM Model: ~83%
Current F1 Score for LSTM Model: ~85% Faster training than LSTM approach as we use pre-trained model

## Logging & Monitoring
TensorBoard for tracking training progress (logs/ directory).
Model Checkpointing to save the best model based on validation F1 score.

Future Enhancements
Try cosine or exponential learning rate decay for smoother training.
Use larger datasets and fine-tuning for improved generalization.
