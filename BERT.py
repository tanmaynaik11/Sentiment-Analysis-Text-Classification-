import datasets
import pandas
import transformers
import tensorflow as tf
import numpy as np
from transformers import TFBertModel, AutoTokenizer


tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(examples):
    tokens = tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    }

def train(model_path="model",
          train_path="/content/drive/MyDrive/Colab Notebooks/train.csv",
          dev_path="/content/drive/MyDrive/Colab Notebooks/dev.csv"):

    # Load dataset
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path,
        "validation": dev_path
    })

    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        return {"labels": [float(example[l]) for l in labels]}

    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=16,
        shuffle=True
    )
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=16
    )

    # Load BERT backbone
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")

    # Freeze embeddings
    bert_model.bert.embeddings.trainable = False

    # Freeze encoder layers 0–8, fine-tune 9–11
    for i, layer in enumerate(bert_model.bert.encoder.layer):
        if i < 9:
            layer.trainable = False
        else:
            layer.trainable = True

    # Define your classifier
    class BertMultiLabelClassifier(tf.keras.Model):
        def __init__(self, bert_model, num_labels):
            super().__init__()
            self.bert = bert_model
            self.pooling = tf.keras.layers.GlobalAveragePooling1D()
            self.norm = tf.keras.layers.LayerNormalization()
            self.dropout = tf.keras.layers.Dropout(0.4)
            self.dense = tf.keras.layers.Dense(256, activation="relu",
                                               kernel_regularizer=tf.keras.regularizers.l2(0.0005))
            self.out = tf.keras.layers.Dense(num_labels, activation="sigmoid")

        def call(self, inputs):
            x = self.bert(inputs)[0]  # last_hidden_state
            x = tf.keras.layers.Dropout(0.3)(x)
            x = self.pooling(x)
            x = self.norm(x)
            x = self.dropout(x)
            x = self.dense(x)
            return self.out(x)

    # Initialize model
    model = BertMultiLabelClassifier(bert_model, num_labels=len(labels))

    # ✅ Recompile after freezing layers
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)]
    )

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs")

    # Train
    model.fit(
        train_dataset,
        validation_data=dev_dataset,
        epochs=5,
        callbacks=[early_stopping, lr_scheduler, tensorboard]
    )

    # Save model
    bert_model.save_pretrained(f"{model_path}/bert")
    tokenizer.save_pretrained(f"{model_path}/bert")
    model.save_weights(f"{model_path}/custom_head.weights.h5")

    print(f"Training complete")

class BertMultiLabelClassifier(tf.keras.Model):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.dense = tf.keras.layers.Dense(256, activation="relu",
                                           kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.out = tf.keras.layers.Dense(num_labels, activation="sigmoid")

    def call(self, inputs):
        x = self.bert(inputs)[0]
        x = tf.keras.layers.Dropout(0.3)(x)
        x = self.pooling(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.dense(x)
        return self.out(x)

def predict(model_path="model", input_path="/content/drive/MyDrive/Colab Notebooks/dev.csv"):
    # Load tokenizer and base BERT model
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/bert")
    bert_model = TFBertModel.from_pretrained(f"{model_path}/bert")

    # Load input data
    df = pd.read_csv(input_path)
    text_column = "text"
    num_labels = df.shape[1] - 1  # assuming first column is 'text', rest are labels

    # Tokenize using HuggingFace Dataset
    def tokenize(example):
        tokens = tokenizer(example[text_column], padding="max_length", truncation=True, max_length=64)
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

    hf_dataset = Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    tf_dataset = hf_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        batch_size=16,
        shuffle=False
    )

    # Rebuild the classifier
    model = BertMultiLabelClassifier(bert_model, num_labels)

    # call the model once to build it before loading weights
    dummy_inputs = {
        "input_ids": tf.zeros((1, 64), dtype=tf.int32),
        "attention_mask": tf.zeros((1, 64), dtype=tf.int32)
    }
    model(dummy_inputs)  # builds model graph

    # Load classifier head weights
    model.load_weights(f"{model_path}/custom_head.weights.h5")

    # Predict
    probs = model.predict(tf_dataset)
    predictions = np.where(probs > 0.5, 1, 0)

    # Replace label columns with predictions
    df.iloc[:, 1:] = predictions

    # Save to zip
    df.to_csv("submission_85.zip", index=False, compression=dict(
        method='zip', archive_name='submission_85.csv'))

    print("Predictions written")
