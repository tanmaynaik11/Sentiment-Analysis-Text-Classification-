import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy


# For BERT based approach
# use the tokenizer from BERT
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(examples):
    """Tokenize the input text and return input_ids and attention_mask."""
    tokens = tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    }

def train(model_path="model",
          train_path="/content/drive/MyDrive/Colab Notebooks/train.csv",
          dev_path="/content/drive/MyDrive/Colab Notebooks/dev.csv"):

    # Load CSVs into HuggingFace datasets
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path,
        "validation": dev_path
    })

    # Define label names
    labels = hf_dataset["train"].column_names[1:]  # First column is 'text'

    def gather_labels(example):
        """Combine all label columns into a single list."""
        return {"labels": [float(example[l]) for l in labels]}

    # Preprocess: label formatting + tokenization
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # Convert HuggingFace dataset to TensorFlow dataset
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

    # Load pretrained BERT base model (without classification head)
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")

    # Input layers
    input_ids = tf.keras.Input(shape=(64,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(64,), dtype=tf.int32, name="attention_mask")

    # BERT output returns a tuple with multiple outputs (depending on the BERT variant you're using)
    # output = (last_hidden_state, pooled_output, hidden_states, attentions)
    # [0] gives last hidden state, This contains the contextualized embeddings for every token in the input
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]

    # Pooling
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(bert_output)
    x = tf.keras.layers.LayerNormalization()(pooled_output)

    # Dense head for multi-label classification
    dropout = tf.keras.layers.Dropout(0.4)(x)
    dense = tf.keras.layers.Dense(256, activation="relu",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005))(dropout)
    output = tf.keras.layers.Dense(len(labels), activation="sigmoid")(dense)

    # Final model
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)]
    )

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{model_path}.keras",
        monitor="val_f1_score", mode="max", save_best_only=True, verbose=1
    )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

    # Train
    model.fit(
        train_dataset,
        validation_data=dev_dataset,
        epochs=4,
        callbacks=[early_stopping, lr_scheduler, model_checkpoint, tensorboard]
    )




# For LSTM Based Approach

# use the tokenizer from DistilRoBERT
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text"""
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length", return_tensors="tf")

def train(model_path="model", train_path="/content/drive/MyDrive/Colab Notebooks/train.csv", dev_path="/content/drive/MyDrive/Colab Notebooks/dev.csv"):
    # Load the CSVs into HuggingFace datasets to allow tokenizer usage
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path
    })

    # Define labels as column names except the first (text column)
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Convert label columns into a list of 0s and 1s"""
        return {"labels": [float(example[l]) for l in labels]}

    # Map the tokenizer and label gather functions
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # Convert HuggingFace datasets to TensorFlow datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns=["input_ids"], label_cols="labels", batch_size=16, shuffle=True
    )
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns=["input_ids"], label_cols="labels", batch_size=16
    )

    # Define model architecture
    input_layer = tf.keras.Input(shape=(64,), dtype=tf.int32, name="input_ids")
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=tokenizer.vocab_size, output_dim=128, input_length=64)(input_layer)
    lstm_output = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(embedding_layer)
    attention_output = tf.keras.layers.Attention()([lstm_output, lstm_output])
    attention_output = tf.keras.layers.LayerNormalization()(attention_output) #trial
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
    dropout_layer = tf.keras.layers.Dropout(0.5)(pooled_output)
    dense_layer = tf.keras.layers.Dense(256, activation="relu")(dropout_layer) #trial # add L2 regularization for dense layer with value 0.0005
    output_layer = tf.keras.layers.Dense(len(labels), activation="sigmoid")(dense_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1), #try label smmothing as 0.05
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)]
    )

    # model = transformers.TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels))
    # optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    # model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)])

    #Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    )
    # try cosine or exponential LR for smoother rate decay
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{model_path}.keras",
        monitor="val_f1_score",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

    # Train the model
    model.fit(
        train_dataset,
        validation_data=dev_dataset,
        epochs=10,
        callbacks=[early_stopping, lr_scheduler, model_checkpoint, tensorboard]
    )