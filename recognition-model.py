import numpy as np
import struct
import joblib
import tensorflow as tf
from keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

def load_idx(filepath):
    """
    Loads MNIST/EMNIST idx file format.
    Returns a NumPy array of shape:
       - (N,) for label files
       - (N, rows, cols) for image files
    """
    with open(filepath, 'rb') as f:
        # Read the magic number and the size (# of items)
        magic, size = struct.unpack('>II', f.read(8))
        
        # Magic 2051 = images, 2049 = labels
        if magic == 2051:
            # This is an images file
            rows, cols = struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape((size, rows, cols))
        elif magic == 2049:
            # This is a labels file
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Invalid magic number {magic} in file {filepath}")
    
    return data

# Based on Le-Net
def build_cnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


train_images_path = "/Users/anthonyszykowny/Downloads/archive 2/emnist_source_files/emnist-byclass-train-images-idx3-ubyte"
train_labels_path = "/Users/anthonyszykowny/Downloads/archive 2/emnist_source_files/emnist-byclass-train-labels-idx1-ubyte"
test_images_path  = "/Users/anthonyszykowny/Downloads/archive 2/emnist_source_files/emnist-byclass-test-images-idx3-ubyte"
test_labels_path  = "/Users/anthonyszykowny/Downloads/archive 2/emnist_source_files/emnist-byclass-test-labels-idx1-ubyte"
# train_images_path = "/Users/anthonyszykowny/Downloads/archive 2/emnist_source_files/emnist-balanced-train-images-idx3-ubyte"
# train_labels_path = "/Users/anthonyszykowny/Downloads/archive 2/emnist_source_files/emnist-balanced-train-labels-idx1-ubyte"
# test_images_path  = "/Users/anthonyszykowny/Downloads/archive 2/emnist_source_files/emnist-balanced-test-images-idx3-ubyte"
# test_labels_path  = "/Users/anthonyszykowny/Downloads/archive 2/emnist_source_files/emnist-balanced-test-labels-idx1-ubyte"


# Load data from IDX files
X_train = load_idx(train_images_path)  # shape: (N_train, 28, 28)
y_train = load_idx(train_labels_path)  # shape: (N_train,)
X_test  = load_idx(test_images_path)   # shape: (N_test, 28, 28)
y_test  = load_idx(test_labels_path)   # shape: (N_test,)

print("Train images shape:", X_train.shape)
print("Train labels shape:", y_train.shape)
print("Test images shape: ", X_test.shape)
print("Test labels shape: ", y_test.shape)

# Preprocess: reshape & scale
# Reshape to (N, 28, 28, 1) and scale pixel values to [0,1].
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Encode labels as needed
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

num_classes = len(np.unique(y_train_enc))
print("Number of classes:", num_classes)

# One-hot encode the labels for training
y_train_oh = tf.keras.utils.to_categorical(y_train_enc, num_classes)
y_test_oh  = tf.keras.utils.to_categorical(y_test_enc, num_classes)

# Build & train the CNN

model = build_cnn_model((28, 28, 1), num_classes)
model.summary()

EPOCHS = 10
BATCH_SIZE = 64

history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_test, y_test_oh),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Evaluate
y_pred = model.predict(X_test).argmax(axis=1)
acc = accuracy_score(y_test_enc, y_pred)
f1 = f1_score(y_test_enc, y_pred, average='weighted')

print("Test Accuracy:", acc)
print("Weighted F1 Score:", f1)
print("Classification Report:")
print(classification_report(y_test_enc, y_pred))

# Save the model + label encoder
MODEL_PATH = "cnn_emnist.h5"
LE_PATH = "label_encoder_emnist.joblib"

model.save(MODEL_PATH)
print(f"Saved CNN model to {MODEL_PATH}")

joblib.dump(le, LE_PATH)
print(f"Saved label encoder to {LE_PATH}")
