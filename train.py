#!/usr/bin/env python3
"""
@project: cv_classification
@file: train
@author: mike
@time: 2021/3/3
 
@function:
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from keras import backend as K
from models.LeNet import LeNet
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Step 1: load the dataset
print('[INFO] loading MNIST...')
dataset = datasets.load_digits()
# data
# target
# frame
# feature_names
# target_names
# images
# DESCR
data = dataset.data
target = dataset.target

# Step 2: process
# Step 2-1: scale the input arrange
data = data / 16.0  # from [0-16] to [0-1]
target = target.astype('int')

# Step 2-2: change shape
if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0], 1, 8, 8)
else:
    data = data.reshape(data.shape[0], 8, 8, 1)

# Step 2-3: train, test split (this is optional)
# If the train test is split before the loading,
# there is no more split
train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=16)

# Step 2-4: convert the labels from integers to vectors
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# Step 3: initialize the optimizer and model
print('[INFO] compiling model...')
opt = SGD(learning_rate=0.01)
model = LeNet(width=8, height=8, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

# Step 4: train the model
print('[INFO] training network...')
epochs = 100
# fit method can only be used for small dataset, for large dataset use the fit_generate instead
H = model.fit(train_x, train_y, batch_size=128, epochs=epochs, verbose=2, validation_data=(test_x, test_y))

# Step 5: evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(test_x, batch_size=128)

# step 6: save the model
model.save('checkpoints/lenet100.h5')

# Step 7: plot the train process
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, epochs), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, epochs), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# Step 8: write the statistics to a file (or just print)
statistics = classification_report(test_y.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names=[str(x) for x in lb.classes_])
with open('statistics/statistics', 'w') as fh:
    for line in statistics:
        fh.write(line)
