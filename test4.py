# from goodboy_Tien_Anh_Nguyen import Love

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os


# call me daddy
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="training_set",
                help="path to input dataset")
ap.add_argument("-m", "--model", type=str, default="trained_model",
                help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# khoi tao 1 so tham so co ban
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

# data va nhan
data = []
labels = []
dir_labels = ()
num_class = 0


# nhan kem rieng tai nha
# 0328874444
print("[INFO] Finding Labels...")
for file in os.listdir(args["dataset"]):
    temp_tuple = (file, 'null')
    dir_labels = dir_labels + temp_tuple
    dir_labels = dir_labels[:-1]

    num_class = num_class + 1

print("[INFO] Loading Images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)


for imagePath in imagePaths:
    # load anh, xu ly va luu
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]


    for i in range(num_class):
        if label == dir_labels[i]:
            label = i

    labels.append(label)


data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# chia tap du lieu thanh train va test
(train_val_X, test_X, train_val_Y, test_Y) = train_test_split(data,
                                                  labels, test_size=0.2, random_state=42)

train_X, val_X, train_Y, val_Y = train_test_split(train_val_X, train_val_Y, test_size=0.2)

# chuyen nhan tu so thanh vector
train_Y = to_categorical(train_Y, num_classes=num_class)
test_Y = to_categorical(test_Y, num_classes=num_class)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# xay dung model
print("[INFO] Compiling Model...")
model = LeNet.build(width=28, height=28, depth=3, classes=num_class)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train model
print("[INFO] Training Network...")
H = model.fit_generator(aug.flow(train_X, train_Y, batch_size=BS),
                        validation_data=(test_X, test_Y), steps_per_epoch=len(train_X) // BS,
                        epochs=EPOCHS, verbose=1)

# danh gia model
result = model.evaluate(test_X, test_Y)
print("loss: ", result[0])
print("accuracy: ", result[1])

print("[INFO] Completed...")