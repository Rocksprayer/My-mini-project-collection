import tensorflow as tf
import numpy as np
from tensorflow import keras
import cv2 as cv


print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
img=cv.imread('1.png',0)
resized=cv.resize(img,[28,28], interpolation = cv.INTER_AREA)
A=resized[None,:,:]
classifications = model.predict(A)
print('shape',resized.shape)
print(test_labels[0])
shoe=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,1])
if set(classifications[0])==set(shoe):
    print ("yep it is a shoe")
else:
    print("this one is not a shoes ")
cv.imshow('resize',resized)
cv.waitKey()






# fashion_mnist=keras.datasets.fashion_mnist
# (train_images, train_lables),(test_images,test_labels)=fashion_mnist.load_data()
# model=keras.Sequential([
#     #analyze pixel by pixel
#     keras.layers.Flatten(input_shape=(28,28)),
#     #128 function to define characteristic if they combine in someway
#     # in this case numeric adding and equal lables 9 its a shoes
#     keras.layers.Dense(128,activation=tf.nn.relu),#if x>0 f=x else f=0
#     #picking number with highest in output set it to 1 and others to 0
#     keras.layers.Dense(10,activation=tf.nn.softmax)
# ])
# #Adam generate ramdom values from begin ning and regenerates values base on loss fuctions
#
# model.compile(optimizer='adam',
#               loss='sqarse_category_crossentropy')
# model.fit(train_images,train_lables,epochs=5)
# test_loss,test_acc=model.evaluate(test_images,test_labels)
# img=cv.imread('1.png')
# resized=cv.resize(img,[28,28], interpolation = cv.INTER_AREA)
# predictions=model.predict(resized)
