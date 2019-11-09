import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pickle
import os
import numpy as np
import keras
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.preprocessing.image import img_to_array, array_to_img

np.set_printoptions(precision=2, suppress=True)


def f1():
    def indexAllImage(directory):
        x = np.empty((0, 256, 256, 3))
        a = os.listdir(directory)
        counter = 0
        for i in a:
            counter += 1
            print(counter, i)
            path = directory + i
            image = load_img(path, target_size=(256, 256))  # target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            x = np.append(x, image, axis=0)
        return x

    # x = indexAllImage('C:/Users/stoun/PycharmProjects/untitled3/BeagleTest/')
    # x = x.astype('float32')                                     # корректируем данные в диапозон от 0.0 до 1.0
    # x /= 255                                                    # (до этого данные в дапозоне от 0.0 до 255.0)
    # with open('C:/Users/stoun/PycharmProjects/untitled3/beagleTest256.pickle', 'wb') as f:
    #     pickle.dump(x, f)
    # # print(x.shape, x)

    # with open('C:/Users/stoun/PycharmProjects/untitled3/samoedTest256.pickle', 'rb') as f:
    #     data_new1 = pickle.load(f)
    # print(data_new1.shape)
    # with open('C:/Users/stoun/PycharmProjects/untitled3/beagleTest256.pickle', 'rb') as f:
    #     data_new2 = pickle.load(f)
    # print(data_new2.shape)
    # x = np.append(data_new1, data_new2, axis=0)
    # print(x.shape)
    # with open('C:/Users/stoun/PycharmProjects/untitled3/test256.pickle', 'wb') as f:
    #     pickle.dump(x, f)


# f1()


# (414, 64, 64, 3)    sam
# (461, 64, 64, 3)    bgl
# (875, 64, 64, 3)   all
# x = np.append(np.tile(0, 414), np.tile(1, 461))
# print(x[541])

def f2():
    image = load_img('C:/Users/stoun/PycharmProjects/untitled3/data/train/Sm/171.jpg',
                     target_size=(160, 160))  # target_size=(224, 224))
    image = img_to_array(image)
    x = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    x = x.astype('float32')
    x /= 255
    model = keras.models.load_model('C:/Users/stoun/PycharmProjects/untitled3/model94_160.h5')
    predict = model.predict(x)  # предсказать для тестового изображения
    class_result = np.argmax(predict, axis=1)
    print(predict, class_result)


# f2()


def f3():
    # def indexAllImage(directory):
    #     x = np.empty((0, 64, 64, 3))
    #     a = os.listdir(directory)
    #     for i in a:
    #         print(i)
    #         path = directory + i
    #         image = load_img(path, target_size=(64, 64))  # target_size=(224, 224))
    #         image = img_to_array(image)
    #         image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #         x = np.append(x, image, axis=0)
    #     return x
    # x = indexAllImage('C:/Users/stoun/PycharmProjects/untitled3/SamoedTest/')
    # x = x.astype('float32')
    # x /= 255
    # with open('C:/Users/stoun/PycharmProjects/untitled3/SamoedTest/samoedTest.pickle', 'rb') as f:
    #     data_new1 = pickle.load(f)
    # print(data_new1.shape)
    # with open('C:/Users/stoun/PycharmProjects/untitled3/BeagleTest/beagleTest.pickle', 'rb') as f:
    #     data_new2 = pickle.load(f)
    # print(data_new2.shape)
    # x = np.append(data_new1, data_new2, axis=0)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(160, 160),
        batch_size=300,
        class_mode='categorical',
        shuffle=False
    )
    x, y = validation_generator.next()
    # for i in range(0, 1):
    #     image = x[i]
    #     plt.imshow(image)
    #     plt.show()
    print(y)
    print(x.shape)

    # # d = {'data': x, 'labels': np.append(np.tile(0, 100), np.tile(1, 100)), 'name_labels': ['Samoed', 'Beagle']}
    model = keras.models.load_model('C:/Users/stoun/PycharmProjects/untitled3/model.h5')
    predict = model.predict(x)  # предсказать для тестового изображения
    # print(predict, predict.reshape(1, -1))
    # class_result = np.rint(predict)
    # class_result = class_result.reshape(-1, )
    class_result = np.argmax(predict, axis=1)
    print(y, class_result)
    # print(predict, class_result, class_result.sum())
    # print(classification_report(np.append(np.tile(0, 100), np.tile(1, 100)), class_result,
    #                             target_names=['Samoed', 'Beagle']))
    y = np.append(np.tile(0, 100), np.tile(1, 100))
    y = np.append(y, np.tile(2, 100))
    print(classification_report(y, class_result))
    # print(model.summary())


# f3()


def f4():
    def downScale(directory):
        a = os.listdir(directory)
        counter = 0
        for i in a:
            if i.find('.jpg') != -1:
                counter += 1
                print(counter, i)
                path = directory + i
                image = load_img(path, target_size=(256, 256))  # target_size=(224, 224))
                image = img_to_array(image)
                path = 'C:/Users/stoun/PycharmProjects/untitled3/data/train/Gr/'  # TO
                array_to_img(image).save(fp=path + i)

    downScale('C:/Users/stoun/PycharmProjects/newInstImgGERMAN/')  # FROM

# f4()


# x, y = train_generator.next()
# for i in range(0, 1):
#     image = x[i]
#     plt.imshow(image)
#     plt.show()
