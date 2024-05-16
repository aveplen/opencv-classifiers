import time

from abc import ABC
from enum import Enum

import cv2
import numpy as np


class ClassifierType(Enum):
  SVM = 1
  HOG_SVM = 2
  CNN_SVM = 3
  RESNET_50 = 4


class ClassificationResult:
  def __init__(self, label, timing):
    self.label = label
    self.timing = timing


class Classifier(ABC):
  def get_dataset(self) -> str:
    raise NotImplemented()

  def get_type(self) -> ClassifierType:
    raise NotImplemented()
  
  def classify(self, image) -> ClassificationResult:
    raise NotImplemented()


class SVMClassifier(Classifier):
  def __init__(self, dataset, labels_path, model_weights_path, image_width, image_height):
    self._labels = SVMClassifier.__load_labels(labels_path)
    self._model = SVMClassifier.__load_model(model_weights_path)
    self._image_width = image_width
    self._image_height = image_height
    self._dataset = dataset


  @staticmethod
  def __load_labels(labels_path):
    with open(labels_path, "r") as labels_file:
      labels = []

      for line in labels_file.readlines():
        labels.append(line.strip())

    return labels
  

  @staticmethod
  def __load_model(model_weights_path):
    return cv2.ml.SVM_load(model_weights_path)
  

  def get_dataset(self) -> str:
    return self._dataset
  

  def get_type(self) -> ClassifierType:
    return ClassifierType.SVM
  

  def classify(self, image) -> ClassificationResult:
    start = time.time()
    data = self.__preprocess_image(image)
    y_pred = self.__classify_internal(data)
    label = self.__postprocess_result(y_pred)
    timing = time.time() - start
    return ClassificationResult(label, timing)


  def __preprocess_image(self, image):
    dimension = (self._image_width, self._image_height)
    image = cv2.resize(image, dimension)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image / 255.0) * 2 - 1
    image = image.flatten()
    return image.astype(np.float32)


  def __classify_internal(self, image):
    clazz = int(self._model.predict(np.asarray([image]))[1][0])
    return clazz


  def __postprocess_result(self, clazz):
    label = self._labels[clazz]
    return label
  

class HOGSVMClassifier(Classifier):
  def __init__(self, dataset, labels_path, model_weights_path, image_width, image_height):
    self._labels = HOGSVMClassifier.__load_labels(labels_path)
    self._model = HOGSVMClassifier.__load_model(model_weights_path)
    self._image_width = image_width
    self._image_height = image_height
    self._dataset = dataset
    self._hog = self.__init_hog(image_width, image_height)


  @staticmethod
  def __load_labels(labels_path):
    with open(labels_path, "r") as labels_file:
      labels = []

      for line in labels_file.readlines():
        labels.append(line.strip())

    return labels
  

  @staticmethod
  def __load_model(model_weights_path):
    return cv2.ml.SVM_load(model_weights_path)
  

  @staticmethod
  def __init_hog(width, height):
    win_size = (width, height)
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    return hog
  

  def get_dataset(self) -> str:
    return self._dataset
  

  def get_type(self) -> ClassifierType:
    return ClassifierType.HOG_SVM
  

  def classify(self, image) -> ClassificationResult:
    start = time.time()
    data = self.__preprocess_image(image)
    y_pred = self.__classify_internal(data)
    label = self.__postprocess_result(y_pred)
    timing = time.time() - start
    return ClassificationResult(label, timing)


  def __preprocess_image(self, image):
    dimension = (self._image_width, self._image_height)
    image = cv2.resize(image, dimension)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_descriptor = self._hog.compute(image)
    return hog_descriptor.astype(np.float32)


  def __classify_internal(self, descriptor):
    clazz = int(self._model.predict(np.asarray([descriptor]))[1][0])
    return clazz


  def __postprocess_result(self, clazz):
    label = self._labels[clazz]
    return label
  

class CNNSVMClassifier(Classifier):
  def __init__(
      self, 
      dataset, 
      labels_path, 
      model_weights_path, 
      preprocessor_onnx_path, 
      image_width, 
      image_height
    ):
    self._labels = CNNSVMClassifier.__load_labels(labels_path)
    self._model = CNNSVMClassifier.__load_model(model_weights_path)
    self._preprocessor = CNNSVMClassifier.__load_preprocessor(preprocessor_onnx_path)
    self._image_width = image_width
    self._image_height = image_height
    self._dataset = dataset


  @staticmethod
  def __load_labels(labels_path):
    with open(labels_path, "r") as labels_file:
      labels = []

      for line in labels_file.readlines():
        labels.append(line.strip())

    return labels
  

  @staticmethod
  def __load_model(model_weights_path):
    return cv2.ml.SVM_load(model_weights_path)
  

  @staticmethod
  def __load_preprocessor(preprocessor_onnx_path):
    return cv2.dnn.readNetFromONNX(preprocessor_onnx_path)
  

  def get_dataset(self) -> str:
    return self._dataset
  

  def get_type(self) -> ClassifierType:
    return ClassifierType.CNN_SVM
  

  def classify(self, image) -> ClassificationResult:
    start = time.time()
    data = self.__preprocess_image(image)
    y_pred = self.__classify_internal(data)
    label = self.__postprocess_result(y_pred)
    timing = time.time() - start
    return ClassificationResult(label, timing)


  def __preprocess_image(self, image):
    dimension = (self._image_width, self._image_height)
    image = cv2.resize(image, dimension)
    blob = cv2.dnn.blobFromImage(image, 1.0, (64, 64), (104, 117, 123))
    blob = np.transpose(blob, (0, 2, 3, 1))
    self._preprocessor.setInput(blob)
    feature_vector = self._preprocessor.forward()
    return feature_vector[0].astype(np.float32)


  def __classify_internal(self, descriptor):
    clazz = int(self._model.predict(np.asarray([descriptor]))[1][0])
    return clazz


  def __postprocess_result(self, clazz):
    label = self._labels[clazz]
    return label
  

class ResnetClassifier(Classifier):
  def __init__(
      self, 
      dataset, 
      labels_path, 
      model_onnx_path, 
      image_width, 
      image_height
    ):
    self._labels = ResnetClassifier.__load_labels(labels_path)
    self._model = ResnetClassifier.__load_model(model_onnx_path)
    self._image_width = image_width
    self._image_height = image_height
    self._dataset = dataset


  @staticmethod
  def __load_labels(labels_path):
    with open(labels_path, "r") as labels_file:
      labels = []

      for line in labels_file.readlines():
        labels.append(line.strip())

    return labels
  

  @staticmethod
  def __load_model(model_onnx_path):
    return cv2.dnn.readNetFromONNX(model_onnx_path)
  

  def get_dataset(self) -> str:
    return self._dataset
  

  def get_type(self) -> ClassifierType:
    return ClassifierType.RESNET_50
  

  def classify(self, image) -> ClassificationResult:
    start = time.time()
    data = self.__preprocess_image(image)
    y_pred = self.__classify_internal(data)
    label = self.__postprocess_result(y_pred)
    timing = time.time() - start
    return ClassificationResult(label, timing)


  def __preprocess_image(self, image):
    dimension = (self._image_width, self._image_height)
    image = cv2.resize(image, dimension)
    blob = cv2.dnn.blobFromImage(image, 1.0, (64, 64), (104, 117, 123))
    blob = np.transpose(blob, (0, 2, 3, 1))
    return blob.astype(np.float32)


  def __classify_internal(self, blob):
    self._model.setInput(blob)
    result = self._model.forward()[0]
    clazz = np.argmax(result)
    return clazz


  def __postprocess_result(self, clazz):
    label = self._labels[clazz]
    return label
  

def init_cifar_svm():
  return SVMClassifier(
    "cifar-10", 
    "./benchmark/cifar10_labels.txt", 
    "./models/svm_cifar10_12288_f32.dat", 
    64, 64
  )

def init_cifar_hog_svm():
  return HOGSVMClassifier(
    "cifar-10", 
    "./benchmark/cifar10_labels.txt", 
    "./models/hog_svm_cifar10_f32.dat", 
    64, 64
  )

def init_cifar_cnn_svm():
  return CNNSVMClassifier(
    "cifar-10", 
    "./benchmark/cifar10_labels.txt", 
    "./models/cnn_svm_cifar10_12288_f32.dat", 
    "./models/resnet50_feature_extractor_cifar10_64_64_3.onnx", 
    64, 64
  )

def init_cifar_resnet():
  return ResnetClassifier(
    "cifar-10", 
    "./benchmark/cifar10_labels.txt", 
    "./models/finetuned_resnet50_cifar10_64_64_3.onnx", 
    64, 64
  )

def init_groceries_svm():
  return SVMClassifier(
    "groceries", 
    "./benchmark/groceries_labels.txt", 
    "./models/svm_groceries_37632_f32.dat", 
    112, 112
  )


MODEL_ID_CLASSIFIER = {
  "cifar-10-svm": init_cifar_svm,
  "cifar-10-hog-svm": init_cifar_hog_svm,
  "cifar-10-cnn-svm": init_cifar_cnn_svm,
  "cifar-10-resnet": init_cifar_resnet,
  "groceries-svm": init_groceries_svm,
}