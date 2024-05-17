import time

from abc import ABC
from enum import Enum

import cv2
import numpy as np


def crop_center(img):
  width, height = img.shape[1], img.shape[0]
  crop_width = min(img.shape[0], img.shape[1])
  crop_height = min(img.shape[0], img.shape[1])
  mid_x, mid_y = int(width/2), int(height/2)
  cw2, ch2 = int(crop_width/2), int(crop_height/2) 
  crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
  return crop_img


class ClassifierType(Enum):
  SVM = 1
  HOG_SVM = 2
  SIFT_KMEANS_SVM = 3
  RESNET_50 = 4
  INCEPTION = 5


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
    label = self._labels[clazz-1]
    return label


class HOGSVMClassifier(Classifier):
  def __init__(self, dataset, labels_path, model_weights_path, image_width, image_height):
    self._labels = HOGSVMClassifier.__load_labels(labels_path)
    self._model = HOGSVMClassifier.__load_model(model_weights_path)
    self._image_width = image_width
    self._image_height = image_height
    self._dataset = dataset
    self._hog = HOGSVMClassifier.__init_hog(image_width, image_height)


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
    cell_size = (width//8, height//8)
    block_size = (width//4, height//4)
    block_stride = (width//8, height//8)
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
    image = crop_center(image)
    image = cv2.resize(image, dimension)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_descriptor = self._hog.compute(image_gray)
    return hog_descriptor.astype(np.float32)


  def __classify_internal(self, descriptor):
    clazz = int(self._model.predict(np.asarray([descriptor]).astype(np.float32))[1][0])
    return clazz


  def __postprocess_result(self, clazz):
    label = self._labels[clazz-1]
    return label


class SIFTkMeansSVMClassifier(Classifier):
  def __init__(self, dataset, labels_path, svm_weights_path, centers_path, image_width, image_height):
    self._labels = SIFTkMeansSVMClassifier.__load_labels(labels_path)
    self._svm = SIFTkMeansSVMClassifier.__load_svm(svm_weights_path)
    self._sift = SIFTkMeansSVMClassifier.__load_sift()
    self._centers = SIFTkMeansSVMClassifier.__load_centers(centers_path)
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
  def __load_svm(model_weights_path):
    return cv2.ml.SVM_load(model_weights_path)
  
  @staticmethod
  def __load_sift():
    return cv2.SIFT_create()
  
  @staticmethod
  def __load_centers(kmeans_weights_path):
    return np.load(kmeans_weights_path)
  

  def get_dataset(self) -> str:
    return self._dataset
  

  def get_type(self) -> ClassifierType:
    return ClassifierType.SIFT_KMEANS_SVM
  

  def classify(self, image) -> ClassificationResult:
    start = time.time()
    data = self.__preprocess_image(image)
    y_pred = self.__classify_internal(data)
    label = self.__postprocess_result(y_pred)
    timing = time.time() - start
    return ClassificationResult(label, timing)


  def __calc_histogram(self, sift_descriptor):
    histogram = np.zeros(len(self._centers))
    if sift_descriptor is None:
      return histogram
    
    for descriptor in sift_descriptor:
        distances = np.linalg.norm(descriptor - self._centers, axis=1)
        nearest_cluster_index = np.argmin(distances)
        histogram[nearest_cluster_index] += 1

    return histogram


  def __preprocess_image(self, image):
    image = crop_center(image)
    image = cv2.resize(image, (self._image_width, self._image_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = self._sift.detectAndCompute(image, None)
    hist = self.__calc_histogram(descriptors)
    return np.asarray(hist).astype(np.float32)


  def __classify_internal(self, descriptor):
    clazz = int(self._svm.predict(np.asarray([descriptor]))[1][0])
    return clazz


  def __postprocess_result(self, clazz):
    label = self._labels[clazz-1]
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
    image = crop_center(image)
    dimension = (self._image_width, self._image_height)
    image = cv2.resize(image, dimension)
    blob = cv2.dnn.blobFromImage(image, 1.0, dimension, (104, 117, 123))
    blob = np.transpose(blob, (0, 2, 3, 1))
    return blob.astype(np.float32)


  def __classify_internal(self, blob):
    self._model.setInput(blob)
    result = self._model.forward()[0]
    clazz = np.argmax(result)
    return clazz


  def __postprocess_result(self, clazz):
    label = self._labels[clazz-1]
    return label


class InceptionClassifier(Classifier):
  def __init__(
      self, 
      dataset, 
      labels_path, 
      model_onnx_path, 
      image_width, 
      image_height
    ):
    self._labels = InceptionClassifier.__load_labels(labels_path)
    self._model = InceptionClassifier.__load_model(model_onnx_path)
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
    return ClassifierType.INCEPTION
  

  def classify(self, image) -> ClassificationResult:
    start = time.time()
    data = self.__preprocess_image(image)
    y_pred = self.__classify_internal(data)
    label = self.__postprocess_result(y_pred)
    timing = time.time() - start
    return ClassificationResult(label, timing)


  def __preprocess_image(self, image):
    image = crop_center(image)
    dimension = (self._image_width, self._image_height)
    image = cv2.resize(image, dimension)
    blob = cv2.dnn.blobFromImage(image, 1.0, dimension, (104, 117, 123))
    blob = np.transpose(blob, (0, 2, 3, 1))
    return blob.astype(np.float32)


  def __classify_internal(self, blob):
    self._model.setInput(blob)
    result = self._model.forward()[0]
    clazz = np.argmax(result)
    return clazz


  def __postprocess_result(self, clazz):
    label = self._labels[clazz-1]
    return label


def init_cifar_svm():
  return SVMClassifier(
    "cifar-10", 
    "./benchmark/cifar10_labels.txt", 
    "./new_models/svm_cifar10_32x32_f32.dat", 
    32, 32
  )

def init_groceries_svm():
  return SVMClassifier(
    "groceries", 
    "./benchmark/groceries_labels.txt", 
    "./new_models/svm_groceries_112x112_f32.dat", 
    112, 112
  )

def init_petimages_svm():
  return SVMClassifier(
    "petimages", 
    "./benchmark/petimages_labels.txt", 
    "./new_models/svm_petimages_100x100_f32.dat", 
    100, 100
  )

def init_cifar_hog_svm():
  return HOGSVMClassifier(
    "cifar-10", 
    "./benchmark/cifar10_labels.txt", 
    "./new_models/hog_svm_cifar10_32x32_f32.dat", 
    32, 32
  )

def init_groceries_hog_svm():
  return HOGSVMClassifier(
    "groceries", 
    "./benchmark/groceries_labels.txt", 
    "./new_models/hog_svm_groceries_224x224_f32.dat", 
    224, 224
  )

def init_petimages_hog_svm():
  return HOGSVMClassifier(
    "petimages", 
    "./benchmark/petimages_labels.txt", 
    "./new_models/hog_svm_petimages_150x150_f32.dat", 
    160, 160
  )

def init_cifar_sift_kmeans_svm():
  return SIFTkMeansSVMClassifier(
    "cifar-10", 
    "./benchmark/cifar10_labels.txt", 
    "./new_models/sift_kmeans_svm_cifar_32x32_f32.dat", 
    "./new_models/sift_kmeans_svm_cifar_32x32_f32_centers.npy",
    32, 32
  )

def init_groceries_sift_kmeans_svm():
  return SIFTkMeansSVMClassifier(
    "groceries",
    "./benchmark/groceries_labels.txt", 
    "./new_models/sift_kmeans_svm_groceries_224x224_f32.dat", 
    "./new_models/sift_kmeans_svm_groceries_224x224_f32_centers.npy",
    224, 224
  )


def init_petimages_sift_kmeans_svm():
  return SIFTkMeansSVMClassifier(
    "petimages", 
    "./benchmark/petimages_labels.txt", 
    "./new_models/sift_kmeans_svm_petimages_150x150_f32.dat", 
    "./new_models/sift_kmeans_svm_petimages_150x150_f32_centers.npy", 
    150, 150
  )

def init_imagenet_resnet():
  return ResnetClassifier(
    "imagenet",
    "./benchmark/imagenet_labels.txt", 
    "./new_models/resnet_224x224_f64.onnx", 
    224, 224
  )

def init_imagenet_inception():
  return InceptionClassifier(
    "imagenet",
    "./benchmark/imagenet_labels.txt", 
    "./new_models/inception_299x299_f64.onnx", 
    299, 299
  )


MODEL_ID_CLASSIFIER = {
  "cifar-10-svm": init_cifar_svm,
  "cifar-10-hog-svm": init_cifar_hog_svm,
  "cifar-10-sift-kmeans-svm": init_cifar_sift_kmeans_svm,

  "groceries-svm": init_groceries_svm,
  "groceries-hog-svm": init_groceries_hog_svm,
  "groceries-sift-kmeans-svm": init_groceries_sift_kmeans_svm,

  "petimages-svm": init_petimages_svm,
  "petimages-hog-svm": init_petimages_hog_svm,
  "petimages-sift-kmeans-svm": init_petimages_sift_kmeans_svm,

  "imagenet-resnet": init_imagenet_resnet,
  "imagenet-inception": init_imagenet_inception,
}