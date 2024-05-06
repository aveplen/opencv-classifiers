import os
import sys
import time
import pathlib

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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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


def prepare_classifiers():
  return {
    "cifar-10": [
      # SVMClassifier(
      #   "cifar-10", 
      #   "./benchmark/cifar10_labels.txt", 
      #   "./models/svm_cifar10_12288_f32.dat", 
      #   64, 64
      # ),
      # HOGSVMClassifier(
      #   "cifar-10", 
      #   "./benchmark/cifar10_labels.txt", 
      #   "./models/hog_svm_cifar10_f32.dat", 
      #   64, 64
      # ),
      # CNNSVMClassifier(
      #   "cifar-10", 
      #   "./benchmark/cifar10_labels.txt", 
      #   "./models/cnn_svm_cifar10_12288_f32.dat", 
      #   "./models/resnet50_feature_extractor_cifar10_64_64_3.onnx", 
      #   64, 64
      # ),
      ResnetClassifier(
        "cifar-10", 
        "./benchmark/cifar10_labels.txt", 
        "./models/finetuned_resnet50_cifar10_64_64_3.onnx", 
        64, 64
      )
    ],
    # "groceries": [
    #   SVMClassifier(
    #     "groceries", 
    #     "./benchmark/groceries_labels.txt", 
    #     "./models/svm_groceries_37632_f32.dat", 
    #     112, 112
    #   )
    # ]
  }


def precision(conf_matr):
  class_precision = {}

  for actual_class in conf_matr:
    true_positive = 0
    false_positive = 0
    
    for pred_class in conf_matr[actual_class]:
      if pred_class == actual_class:
        true_positive += conf_matr[actual_class][pred_class]
        continue

      false_positive += conf_matr[actual_class][pred_class]

    class_precision[pred_class] = true_positive / (true_positive + false_positive)

  total_precision = 0.0
  for clazz in class_precision:
    total_precision += class_precision[clazz] / len(class_precision)

  return (total_precision, class_precision)


def recall(conf_matr):
  class_recall = {}

  false_negative = {}
  for clazz in conf_matr:
    false_negative[clazz] = 0

  for actual_class in conf_matr:
    for pred_class in conf_matr[actual_class]:
      if actual_class != pred_class:
        false_negative[pred_class] += conf_matr[actual_class][pred_class]

  for actual_class in conf_matr:
    class_true_positive = conf_matr[actual_class][actual_class]
    class_false_negative = false_negative[actual_class]
    class_recall[actual_class] = class_true_positive / (class_true_positive + class_false_negative)

  total_recall = 0.0
  for clazz in class_recall:
    total_recall += class_recall[clazz] / len(class_recall)

  return (total_recall, class_recall)


def f1(conf_matr):
  _, class_precision = precision(conf_matr)
  _, class_recall = recall(conf_matr)

  class_f1 = {}

  for clazz in class_precision:
    prd_ = 2 * class_precision[clazz] * class_recall[clazz]
    sum_ = class_precision[clazz] + class_recall[clazz]

    if sum_ == 0:
      class_f1[clazz] = 0.0
      continue
    
    class_f1[clazz] =  prd_/sum_

  total_f1 = 0.0
  for clazz in class_f1:
    total_f1 += class_f1[clazz] / len(class_f1)

  return (total_f1, class_f1)


def percentile(perc, timings):
  time_seq_list = list(reversed(sorted(timings)))
  sub_seq_len = int(len(time_seq_list)*(1 - perc/100))
  return min(time_seq_list[:sub_seq_len])


def main(argv):
  benchmark_pics_path = ""

  if len(argv) > 1:
    benchmark_pics_path = argv[1]

  if benchmark_pics_path == "":
    benchmark_pics_path = "./benchmark"

  benchmark_pics = pathlib.Path(benchmark_pics_path)
  benchmark_suits = []
  for p in benchmark_pics.iterdir():
    if p.is_dir():
      benchmark_suits.append(p)
  
  classifiers = prepare_classifiers()
  report = {}

  
  for suit in benchmark_suits:
    if suit.name not in classifiers:
      continue

    suit_classifiers: list[Classifier] = classifiers[suit.name]

    for classifier in suit_classifiers:
      confusion_matrix = {}
      positive = []
      timing = []

      for class_dir_1 in suit.iterdir():
        cm_line = class_dir_1.name
        confusion_matrix[cm_line] = {}
        for class_dir_2 in suit.iterdir():
          cm_col = class_dir_2.name
          confusion_matrix[cm_line][cm_col] = 0

      for class_dir in suit.iterdir():
        expected_label = class_dir.name.lower()

        for img_file in class_dir.iterdir():
          image = cv2.imread(img_file.as_posix(), cv2.IMREAD_COLOR)
          res = classifier.classify(image)
          
          confusion_matrix[expected_label][res.label] += 1
          positive.append(expected_label == res.label)
          timing.append(res.timing)

      total_precision, class_precision = precision(confusion_matrix)
      total_recall, class_recall = recall(confusion_matrix)
      total_f1, class_f1 = f1(confusion_matrix)

      timing_99 = percentile(99, timing) * 1000
      timing_95 = percentile(95, timing) * 1000
      timing_50 = percentile(50, timing) * 1000

      if classifier.get_dataset() not in report:
        report[classifier.get_dataset()] = {}

      report[classifier.get_dataset()][classifier.get_type()] = {
        "total_precision": total_precision,
        "class_precision": class_precision,
        "total_recall": total_recall,
        "class_recall": class_recall,
        "total_f1": total_f1,
        "class_f1": class_f1,
        "timing_99": timing_99,
        "timing_95": timing_95,
        "timing_50": timing_50,
      }

  print(report)


if __name__ == "__main__":
  main(sys.argv)