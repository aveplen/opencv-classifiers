import os
import sys
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

      for line in labels_file.readlines:
        labels.append(line)

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
    image_resized = cv2.resize(image, dimension, anti_aliasing=True, mode='reflect')
    image_flat = image_resized.flatten()
    return image_flat


  def __classify_internal(self, image):
    clazz = self._model.predict(image)
    return clazz


  def __postprocess_result(self, clazz):
    label = self._labels[clazz]
    return label
  

def prepare_classifiers():
  return {
    "32x32": [
      SVMClassifier(
        "cifar-10", 
        "some_labels_path", 
        "some_model_weights_path", 
        64, 64
      )
    ],
    "224x224": [
      SVMClassifier(
        "groceries", 
        "some_labels_path", 
        "some_model_weights_path", 
        112, 112
      )
    ]
  }


def precision(conf_matr):
  class_precision = {}

  for pred_class in conf_matr:
    true_positive = 0
    false_positive = 0
    
    for actual_class in conf_matr[pred_class]:
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

  for actual_class in conf_matr[pred_class]:
    true_positive = 0
    false_negative = 0

    for pred_class in conf_matr:
      if pred_class == actual_class:
        true_positive += conf_matr[actual_class][pred_class]
        continue

      
      false_negative += conf_matr[actual_class][pred_class]

    class_recall[actual_class] = true_positive / (true_positive + false_negative)

  total_recall = 0.0
  for clazz in class_recall:
    total_recall += class_recall[clazz] / len(class_recall)

  return (total_recall, class_recall)


def f1(conf_matr):
  _, class_precision = precision(conf_matr)
  _, class_recall = recall(conf_matr)

  class_f1 = {}

  for clazz in class_precision:
    class_f1[clazz] = (2 * class_precision[clazz] * class_recall[clazz]) / (class_precision[clazz] + class_recall[clazz])

  total_f1 = 0.0
  for clazz in class_f1:
    total_f1 += class_f1[clazz] / len(class_f1)

  return (total_f1, class_f1)


def percentile(perc, time_seq):
  return max(time_seq[:int(len(time_seq)*(1 - perc/100))])


def main(argv):
  benchmark_pics_path = ""

  if len(argv) > 1:
    benchmark_pics_path = argv[1]

  if benchmark_pics_path == "":
    benchmark_pics_path = "./benchmark"

  benchmark_pics = os.path.Path(benchmark_pics_path)
  benchmark_suits = [directory for directory in benchmark_pics.iterdir()]
  
  classifiers = prepare_classifiers()
  report = {}

  
  for suit in benchmark_suits:
    suit_classifiers: list[Classifier] = classifiers[suit.filename()]

    for classifier in suit_classifiers:
      confusion_matrix = {}
      positive = []
      timing = []

      for class_dir_1 in suit.iterdir():
        cm_line = class_dir_1.dirname()

        confusion_matrix[cm_line] = {}
        for class_dir_2 in suit.iterdir():
          cm_col = class_dir_2.dirname()

          confusion_matrix[cm_line][cm_col] = 0


      for class_dir in suit.iterdir():
        expected_label = class_dir.dirname().lower()

        for img_filename in class_dir.iterdir():
          image = cv2.imread(img_filename, cv2.IMREAD_COLOR)
          res = classifier.classify(image)
          
          confusion_matrix[expected_label][res.label] += 1
          positive.append(expected_label == res.label)
          timing.append(res.timing)


      total_precision, class_precision = precision(confusion_matrix)
      total_recall, class_recall = precision(confusion_matrix)
      total_f1, class_f1 = f1(confusion_matrix)

      timing = reversed(sorted(timing))
      timing_99 = percentile(99, timing)
      timing_95 = percentile(95, timing)
      timing_50 = percentile(50, timing)

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