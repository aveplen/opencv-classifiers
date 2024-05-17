import cv2
import sys
import pathlib
from typing import List

import lib.classifiers as clf


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
  
  report = {}
  classifiers = {
    "cifar-10": [
      clf.init_cifar_svm(),
      clf.init_cifar_hog_svm(),
      clf.init_cifar_sift_kmeans_svm(),
    ],
    "groceries": [
      clf.init_groceries_svm(),
      clf.init_groceries_hog_svm(),
      clf.init_groceries_sift_kmeans_svm(),
    ],
    "petimages": [
      clf.init_petimages_svm(),
      clf.init_petimages_hog_svm(),
      clf.init_petimages_sift_kmeans_svm(),
    ],
    "imagenette": [
      clf.init_imagenet_resnet(),
      clf.init_imagenet_inception(),
    ]
  }
  
  for suit in benchmark_suits:
    if suit.name not in classifiers:
      continue

    suit_classifiers: List[clf.Classifier] = classifiers[suit.name]

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