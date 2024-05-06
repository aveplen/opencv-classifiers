import sys
import pathlib
import cv2

import lib.classifiers as clf


def main(argv):
  if len(argv) != 3:
    print("Expected exactly 2 arguments: <model-id> <image.png>")
    sys.exit(1)

  model_id = argv[1]
  image_filename = argv[2]

  if model_id not in clf.MODEL_ID_CLASSIFIER:
    print(f"Unknown model-id '{model_id}', list of known model-id's:")
    for known_model_id in clf.MODEL_ID_CLASSIFIER:
      print(f"  - {known_model_id}")
    sys.exit(1)
  
  if not pathlib.Path(image_filename).exists():
    print(f"File with name '{image_filename}' not found")
    sys.exit(1)
  
  classifier: clf.Classifier = clf.MODEL_ID_CLASSIFIER[model_id]()
  image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
  result = classifier.classify(image)

  print(f"Image '{image_filename}' was classified as '{result.label}'")
  print()
  print(f"Classification took {result.timing*1000} ms")
  print("All the possible labels were:")
  for label in classifier._labels:
    print(f"  - {label}")
  sys.exit(0)


if __name__ == "__main__":
  main(sys.argv)