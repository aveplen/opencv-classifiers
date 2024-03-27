import argparse
import random
import time

import onnxruntime as rt
import numpy as np
import cv2 as cv

from enum import Enum
from pathlib import Path
from scipy.special import softmax


__default_test_config = [
    {
        "test_name": "SVM Cifar10 (64, 64, 3) - load and predict",
        "type": "model",
        "model_path": "learning/svm_cifar10_64_64_3.onnx",
        "labels_path": "learning/cifar10_labels.txt",
        "images_root": "learning/cifar-10",
        "image_width": 64,
        "image_height": 64,
        "need_flatten": True,
        "need_transpose": False,
        "need_softmax": False,
        "input_is_bulk": False,
        "output_is_bulk": False,
    },
    {
        "test_name": "SVM Groceries (224, 224, 3) - load and predict",
        "type": "model",
        "model_path": "learning/svm_groceries_224_224_3.onnx",
        "labels_path": "learning/groceries_labels.txt",
        "images_root": "learning/groceries",
        "image_width": 224,
        "image_height": 224,
        "need_flatten": True,
        "need_transpose": False,
        "need_softmax": False,
        "input_is_bulk": False,
        "output_is_bulk": False,
    },
    # todo: hog svm groceries
    # todo: hog svm cifar10
    {
        "test_name": "CNN-SVM Groceries (224, 224, 3) - load and predict",
        "type": "pipeline",
        "labels_path": "learning/groceries_labels.txt",
        "images_root": "learning/groceries",
        "image_width": 224,
        "image_height": 224,
        "need_flatten": False,
        "need_transpose": False,
        "need_softmax": False,
        "stages": [
            {
                "input_data_type": "double",
                "input_is_bulk": True,
                "output_is_bulk": False,
                "model_path": "learning/resnet50_feature_extractor_groceries_224_224_3.onnx"
            },
            {
                "input_data_type": "float",
                "input_is_bulk": False,
                "output_is_bulk": False,
                "model_path": "learning/cnn_feature_svm_groceries_224_224_3.onnx"
            }
        ]
    },
]


def assert_path_exists(filename, implication=None):
    path = Path(filename)
    message = f"Path {filename} doesn't exist"
    if implication is not None:
        message = f"Path ({implication}) {filename} doesn't exist"
    assert path.exists(), message


# =============================================================================


def do_tests(test_configs):
    ran = []
    success = []
    fail = []

    print("Total tests found:", len(test_configs))
    for i, test_config in enumerate(test_configs):
        try:
            test_name = test_config["test_name"]
            print(f"Running test '{test_name}'")
            ran.append(test_config)
            start = time.time()
            result = do_test(test_config)
            end = time.time()
            print(f"🟩 Test {i+1} passed in {end - start}, result: {result}")
            success.append(test_config)
        except BaseException as e:
            print(f"🟥 Test {i} failed, reason: {e}")
            fail.append(test_config)

    print("=========================================================")
    print("Total tests started:", len(ran))
    print()

    print("🟩 Tests succeded:", len(success))
    for test_config in success:
        print("  -", test_config["test_name"])
    print()

    print("🟥 Tests failed:", len(fail))
    for test_config in fail:
        print("  -", test_config["test_name"])


class ClassifierStage:
    def __init__(self, stage_config):
        self._session = self._acquire_session(stage_config["model_path"])
        self._input_data_type = self._decode_input_data_type(stage_config["input_data_type"])
        self._input_is_bulk = stage_config["input_is_bulk"]
        self._output_is_bulk = stage_config["output_is_bulk"]


    def stage_predict(self, input):
        model_input = np.asarray(input.astype(self._input_data_type))

        if self._input_is_bulk:
            model_input = np.asarray([model_input])

        input_name = self._session.get_inputs()[0].name
        model_output = self._session.run(None, {input_name: model_input})[0][0]

        if self._output_is_bulk:
            model_output = model_output[0]

        return model_output

    
    def _decode_input_data_type(self, input_data_type):
        if input_data_type == "float":
            return np.float32
        if input_data_type == "double":
            return np.float64
        raise ValueError(f"Unknown data type: {input_data_type}")


    def _acquire_session(self, model_path):
        assert_path_exists(model_path)
        session = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        assert session is not None, "Acquired session is None"
        return session        


class Classifier:
    def __init__(self, classifier_config):
        classifier_type = classifier_config["type"]

        if classifier_type == "model":
            self._stages = [ClassifierStage(classifier_config)]
            return
        
        if classifier_type == "pipeline":
            self._stages = []
            for stage_config in classifier_config["stages"]:
                self._stages.append(ClassifierStage(stage_config))
            return
        
        raise ValueError(f"Unknown classsifier type: {classifier_type}")

    def predict(self, input):
        output = input
        for stage in self._stages:
            output = stage.stage_predict(output)
        return output


def do_test(test_config):
    classifier = Classifier(test_config)
    labels = collect_labels(test_config["labels_path"])
    (sample_image, sample_path) = load_sample_image(test_config["images_root"])
    image_size = (test_config["image_width"], test_config["image_height"])
    need_flatten = test_config["need_flatten"]
    need_transpose = test_config["need_transpose"]
    need_softmax = test_config["need_softmax"]

    preprocessed_image = preprocess_image(sample_image, image_size, need_flatten, need_transpose)
    classifier_result = classifier.predict(preprocessed_image)
    predicted_label = decode_result(classifier_result, need_softmax, labels)
    return {"predicted_label": predicted_label, "sample_path": sample_path}


def acquire_session(model_path):
    assert_path_exists(model_path)
    session = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    assert session is not None, "Acquired session is None"
    return session


def collect_labels(labels_path):
    assert_path_exists(labels_path)
    labels = []
    with open(labels_path, "r") as labels_file:
        for class_index, label in enumerate(labels_file.readlines()):
            labels.append({"class": class_index, "label": label.strip()})
    assert len(labels) != 0, "Collected zero labels"
    return labels


def load_sample_image(images_root):
    assert_path_exists(images_root)
    images_root_path = Path(images_root)
    assert images_root_path.is_dir(), f"Images root '{images_root}' is not directory"
    class_images_dirs = [dir for dir in images_root_path.iterdir()]

    images_paths = []
    for class_images_dir in class_images_dirs:
        for image_path in class_images_dir.iterdir():
            images_paths.append(image_path)

    sample_img_idx = random.randint(0, len(images_paths))
    sample_image_path = images_paths[sample_img_idx]
    return (cv.imread(f"{sample_image_path}", cv.IMREAD_COLOR), sample_image_path)


def preprocess_image(sample_image, image_size, need_flatten, need_transpose):
    # normalize
    sample_image = sample_image / 255.0

    # resize to sane dimensions
    sample_image = cv.resize(sample_image, (256, 256))

    # crop center
    h, w = sample_image.shape[0], sample_image.shape[1]
    x0 = (w - image_size[0]) // 2
    y0 = (h - image_size[1]) // 2
    sample_image = sample_image[y0 : y0 + image_size[1], x0 : x0 + image_size[0], :]

    # subtract mean
    sample_image = (sample_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # flatten (if needed)
    if need_flatten:
        sample_image = np.array(sample_image).flatten()

    # transpose (if needed)
    if need_transpose:
        sample_image = np.transpose(sample_image, axes=[2, 0, 1])

    # turn to float64 (expected by onnx-runtime)
    sample_image = sample_image.astype(np.float64)

    return sample_image


def classify(session, data):
    model_input = np.asarray(data.astype(np.float64))
    input_name = session.get_inputs()[0].name
    model_output = session.run(None, {input_name: model_input})[0][0]
    return model_output


def decode_result(model_output, need_softmax, labels):
    if need_softmax:
        model_output = softmax(model_output)
    return labels[model_output]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_path", required=False)
    parser.add_argument("-labels_path", required=False)
    parser.add_argument("-images_root", required=False)
    parser.add_argument("-image_width", required=False)
    parser.add_argument("-image_height", required=False)
    parser.add_argument("-need_flatten", required=False)
    parser.add_argument("-need_transpose", required=False)
    parser.add_argument("-need_softmax", required=False)
    parser.add_argument("-response_is_bulk", required=False)
    args = parser.parse_args()

    test_config = __default_test_config
    if args.model_path is not None and args.model_path != "default":
        test_config = [
            {
                "model_path": args.model_path,
                "labels_path": args.labels_path,
                "images_root": args.images_root,
                "image_width": int(args.image_width),
                "image_height": int(args.image_height),
                "need_flatten": bool(args.need_flatten),
                "need_transpose": bool(args.need_transpose),
                "need_softmax": bool(args.need_softmax),
                "response_is_bulk": bool(args.response_is_bulk),
            }
        ]

    do_tests(test_config)


if __name__ == "__main__":
    main()
