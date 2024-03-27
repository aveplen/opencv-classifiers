import argparse
import random
import time

import onnxruntime as rt
import numpy as np
import cv2 as cv

from enum import Enum
from pathlib import Path
from scipy.special import softmax


TEST_CONFIG = [
    {
        "test_name": "SVM Cifar10 (64, 64, 3) - load and predict",
        "type": "svm-classifier",
        "weights_path": "learning/svm_cifar10_12288_f32.dat",
        "labels_path": "learning/cifar10_labels.txt",
        "images_root": "learning/cifar-10",
        "image_width": 64,
        "image_height": 64,
        "image_depth": 3,
        "need_flatten": True,
        "need_bgr2rgb": True,
        "need_transpose": False,
        "need_softmax": False,
        "input_is_bulk": True,
        "output_is_bulk": True,
        "input_data_type": "float",
    },
    {
        "test_name": "SVM Groceries (112, 112, 3) - load and predict",
        "type": "svm-classifier",
        "weights_path": "learning/svm_groceries_37632_f32.dat",
        "labels_path": "learning/groceries_labels.txt",
        "images_root": "learning/groceries",
        "image_width": 112,
        "image_height": 112,
        "image_depth": 3,
        "need_flatten": True,
        "need_bgr2rgb": True,
        "need_transpose": False,
        "need_softmax": False,
        "input_is_bulk": True,
        "output_is_bulk": True,
        "input_data_type": "float",
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
        "image_depth": 3,
        "need_bgr2rgb": True,
        "need_flatten": False,
        "need_transpose": False,
        "need_softmax": False,
        "stages": [
            {
                "type": "onnx-model",
                "input_data_type": "double",
                "input_is_bulk": True,
                "output_is_bulk": False,
                "model_path": "learning/resnet50_feature_extractor_groceries_224_224_3.onnx",
            },
            {
                "type": "svm-classifier",
                "input_data_type": "float",
                "input_is_bulk": True,
                "output_is_bulk": True,
                "weights_path": "learning/cnn_svm_groceries_37632_f32.dat",
            },
        ],
    },
    {
        "test_name": "CNN-SVM Groceries (224, 224, 3) - load and predict",
        "type": "pipeline",
        "labels_path": "learning/groceries_labels.txt",
        "images_root": "learning/groceries",
        "image_width": 224,
        "image_height": 224,
        "image_depth": 3,
        "need_bgr2rgb": True,
        "need_flatten": False,
        "need_transpose": False,
        "need_softmax": False,
        "stages": [
            {
                "type": "onnx-model",
                "input_data_type": "double",
                "input_is_bulk": True,
                "output_is_bulk": False,
                "model_path": "learning/resnet50_feature_extractor_groceries_224_224_3.onnx",
            },
            {
                "type": "svm-classifier",
                "input_data_type": "float",
                "input_is_bulk": True,
                "output_is_bulk": True,
                "weights_path": "learning/cnn_svm_groceries_37632_f32.dat",
            },
        ],
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


class ModelType(Enum):
    SVM_CLASSIFIER = "svm-classifier"
    ONNX_MODEL = "onnx-model"
    PIPELINE = "pipeline"


class ClassifierStage:
    def __init__(self, stage_config):
        self._input_data_type = self._decode_input_data_type(stage_config["input_data_type"])
        self._input_is_bulk = stage_config["input_is_bulk"]
        self._output_is_bulk = stage_config["output_is_bulk"]

        stage_type = stage_config["type"]

        if stage_type == "pipeline":
            raise ValueError("Classifier stage could not be a pipeline")

        if stage_type == "svm-classifier":
            self._stage_type = ModelType.SVM_CLASSIFIER
            weights_path = stage_config["weights_path"]
            assert_path_exists(weights_path)
            self._svm_classifier = cv.ml.SVM_load(weights_path)
            return
        
        if stage_type == "onnx-model":
            self._stage_type = ModelType.ONNX_MODEL
            model_path = stage_config["model_path"]
            assert_path_exists(model_path)
            self._session = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            return
    
        raise ValueError(f"Unknown stage type: {stage_type}")


    def _decode_input_data_type(self, input_data_type):
        if input_data_type == "float":
            return np.float32
        if input_data_type == "double":
            return np.float64
        raise ValueError(f"Unknown data type: {input_data_type}")


    def stage_predict(self, input):
        input = self.__stage_preprocess(input)
        output = self.__stage_predict(input)
        output = self.__stage_postprocess(output)
        return output


    def __stage_preprocess(self, input):
        input = np.asarray(input.astype(self._input_data_type))
        if self._input_is_bulk:
            input = np.asarray([input])
        return input


    def __stage_predict(self, input):
        if self._stage_type == ModelType.PIPELINE:
            raise ValueError("Could not __stage_predict() on pipeline")
        
        if self._stage_type == ModelType.ONNX_MODEL:
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: input.astype(self._input_data_type)})[0][0]
            return output

        if self._stage_type == ModelType.SVM_CLASSIFIER:
            output = self._svm_classifier.predict(input.astype(self._input_data_type))[1]
            return output

        raise ValueError(f"Unsupported stage type: {self._stage_type}")


    def __stage_postprocess(self, output):
        if self._output_is_bulk:
            output = output[0]
        return output


class Classifier:
    def __init__(self, classifier_config):
        classifier_type = classifier_config["type"]

        if classifier_type != "pipeline":
            self._stages = [ClassifierStage(classifier_config)]
            return
        
        self._stages = []
        for stage_config in classifier_config["stages"]:
            self._stages.append(ClassifierStage(stage_config))


    def predict(self, input):
        output = input
        for stage in self._stages:
            output = stage.stage_predict(output)
        return output


def do_test(test_config):
    classifier = Classifier(test_config)
    labels = collect_labels(test_config["labels_path"])
    (sample_image, sample_path) = load_sample_image(test_config["images_root"], test_config["image_depth"])
    image_size = (test_config["image_width"], test_config["image_height"])
    need_flatten = test_config["need_flatten"]
    need_transpose = test_config["need_transpose"]
    need_softmax = test_config["need_softmax"]
    need_bgr2rgb = test_config["need_bgr2rgb"]

    preprocessed_image = preprocess_image(
        sample_image, image_size, need_flatten, need_transpose, need_bgr2rgb
    )
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


def load_sample_image(images_root, depth):
    assert_path_exists(images_root)
    images_root_path = Path(images_root)
    assert images_root_path.is_dir(), f"Images root '{images_root}' is not directory"
    class_images_dirs = [dir for dir in images_root_path.iterdir()]

    images_paths = []
    for class_images_dir in class_images_dirs:
        for image_path in class_images_dir.iterdir():
            images_paths.append(image_path)

    sample_img_idx = int(time.time()) % len(images_paths)
    sample_image_path = images_paths[sample_img_idx]

    load_format = cv.IMREAD_COLOR
    if depth == 1:
        load_format = cv.IMREAD_GRAYSCALE

    return (cv.imread(f"{sample_image_path}", load_format), sample_image_path)


def preprocess_image(sample_image, image_size, need_flatten, need_transpose, need_bgr2rgb):
    # crop center
    # h, w = sample_image.shape[0], sample_image.shape[1]
    # x0 = (w - image_size[0]) // 2
    # y0 = (h - image_size[1]) // 2
    # sample_image = sample_image[y0 : y0 + image_size[1], x0 : x0 + image_size[0], :]

    # resize to sane dimensions
    sample_image = cv.resize(sample_image, image_size)

    if need_bgr2rgb:
        sample_image = cv.cvtColor(sample_image, cv.COLOR_BGR2RGB)

    # normalize
    sample_image = sample_image / 255.0

    # subtract mean
    # sample_image = (sample_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # transpose (if needed)
    if need_transpose:
        sample_image = np.transpose(sample_image, axes=[2, 0, 1])

    # flatten (if needed)
    if need_flatten:
        sample_image = np.array(sample_image).flatten()

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
    return labels[int(model_output)]


if __name__ == "__main__":
    do_tests(TEST_CONFIG)
