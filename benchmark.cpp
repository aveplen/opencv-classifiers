#include "./cnn_svm.hpp"
#include "./hog_svm.hpp"
#include "./resnet50.hpp"
#include "./svm.hpp"
#include "./utils.hpp"

#include <cctype>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct BenchmarkResult {
    double svm_precision;
    double hog_svm_precision;
    double cnn_svm_precision;
    double resnet50_precision;
};

std::string lowercase(std::string orig) {
    std::stringstream result;
    for (auto& c : orig) {
        result << (char)tolower(c);
    }
    return result.str();
}

BenchmarkResult run_224x224_benchmark(std::string path) {
    svm::SVMClassifier svm(
    "learning/svm_groceries_37632_f32.dat",
    "learning/groceries_labels.txt",
    112, 112
    );

    hog_svm::HOGSVMClassifier hog_svm(
    "learning/hog_svm_groceries_f32.dat",
    "learning/groceries_labels.txt",
    224, 224
    );

    cnn_svm::CNNSVMClassifier cnn_svm(
    "learning/resnet50_feature_extractor_groceries_224_224_3.onnx",
    "learning/cnn_svm_groceries_37632_f32.dat",
    "learning/groceries_labels.txt",
    224, 224
    );

    resnet50::Resnet50Classifier resnet50(
    "learning/finetuned_resnet50_groceries_224_224_3.onnx",
    "learning/groceries_labels.txt",
    224, 224
    );

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;

    std::vector<bool> svm_results;
    std::vector<double> svm_timings;

    std::vector<bool> hog_svm_results;
    std::vector<double> hog_svm_timings;

    std::vector<bool> cnn_svm_results;
    std::vector<double> cnn_svm_timings;

    std::vector<bool> resnet50_results;
    std::vector<double> resnet50_timings;

    for (const auto& dir_entry : recursive_directory_iterator(path)) {
        std::string expected_label = lowercase(dir_entry.path().filename());

        if (!dir_entry.is_directory()) {
            continue;
        }

        for (const auto& pic_entry : recursive_directory_iterator(dir_entry.path())) {
            cv::Mat image = cv::imread(pic_entry.path(), cv::IMREAD_COLOR);

            svm::Result svm_result = svm.classify(image, "bgr");
            svm_results.push_back(svm_result.label == expected_label);

            hog_svm::Result hog_svm_result = hog_svm.classify(image, "bgr");
            hog_svm_results.push_back(hog_svm_result.label == expected_label);

            cnn_svm::Result cnn_svm_result = cnn_svm.classify(image, "bgr");
            cnn_svm_results.push_back(cnn_svm_result.label == expected_label);

            std::vector<resnet50::Result> resnet50_result = resnet50.classify(image, "bgr");
            resnet50::Result argmax = { 0 };
            for (auto& i : resnet50_result) {
                if (i.probability > argmax.probability) {
                    argmax = i;
                }
            }
            resnet50_results.push_back(argmax.label == expected_label);
        }
    }

    int svm_good_guesses = 0;
    int hog_svm_good_guesses = 0;
    int cnn_svm_good_guesses = 0;
    int resnet50_good_guesses = 0;

    for (std::size_t i = 0; i < svm_results.size(); i++) {
        if (svm_results[i])
            svm_good_guesses++;

        if (hog_svm_results[i])
            hog_svm_good_guesses++;

        if (cnn_svm_results[i])
            cnn_svm_good_guesses++;

        if (resnet50_results[i])
            resnet50_good_guesses++;
    }

    BenchmarkResult benchmark_result = { 0 };
    benchmark_result.svm_precision = (double)svm_good_guesses / (double)svm_results.size();
    benchmark_result.hog_svm_precision = (double)hog_svm_good_guesses / (double)hog_svm_results.size();
    benchmark_result.cnn_svm_precision = (double)cnn_svm_good_guesses / (double)cnn_svm_results.size();
    benchmark_result.resnet50_precision = (double)resnet50_good_guesses / (double)resnet50_results.size();
    return benchmark_result;
}

BenchmarkResult run_32x32_benchmark(std::string path) {
    svm::SVMClassifier svm(
    "learning/svm_cifar10_12288_f32.dat",
    "learning/cifar10_labels.txt",
    64, 64
    );

    hog_svm::HOGSVMClassifier hog_svm(
    "learning/hog_svm_cifar10_f32.dat",
    "learning/cifar10_labels.txt",
    64, 64
    );

    cnn_svm::CNNSVMClassifier cnn_svm(
    "learning/resnet50_feature_extractor_cifar10_64_64_3.onnx",
    "learning/cnn_svm_cifar10_12288_f32.dat",
    "learning/cifar10_labels.txt",
    64, 64
    );

    resnet50::Resnet50Classifier resnet50(
    "learning/finetuned_resnet50_cifar10_64_64_3.onnx",
    "learning/cifar10_labels.txt",
    64, 64
    );

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;

    std::vector<bool> svm_results;
    std::vector<double> svm_timings;

    std::vector<bool> hog_svm_results;
    std::vector<double> hog_svm_timings;

    std::vector<bool> cnn_svm_results;
    std::vector<double> cnn_svm_timings;

    std::vector<bool> resnet50_results;
    std::vector<double> resnet50_timings;

    for (const auto& dir_entry : recursive_directory_iterator(path)) {
        std::string expected_label = lowercase(dir_entry.path().filename());

        if (!dir_entry.is_directory()) {
            continue;
        }

        for (const auto& pic_entry : recursive_directory_iterator(dir_entry.path())) {
            cv::Mat image = cv::imread(pic_entry.path(), cv::IMREAD_COLOR);

            svm::Result svm_result = svm.classify(image, "bgr");
            svm_results.push_back(svm_result.label == expected_label);

            hog_svm::Result hog_svm_result = hog_svm.classify(image, "bgr");
            hog_svm_results.push_back(hog_svm_result.label == expected_label);

            cnn_svm::Result cnn_svm_result = cnn_svm.classify(image, "bgr");
            cnn_svm_results.push_back(cnn_svm_result.label == expected_label);

            std::vector<resnet50::Result> resnet50_result = resnet50.classify(image, "bgr");
            resnet50::Result argmax = { 0 };
            for (auto& i : resnet50_result) {
                if (i.probability > argmax.probability) {
                    argmax = i;
                }
            }
            resnet50_results.push_back(argmax.label == expected_label);
        }
    }

    int svm_good_guesses = 0;
    int hog_svm_good_guesses = 0;
    int cnn_svm_good_guesses = 0;
    int resnet50_good_guesses = 0;

    for (std::size_t i = 0; i < svm_results.size(); i++) {
        if (svm_results[i])
            svm_good_guesses++;

        if (hog_svm_results[i])
            hog_svm_good_guesses++;

        if (cnn_svm_results[i])
            cnn_svm_good_guesses++;

        if (resnet50_results[i])
            resnet50_good_guesses++;
    }

    BenchmarkResult benchmark_result = { 0 };
    benchmark_result.svm_precision = (double)svm_good_guesses / (double)svm_results.size();
    benchmark_result.hog_svm_precision = (double)hog_svm_good_guesses / (double)hog_svm_results.size();
    benchmark_result.cnn_svm_precision = (double)cnn_svm_good_guesses / (double)cnn_svm_results.size();
    benchmark_result.resnet50_precision = (double)resnet50_good_guesses / (double)resnet50_results.size();
    return benchmark_result;
}

// int main() {
//     {
//         svm::SVMClassifier svm(
//         "learning/svm_groceries_37632_f32.dat",
//         "learning/groceries_labels.txt",
//         112, 112
//         );

//         svm::Result svm_result = svm.classify(groceries_sample, "bgr");
//         std::cout << svm_result.clazz << " " << svm_result.label << std::endl;

//         hog_svm::HOGSVMClassifier hog_svm(
//         "learning/hog_svm_groceries_f32.dat",
//         "learning/groceries_labels.txt",
//         224, 224
//         );

//         hog_svm::Result hog_svm_result = hog_svm.classify(groceries_sample, "bgr");
//         std::cout << hog_svm_result.clazz << " " << hog_svm_result.label << std::endl;

//         cnn_svm::CNNSVMClassifier cnn_svm(
//         "learning/resnet50_feature_extractor_groceries_224_224_3.onnx",
//         "learning/cnn_svm_groceries_37632_f32.dat",
//         "learning/groceries_labels.txt",
//         224, 224
//         );

//         cnn_svm::Result cnn_svm_result = cnn_svm.classify(groceries_sample, "bgr");
//         std::cout << cnn_svm_result.clazz << " " << cnn_svm_result.label << std::endl;

//         resnet50::Resnet50Classifier resnet50(
//         "learning/finetuned_resnet50_groceries_224_224_3.onnx",
//         "learning/groceries_labels.txt",
//         224, 224
//         );

//         std::vector<resnet50::Result> resnet50_result = resnet50.classify(groceries_sample, "bgr");
//         resnet50::Result argmax = { 0 };
//         for (auto& i : resnet50_result) {
//             if (i.probability > argmax.probability) {
//                 argmax = i;
//             }
//         }
//     }

//     {
//         svm::SVMClassifier svm(
//         "learning/svm_cifar10_12288_f32.dat",
//         "learning/cifar10_labels.txt",
//         64, 64
//         );

//         svm::Result svm_result = svm.classify(cifar10_sample, "bgr");
//         std::cout << svm_result.clazz << " " << svm_result.label << std::endl;

//         hog_svm::HOGSVMClassifier hog_svm(
//         "learning/hog_svm_cifar10_f32.dat",
//         "learning/cifar10_labels.txt",
//         64, 64
//         );

//         hog_svm::Result hog_svm_result = hog_svm.classify(cifar10_sample, "bgr");
//         std::cout << hog_svm_result.clazz << " " << hog_svm_result.label << std::endl;

//         cnn_svm::CNNSVMClassifier cnn_svm(
//         "learning/resnet50_feature_extractor_cifar10_64_64_3.onnx",
//         "learning/cnn_svm_cifar10_12288_f32.dat",
//         "learning/cifar10_labels.txt",
//         64, 64
//         );

//         cnn_svm::Result cnn_svm_result = cnn_svm.classify(cifar10_sample, "bgr");
//         std::cout << cnn_svm_result.clazz << " " << cnn_svm_result.label << std::endl;

//         resnet50::Resnet50Classifier resnet50(
//         "learning/finetuned_resnet50_cifar10_64_64_3.onnx",
//         "learning/cifar10_labels.txt",
//         64, 64
//         );

//         std::vector<resnet50::Result> resnet50_result = resnet50.classify(cifar10_sample, "bgr");
//         resnet50::Result argmax = { 0 };
//         for (auto& i : resnet50_result) {
//             if (i.probability > argmax.probability) {
//                 argmax = i;
//             }
//         }
//     }
// }

int main() {
    BenchmarkResult result_224x224 = run_224x224_benchmark("./benchmark_pics/224x224");
    BenchmarkResult result_32x32 = run_32x32_benchmark("./benchmark_pics/32x32");
    return 0;
}