#ifndef UTILS_HPP
#define UTILS_HPP

#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <string>

namespace mat_utils {

class DumpMatToFileHook {
    private:
    std::string m_filename;

    public:
    DumpMatToFileHook(std::string filename)
    : m_filename(filename){};

    void dump_to_file(cv::Mat mat) {
        std::ofstream file(m_filename);
        file << "Rows: " << mat.rows << std::endl;
        file << "Cols: " << mat.cols << std::endl;
        file << "Channels: " << mat.channels() << std::endl;
        file << "Type: " << mat.type() << std::endl;
        file << std::endl;

        for (int row = 0; row < mat.rows; row++) {
            for (int col = 0; col < mat.cols; col++) {
                file << mat.col(col).row(row) << " ";
            }
            file << std::endl;
        }
        file << std::endl;


        for (int row = 0; row < mat.rows; row++) {
            for (int col = 0; col < mat.cols; col++) {
                int t = mat.type();

                if (t == CV_8UC1) {
                    file << (int)mat.at<uchar>(row, col) << " ";
                    continue;
                }

                if (t == CV_8UC3) {
                    cv::Mat planes[3];
                    cv::split(mat, planes);
                    file << (int)planes[0].at<uchar>(row, col) << " "
                         << (int)planes[1].at<uchar>(row, col) << " "
                         << (int)planes[2].at<uchar>(row, col) << " ";
                    continue;
                }

                if (t == CV_32FC1) {
                    file << mat.at<float>(row, col) << " ";
                    continue;
                }

                if (t == CV_32FC3) {
                    cv::Mat planes[3];
                    cv::split(mat, planes);
                    file << planes[0].at<float>(row, col) << " "
                         << planes[1].at<float>(row, col) << " "
                         << planes[2].at<float>(row, col) << " ";
                    continue;
                }

                if (t == CV_64FC1) {
                    file << mat.at<double>(row, col) << " ";
                    continue;
                }

                if (t == CV_64FC3) {
                    cv::Mat planes[3];
                    cv::split(mat, planes);
                    file << planes[0].at<double>(row, col) << " "
                         << planes[1].at<double>(row, col) << " "
                         << planes[2].at<double>(row, col) << " ";
                    continue;
                }
            }

            file << std::endl;
        }
    }
};

} // namespace mat_utils

#endif // UTILS_HPP