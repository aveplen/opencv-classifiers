#ifndef BASE64_UTILS_HPP
#define BASE64_UTILS_HPP

#include <sstream>

#include "Poco/Base64Decoder.h"
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>


namespace base64 {

using namespace std;
using namespace cv;

Mat decode_image(string encoded_image, int cv_mode = IMREAD_COLOR) {
    istringstream b64iss(encoded_image);
    ostringstream oss;
    Poco::Base64Decoder decoder(b64iss);

    copy(
    istreambuf_iterator<char>(decoder),
    istreambuf_iterator<char>(),
    ostreambuf_iterator<char>(oss)
    );

    string decoded = oss.str();
    vector<char> bytes(decoded.begin(), decoded.end());
    return imdecode(bytes, cv_mode);
}

}; // namespace base64

#endif // BASE64_UTILS_HPP