#ifndef STATIC_HPP
#define STATIC_HPP

#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/URI.h"
#include <string>
#include <unordered_map>

namespace utils {

class StaticUtils {
    private:
    std::string m_statics_dir;
    std::unordered_map<std::string, std::string> m_mime_types;

    public:
    StaticUtils(std::string statics_dir);
    ~StaticUtils();

    void send_file(Poco::Net::HTTPServerResponse& response, std::string filename);
    void send_404(Poco::Net::HTTPServerResponse& response);
    std::string resolve_filepath(Poco::URI uri);
    std::string resolve_filepath(std::string path);
};

}; // namespace utils

#endif // STATIC_HPP