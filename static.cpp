#include <utility>

#include "./static.hpp"
#include "Poco/Data/SQLite/Connector.h"
#include "Poco/Data/Session.h"
#include "Poco/Data/Statement.h"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPResponse.h"
#include "Poco/Net/HTTPServer.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Net/ServerSocket.h"
#include "Poco/Net/ServerSocketImpl.h"
#include "Poco/Path.h"
#include "Poco/URI.h"
#include "Poco/Util/ServerApplication.h"
#include <Poco/URI.h>
#include <fstream>
#include <sstream>


utils::StaticUtils::StaticUtils(
std::string statics_dir
)
: m_statics_dir(std::move(statics_dir)) {
    m_mime_types["html"] = "text/html";
    m_mime_types["js"] = "text/javascript";
    m_mime_types["css"] = "text/css";
};

utils::StaticUtils::~StaticUtils(){};

bool file_exists(const std::string filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

void utils::StaticUtils::send_file(
Poco::Net::HTTPServerResponse& response,
std::string path
) {
    const std::string filepath = resolve_filepath(path);
    if (!file_exists(filepath)) {
        this->send_404(response);
        return;
    }

    std::string mime_type = "text/plain";
    std::size_t dot_index = filepath.find(".");
    if (dot_index != std::string::npos) {
        auto file_type = filepath.substr(dot_index + 1);
        auto mapped_mime_type = m_mime_types.find(file_type);
        if (mapped_mime_type != m_mime_types.end()) {
            mime_type = mapped_mime_type->second;
        }
    }

    response.sendFile(filepath, mime_type);
    response.setStatus(Poco::Net::HTTPServerResponse::HTTP_OK);
};

void utils::StaticUtils::send_404(
Poco::Net::HTTPServerResponse& response
) {
    std::stringstream path_404;
    path_404 << m_statics_dir << "/404.html";
    response.sendFile(path_404.str(), "text/html");
    response.setStatus(Poco::Net::HTTPServerResponse::HTTP_NOT_FOUND);
};


std::string utils::StaticUtils::resolve_filepath(Poco::URI uri) {
    return this->resolve_filepath(uri.getPath());
};


std::string utils::StaticUtils::resolve_filepath(std::string path) {
    std::stringstream filepath_st;
    filepath_st << m_statics_dir;

    if (path == "" || path == "/") {
        filepath_st << "/index.html";
        return filepath_st.str();
    }

    filepath_st << path;
    return filepath_st.str();
};
