#include "./statements_handler.hpp"
#include "./data.hpp"

#include "Poco/JSON/Array.h"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPServer.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Net/ServerSocket.h"
#include "Poco/Net/ServerSocketImpl.h"
#include "Poco/Path.h"
#include "Poco/Util/ServerApplication.h"
#include <Poco/URI.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::optional<std::string> find_query(
const std::vector<std::pair<std::string, std::string>> query_params,
const std::string name
) {
    std::optional<std::string> query_value = std::nullopt;
    for (const auto& i : query_params) {
        if (i.first == name) {
            query_value = { i.second };
            break;
        }
    }
    return query_value;
}

std::optional<int> parse_int(std::optional<std::string> arg) {
    if (!arg.has_value()) {
        return std::nullopt;
    }

    try {
        return { std::stoi(arg.value()) };
    } catch (std::exception& ex) {
        return std::nullopt;
    }
}

template <typename T>
bool validate_supplied(
Poco::Net::HTTPServerResponse& response,
std::optional<T> opt,
std::string name
) {
    if (opt.has_value()) {
        return true;
    }

    response.setStatus(Poco::Net::HTTPServerResponse::HTTP_BAD_REQUEST);
    auto& body = response.send();
    body << "'" << name << "' query param is required";
    body.flush();
    return false;
}

void handlers::StatementsHandler::handleRequest(
Poco::Net::HTTPServerRequest& request,
Poco::Net::HTTPServerResponse& response
) {
    const Poco::URI uri(request.getURI());
    const Poco::URI::QueryParameters query = uri.getQueryParameters();

    const std::optional<std::string> entity = find_query(query, "entity");
    const std::optional<std::string> name = find_query(query, "name");
    const std::optional<int> id = parse_int(find_query(query, "id"));
    const std::optional<int> width = parse_int(find_query(query, "width"));
    const std::optional<int> height = parse_int(find_query(query, "height"));
    const std::optional<std::string> data = find_query(query, "data");

    if (!validate_supplied(response, entity, "entity"))
        return;
    if (!validate_supplied(response, name, "name"))
        return;

    if (entity.value() == "images" && name.value() == "update") {
        std::optional<Data::Image> image_opt = Data::Image::find(this->session, id.value());
        if (!image_opt.has_value()) {
            response.setStatus(Poco::Net::HTTPServerResponse::HTTP_NOT_FOUND);
            return;
        }

        Data::Image image = image_opt.value();
        if (width.has_value())
            image.width = width.value();
        if (height.has_value())
            image.height = height.value();
        if (data.has_value())
            image.data = data.value();

        image.update(this->session);
    }

    if (entity.value() == "images" && name.value() == "insert") {
        Data::Image image{};

        if (width.has_value())
            image.width = width.value();
        if (height.has_value())
            image.height = height.value();
        if (data.has_value())
            image.data = data.value();

        image.create(this->session);
    }

    if (entity.value() == "images" && name.value() == "select") {
        std::optional<Data::Image> image = Data::Image::find(this->session, id.value());
        if (!image.has_value()) {
            response.setStatus(Poco::Net::HTTPServerResponse::HTTP_NOT_FOUND);
            return;
        }

        response.setStatus(Poco::Net::HTTPServerResponse::HTTP_OK);
        response.set("Content-Type", "application/json");
        auto& body = response.send();
        image->toJson().stringify(body);
        body.flush();
        return;
    }

    if (entity.value() == "images" && name.value() == "all") {
        std::vector<Data::Image> images = Data::Image::all(this->session);
        Poco::JSON::Array results;
        for (Data::Image& image : images) {
            results.add(image.toJson());
        }

        response.setStatus(Poco::Net::HTTPServerResponse::HTTP_OK);
        response.set("Content-Type", "application/json");
        auto& body = response.send();
        results.stringify(body);
        body.flush();
        return;
    }


    response.setStatus(Poco::Net::HTTPServerResponse::HTTP_NOT_FOUND);
};