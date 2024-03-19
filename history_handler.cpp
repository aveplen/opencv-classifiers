#include "./history_handler.hpp"
#include "./model.hpp"
#include "./stacktrace.hpp"
#include "Poco/Base64Decoder.h"
#include "Poco/Dynamic/Var.h"
#include "Poco/JSON/Array.h"
#include "Poco/JSON/JSON.h"
#include "Poco/JSON/Object.h"
#include "Poco/JSON/Parser.h"
#include "history.hpp"
#include "poco/Foundation/src/zconf.h"
#include <algorithm>
#include <cassert>
#include <exception>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <valarray>
#include <vector>

namespace {

// =================== request ===================

struct HistoryRequest {
    std::optional<int> page;
    std::optional<int> page_size;
    std::optional<int> page_internal_size;
};

HistoryRequest parse_request(std::istream& req_stream) {
    Poco::JSON::Parser parser;
    auto obj = parser.parse(req_stream)
               .extract<Poco::JSON::Object::Ptr>();

    auto req = HistoryRequest{};

    if (obj->has("page")) {
        int page_value = obj->get("page");
        req.page = { page_value };
    }

    if (obj->has("page_size")) {
        int page_size_value = obj->get("page_size");
        req.page_size = { page_size_value };
    }

    if (obj->has("page_internal_size")) {
        int page_internal_size_value = obj->get("page_internal_size");
        req.page_internal_size = { page_internal_size_value };
    }

    return req;
}

// =================== response ===================

struct HistoryEntryResult {
    long int id;
    long int class_id;
    std::string class_name;
    double probability;
    long int history_entry_id;

    Poco::JSON::Object toJson() {
        Poco::JSON::Object result;
        result.set("id", this->id);
        result.set("class_id", this->class_id);
        result.set("class_name", this->class_name);
        result.set("probability", this->probability);
        result.set("history_entry_id", this->history_entry_id);
        return result;
    }
};

struct HistoryEntry {
    long int id;
    std::string model_name;
    long int duration;
    std::string time_unit;
    long int history_id;
    std::vector<HistoryEntryResult> results;

    Poco::JSON::Object toJson() {
        Poco::JSON::Object result;
        result.set("id", this->id);
        result.set("model_name", this->model_name);
        result.set("duration", this->duration);
        result.set("time_unit", this->time_unit);
        result.set("history_id", this->history_id);
        {
            Poco::JSON::Array entry_results;
            for (auto& entry_result : this->results) {
                entry_results.add(entry_result.toJson());
            }
            result.set("results", entry_results);
        }

        return result;
    }
};

struct History {
    long int id;
    std::string original_image;
    std::string cropped_image;
    std::string created_at;
    std::vector<HistoryEntry> entries;

    Poco::JSON::Object toJson() {
        Poco::JSON::Object result;
        result.set("id", this->id);
        result.set("original_image", this->original_image);
        result.set("cropped_image", this->cropped_image);
        result.set("created_at", this->created_at);
        {
            Poco::JSON::Array history_entries;
            for (auto& entry : this->entries) {
                history_entries.add(entry.toJson());
            }
            result.set("entries", history_entries);
        }

        return result;
    }
};

struct HistoryResponse {
    std::vector<History> histories;

    Poco::JSON::Object toJson() {
        Poco::JSON::Object result;

        Poco::JSON::Array histories;
        for (auto& history : this->histories) {
            histories.add(history.toJson());
        }
        result.set("histories", histories);

        return result;
    }
};

struct HistoryEntryLessThenKey {
    inline bool operator()(
    const HistoryEntry& entry1,
    const HistoryEntry& entry2
    ) {
        return entry1.id < entry2.id;
    }
};

struct HistoryEntryResultLessThenKey {
    inline bool operator()(
    const HistoryEntryResult& res1,
    const HistoryEntryResult& res2
    ) {
        return res1.id < res2.id;
    }
};

HistoryResponse build_response(
std::vector<history::History> histories,
std::optional<int> page_opt,
std::optional<int> page_size_opt,
std::optional<int> page_internal_size_opt
) {
    std::vector<History> h_dtos;

    for (history::History& history : histories) {
        std::vector<HistoryEntry> he_dtos;

        for (history::HistoryEntry& history_entry : history.m_entries) {
            std::vector<HistoryEntryResult> her_dtos;

            for (history::HistoryEntryResult& history_entry_result : history_entry.m_results) {
                her_dtos.push_back(HistoryEntryResult{
                .id = history_entry_result.m_id,
                .class_id = history_entry_result.m_class_id,
                .class_name = history_entry_result.m_class_name,
                .probability = history_entry_result.m_probability,
                .history_entry_id = history_entry_result.m_history_entry_id,
                });
            }

            std::sort(her_dtos.begin(), her_dtos.end(), HistoryEntryResultLessThenKey());
            int page_internal_size = page_internal_size_opt.has_value() ? page_internal_size_opt.value() : her_dtos.size();
            std::vector<HistoryEntryResult> her_dtos_slice(her_dtos.begin(), her_dtos.begin() + page_internal_size);

            he_dtos.push_back(HistoryEntry{
            .id = history_entry.m_id,
            .model_name = history_entry.m_model_name,
            .duration = history_entry.m_duration,
            .time_unit = history_entry.m_time_unit,
            .history_id = history_entry.m_history_id,
            .results = her_dtos,
            });
        }

        std::sort(he_dtos.begin(), he_dtos.end(), HistoryEntryLessThenKey());
        int page_size = page_size_opt.has_value() ? page_size_opt.value() : he_dtos.size();
        int page = page_opt.has_value() ? page_opt.value() : 0;
        std::vector<HistoryEntry> he_dtos_slice(he_dtos.begin() + page_size * page, he_dtos.begin() + page_size);

        h_dtos.push_back(History{
        .id = history.m_id,
        .original_image = history.m_original_image,
        .cropped_image = history.m_cropped_image,
        .created_at = history.m_created_at,
        .entries = he_dtos,
        });
    }

    return HistoryResponse{
        .histories = h_dtos,
    };
}

// =================== logic ===================

HistoryResponse get_history(
Poco::Data::Session& session,
HistoryRequest req
) {
    std::vector<history::History> histories = history::History::all(session);

    return build_response(
    histories,
    req.page,
    req.page_size,
    req.page_internal_size
    );
};

} // namespace


// =================== handler ===================

handlers::HistoryHandler::HistoryHandler(Poco::Data::Session& session, config::Config config)
: m_config(std::move(config)), m_session(session){};

void handlers::HistoryHandler::handleRequest(
Poco::Net::HTTPServerRequest& request,
Poco::Net::HTTPServerResponse& response
) {
    HistoryRequest req = parse_request(request.stream());

    HistoryResponse res = get_history(m_session, req);

    response.set("Content-Type", "application/json");
    response.setStatus(Poco::Net::HTTPServerResponse::HTTP_OK);
    auto& body = response.send();
    res.toJson().stringify(body);
    body.flush();
};
