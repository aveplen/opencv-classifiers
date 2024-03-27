#ifndef HISTORY_MODEL_HPP
#define HISTORY_MODEL_HPP

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <vector>

#include "Poco/Data/Session.h"
#include "Poco/DateTime.h"
#include "Poco/JSON/Object.h"
#include <cstddef>
#include <string>
#include <vector>

namespace history {

class HistoryEntryResult {
    public:
    long int m_id;
    long int m_class_id;
    std::string m_class_name;
    double m_probability;
    long int m_history_entry_id;

    HistoryEntryResult(
    long int id,
    long int class_id,
    std::string class_name,
    double probability,
    long int history_entry_id
    );
};

class HistoryEntry {
    public:
    long int m_id;
    std::string m_model_name;
    long int m_duration;
    std::string m_time_unit;
    long int m_history_id;
    std::vector<HistoryEntryResult> m_results;

    HistoryEntry(
    long int id,
    std::string model_name,
    long int duration,
    std::string time_unit,
    long int history_id,
    std::vector<HistoryEntryResult> results
    );
};

class History {
    public:
    long int m_id;
    std::string m_original_image;
    std::string m_cropped_image;
    std::string m_preset_name;
    std::string m_created_at;
    std::vector<HistoryEntry> m_entries;

    History(
    long int id,
    std::string original_image,
    std::string cropped_image,
    std::string preset_name,
    std::string created_at,
    std::vector<HistoryEntry> entries
    );

    static std::vector<History> all(Poco::Data::Session& session);
    static void save_all(Poco::Data::Session& session, std::vector<History*> histories);
    void save(Poco::Data::Session& session);
};

} // namespace history

namespace data {

const std::string HISTORY_ENTRY_RESULT_CREATE_TABLE_STATEMENT = "CREATE TABLE IF NOT EXISTS history_entry_result (id INTEGER PRIMARY KEY, class_id INTEGER NOT NULL, class_name TEXT NOT NULL, probability FLOAT NOT NULL, history_entry_id INTEGER NOT NULL);";
// const std::string HISTORY_ENTRY_RESULT_SELECT_BY_ID_STATEMENT = "SELECT id, height, width, data, created_at FROM history_entry_result WHERE id = ?";
const std::string HISTORY_ENTRY_RESULT_SELECT_ALL_STATEMENT = "SELECT id, class_id, class_name, probability, history_entry_id FROM history_entry_result;";
const std::string HISTORY_ENTRY_RESULT_INSERT_STATEMENT = "INSERT INTO history_entry_result (class_id, class_name, probability, history_entry_id) VALUES(?, ?, ?, ?) RETURNING id";
// const std::string HISTORY_ENTRY_RESULT_UPDATE_STATEMENT = "UPDATE history_entry_result SET height = ?, width = ?, data = ? WHERE id = ?;";

struct HistoryEntryResultDB {
    long int id = 0;
    long int class_id;
    std::string class_name;
    double probability;
    long int history_entry_id;

    static void create_table(
    Poco::Data::Session& session
    );

    static std::vector<HistoryEntryResultDB> all(
    Poco::Data::Session& session
    );

    void create(Poco::Data::Session& session);
};

const std::string HISTORY_ENTRY_CREATE_TABLE_STATEMENT = "CREATE TABLE IF NOT EXISTS history_entry (id INTEGER PRIMARY KEY, model_name TEXT NOT NULL, duration INTEGER NOT NULL, time_unit TEXT NOT NULL, history_id INTEGER NOT NULL)";
// const std::string HISTORY_ENTRY_SELECT_BY_ID_STATEMENT = "SELECT id, model_name, duration, time_unit, history_id FROM history_entry WHERE id = ?";
const std::string HISTORY_ENTRY_SELECT_ALL_STATEMENT = "SELECT id, model_name, duration, time_unit, history_id FROM history_entry";
const std::string HISTORY_ENTRY_INSERT_STATEMENT = "INSERT INTO history_entry (model_name, duration, time_unit, history_id) VALUES(?, ?, ?, ?) RETURNING id";
// const std::string HISTORY_ENTRY_UPDATE_STATEMENT = "UPDATE history_entry SET model_name = ?, duration = ?, time_unit = ?, history_id = ? WHERE id = ?";

struct HistoryEntryDB {
    long int id = 0;
    std::string model_name;
    long int duration;
    std::string time_unit;
    long int history_id;

    static void create_table(
    Poco::Data::Session& session
    );

    // static std::optional<HistoryEntry> find(
    // Poco::Data::Session& session,
    // unsigned long int id
    // );

    static std::vector<HistoryEntryDB> all(
    Poco::Data::Session& session
    );

    // void update(Poco::Data::Session& session);

    void create(Poco::Data::Session& session);

    // Poco::JSON::Object toJson();
};


const std::string HISTORY_CREATE_TABLE_STATEMENT = "CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, original_image TEXT NOT NULL, cropped_image TEXT NOT NULL, preset_name TEXT NOT NULL, created_at TEXT NOT NULL)";
// const std::string HISTORY_SELECT_BY_ID_STATEMENT = "SELECT id, original_image, cropped_image, created_at FROM history WHERE id = ?";
const std::string HISTORY_SELECT_ALL_STATEMENT = "SELECT id, original_image, cropped_image, preset_name, created_at FROM history";
const std::string HISTORY_INSERT_STATEMENT = "INSERT INTO history (original_image, cropped_image, preset_name, created_at) VALUES(?, ?, ?) RETURNING id";
// const std::string HISTORY_UPDATE_STATEMENT = "UPDATE history SET original_image = ?, cropped_image = ? WHERE id = ?";

struct HistoryDB {
    long int id = 0;
    std::string original_image;
    std::string cropped_image;
    std::string preset_name;
    std::string created_at;

    static void create_table(
    Poco::Data::Session& session
    );

    // static std::optional<History> find(
    // Poco::Data::Session& session,
    // unsigned long int id
    // );

    static std::vector<HistoryDB> all(
    Poco::Data::Session& session
    );

    // void update(Poco::Data::Session& session);

    void create(Poco::Data::Session& session);

    // Poco::JSON::Object toJson();
};

} // namespace data

#endif // HISTORY_MODEL_HPP