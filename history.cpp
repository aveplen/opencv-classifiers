// todo: implement bulk-save-load

#include <unordered_map>
#include <utility>
#include <vector>

#include "./history.hpp"
#include "./stacktrace.hpp"
#include "Poco/Data/Binding.h"
#include "Poco/Data/BulkExtraction.h"
#include "Poco/Data/Data.h"
#include "Poco/Data/Range.h"
#include "Poco/Data/RecordSet.h"
#include "Poco/Data/SQLite/SQLiteException.h"
#include "Poco/Data/Session.h"
#include "Poco/Data/Statement.h"
#include "Poco/DateTime.h"

using namespace Poco::Data::Keywords;

history::HistoryEntryResult::HistoryEntryResult(
long int id,
long int class_id,
std::string class_name,
double probability,
long int history_entry_id
)
: m_id(id),
  m_class_id(class_id),
  m_class_name(std::move(class_name)),
  m_probability(probability),
  m_history_entry_id(history_entry_id){};


history::HistoryEntry::HistoryEntry(
long int id,
std::string model_name,
long int duration,
std::string time_unit,
long int history_id,
std::vector<HistoryEntryResult> results
)
: m_id(id),
  m_model_name(std::move(model_name)),
  m_duration(duration),
  m_time_unit(std::move(time_unit)),
  m_history_id(history_id),
  m_results(std::move(results)){};


history::History::History(
long int id,
std::string original_image,
std::string cropped_image,
std::string created_at,
std::vector<HistoryEntry> entries
)
: m_id(id),
  m_original_image(std::move(original_image)),
  m_cropped_image(std::move(cropped_image)),
  m_created_at(std::move(created_at)),
  m_entries(std::move(entries)){};


std::vector<history::History> history::History::all(
Poco::Data::Session& session
) {
    std::vector<data::HistoryEntryResultDB> her_dbs = data::HistoryEntryResultDB::all(session);
    std::unordered_map<long int, std::vector<data::HistoryEntryResultDB>> her_dbs_indexed;

    for (data::HistoryEntryResultDB& her_db : her_dbs) {
        if (her_dbs_indexed.find(her_db.history_entry_id) == her_dbs_indexed.end()) {
            her_dbs_indexed[her_db.history_entry_id] = {};
        }
        her_dbs_indexed[her_db.history_entry_id].push_back(her_db);
    }

    std::vector<data::HistoryEntryDB> he_dbs = data::HistoryEntryDB::all(session);
    std::unordered_map<long int, std::vector<data::HistoryEntryDB>> he_dbs_indexed;

    for (data::HistoryEntryDB& he_db : he_dbs) {
        if (he_dbs_indexed.find(he_db.history_id) == he_dbs_indexed.end()) {
            he_dbs_indexed[he_db.history_id] = {};
        }
        he_dbs_indexed[he_db.history_id].push_back(he_db);
    }

    std::vector<data::HistoryDB> history_dbs = data::HistoryDB::all(session);

    //

    std::vector<history::History> histories;

    for (data::HistoryDB& h_db : history_dbs) {
        std::vector<history::HistoryEntry> history_entries;

        for (data::HistoryEntryDB& he_db : he_dbs_indexed[h_db.id]) {
            std::vector<history::HistoryEntryResult> history_entry_results;

            for (data::HistoryEntryResultDB& her_db : her_dbs_indexed[he_db.id]) {
                history_entry_results.emplace_back(
                her_db.id,
                her_db.class_id,
                her_db.class_name,
                her_db.probability,
                her_db.history_entry_id
                );
            }

            history_entries.emplace_back(
            he_db.id,
            he_db.model_name,
            he_db.duration,
            he_db.time_unit,
            he_db.history_id,
            history_entry_results
            );
        }

        histories.emplace_back(
        h_db.id,
        h_db.original_image,
        h_db.cropped_image,
        h_db.created_at,
        history_entries
        );
    }

    return histories;
};


void history::History::save(
Poco::Data::Session& session
) {
    data::HistoryDB hdb{
        .id = m_id,
        .original_image = m_original_image,
        .cropped_image = m_cropped_image,
        .created_at = m_created_at,
    };

    hdb.create(session);
    m_id = hdb.id;

    for (history::HistoryEntry& he : m_entries) {
        data::HistoryEntryDB he_db = data::HistoryEntryDB{
            .id = he.m_id,
            .model_name = he.m_model_name,
            .duration = he.m_duration,
            .time_unit = he.m_time_unit,
            .history_id = m_id
        };

        he_db.create(session);
        he.m_id = he_db.id;

        for (history::HistoryEntryResult& her : he.m_results) {
            data::HistoryEntryResultDB her_db = data::HistoryEntryResultDB{
                .id = her.m_id,
                .class_id = her.m_class_id,
                .class_name = her.m_class_name,
                .probability = her.m_probability,
                .history_entry_id = he.m_id
            };

            her_db.create(session);
            her.m_id = her_db.id;
        }
    }
};


// ========================== HistoryEntryResult ==========================


void data::HistoryEntryResultDB::create_table(
Poco::Data::Session& session
) {
    try {
        Poco::Data::Statement create(session);
        create << data::HISTORY_ENTRY_RESULT_CREATE_TABLE_STATEMENT;
        create.execute();
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

std::vector<data::HistoryEntryResultDB> data::HistoryEntryResultDB::all(
Poco::Data::Session& session
) {
    try {
        std::vector<data::HistoryEntryResultDB> history_entry_results;

        Poco::Data::Statement select(session);
        select << data::HISTORY_ENTRY_RESULT_SELECT_ALL_STATEMENT;

        if (select.done()) {
            return history_entry_results;
        }

        select.execute();
        Poco::Data::RecordSet rs(select);
        for (std::size_t r = 0; r < rs.rowCount(); r++) {

            data::HistoryEntryResultDB history_entry_result{
                .id = rs.row(r).get(0),
                .class_id = rs.row(r).get(1),
                .class_name = rs.row(r).get(2),
                .probability = rs.row(r).get(3),
                .history_entry_id = rs.row(r).get(4),
            };

            history_entry_results.push_back(history_entry_result);
        }

        return history_entry_results;
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

void data::HistoryEntryResultDB::create(
Poco::Data::Session& session
) {
    try {
        Poco::Data::Statement insert(session);
        insert << data::HISTORY_ENTRY_RESULT_INSERT_STATEMENT,
        use(this->class_id),
        use(this->class_name),
        use(this->probability),
        use(this->history_entry_id),
        into(this->id);

        insert.execute();
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

// ========================== HistoryEntry ==========================

void data::HistoryEntryDB::create_table(
Poco::Data::Session& session
) {
    try {
        Poco::Data::Statement create(session);
        create << data::HISTORY_ENTRY_CREATE_TABLE_STATEMENT;
        create.execute();
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

std::vector<data::HistoryEntryDB> data::HistoryEntryDB::all(
Poco::Data::Session& session
) {
    try {
        std::vector<data::HistoryEntryDB> history_entries;

        Poco::Data::Statement select(session);
        select << data::HISTORY_ENTRY_SELECT_ALL_STATEMENT;

        if (select.done()) {
            return history_entries;
        }

        select.execute();
        Poco::Data::RecordSet rs(select);
        for (std::size_t r = 0; r < rs.rowCount(); r++) {

            data::HistoryEntryDB history_entry_result{
                .id = rs.row(r).get(0),
                .model_name = rs.row(r).get(1),
                .duration = rs.row(r).get(2),
                .time_unit = rs.row(r).get(3),
                .history_id = rs.row(r).get(4),
            };

            history_entries.push_back(history_entry_result);
        }

        return history_entries;
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

void data::HistoryEntryDB::create(
Poco::Data::Session& session
) {
    try {
        Poco::Data::Statement insert(session);
        insert << data::HISTORY_ENTRY_INSERT_STATEMENT,
        use(this->model_name),
        use(this->duration),
        use(this->time_unit),
        use(this->history_id),
        into(this->id);

        insert.execute();
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

// ========================== History ==========================

void data::HistoryDB::create_table(
Poco::Data::Session& session
) {
    try {
        Poco::Data::Statement create(session);
        create << data::HISTORY_CREATE_TABLE_STATEMENT;
        create.execute();
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

std::vector<data::HistoryDB> data::HistoryDB::all(
Poco::Data::Session& session
) {
    try {
        std::vector<data::HistoryDB> histories;

        Poco::Data::Statement select(session);
        select << data::HISTORY_SELECT_ALL_STATEMENT;

        if (select.done()) {
            return histories;
        }

        select.execute();
        Poco::Data::RecordSet rs(select);
        for (std::size_t r = 0; r < rs.rowCount(); r++) {

            data::HistoryDB history_entry_result{
                .id = rs.row(r).get(0),
                .original_image = rs.row(r).get(1),
                .cropped_image = rs.row(r).get(2),
                .created_at = rs.row(r).get(3),
            };

            histories.push_back(history_entry_result);
        }

        return histories;
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

void data::HistoryDB::create(Poco::Data::Session& session) {
    try {
        Poco::DateTime now;
        this->created_at = "now"; // todo

        Poco::Data::Statement insert(session);
        insert << data::HISTORY_INSERT_STATEMENT,
        use(this->original_image),
        use(this->cropped_image),
        use(this->created_at),
        into(this->id);

        insert.execute();
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};