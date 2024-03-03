#include "./data.hpp"
#include "./stacktrace.hpp"
#include "Poco/Data/Binding.h"
#include "Poco/Data/BulkExtraction.h"
#include "Poco/Data/Data.h"
#include "Poco/Data/Range.h"
#include "Poco/Data/RecordSet.h"
#include "Poco/Data/SQLite/SQLiteException.h"
#include "Poco/Data/Statement.h"
#include "Poco/DateTime.h"
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

using namespace Poco::Data::Keywords;

void Data::Image::create_table(
Poco::Data::Session& session
) {
    try {
        Poco::Data::Statement create(session);
        create << Data::IMAGE_CREATE_TABLE_STATEMENT;
        create.execute();
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

std::optional<Data::Image> Data::Image::find(
Poco::Data::Session& session,
unsigned long int id
) {
    try {
        Data::Image image{};

        Poco::Data::Statement select(session);
        select << Data::IMAGE_SELECT_BY_ID_STATEMENT,
        use(id),
        into(image.id),
        into(image.height),
        into(image.width),
        into(image.data),
        range(0, 1);

        if (select.done()) {
            return std::nullopt;
        }

        select.execute();
        return { image };
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

std::vector<Data::Image> Data::Image::all(
Poco::Data::Session& session,
unsigned int limit
) {
    try {
        std::vector<Data::Image> images(limit);

        Poco::Data::Statement select(session);
        select << Data::IMAGE_SELECT_ALL_STATEMENT;

        if (select.done()) {
            return images;
        }

        select.execute();
        Poco::Data::RecordSet rs(select);
        for (std::size_t r = 0; r < rs.rowCount(); r++) {

            Data::Image image{
                .id = rs.row(r).get(0),
                .height = rs.row(r).get(1),
                .width = rs.row(r).get(2),
                .data = rs.row(r).get(3),
            };

            images.push_back(image);
        }

        return images;
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

void Data::Image::update(Poco::Data::Session& session) {
    try {
        Poco::Data::Statement update(session);
        update << Data::IMAGE_UPDATE_STATEMENT,
        use(this->height),
        use(this->width),
        use(this->data),
        use(this->id);

        update.execute();
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

void Data::Image::create(Poco::Data::Session& session) {
    try {
        Poco::DateTime now;
        this->created_at = now;

        Poco::Data::Statement insert(session);
        insert << Data::IMAGE_INSERT_STATEMENT,
        use(this->height),
        use(this->width),
        use(this->data),
        use(this->created_at),
        into(this->id);

        insert.execute();
    } catch (Poco::Data::SQLite::SQLiteException& ex) {
        throw ExceptionWithTrace(ex, ex.displayText());
    } catch (std::exception& ex) {
        throw ExceptionWithTrace(ex);
    }
};

Poco::JSON::Object Data::Image::toJson() {
    Poco::JSON::Object result;
    result.set("id", this->id);
    result.set("width", this->width);
    result.set("height", this->height);
    result.set("data", this->data);
    result.set("created_at", this->created_at);
    return result;
};