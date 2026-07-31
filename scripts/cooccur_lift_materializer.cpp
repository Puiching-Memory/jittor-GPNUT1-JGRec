#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

template <typename T>
std::vector<T> read_binary(const std::string& path, std::size_t count) {
    std::vector<T> values(count);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open input: " + path);
    }
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("unexpected input size: " + path);
    }
    return values;
}

template <typename T>
void write_binary(const std::string& path, const std::vector<T>& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open output: " + path);
    }
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!output) {
        throw std::runtime_error("failed to write output: " + path);
    }
}

std::uint64_t directed_key(std::int32_t left, std::int32_t right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32U)
        | static_cast<std::uint32_t>(right);
}

std::uint64_t triangular_cell(std::int32_t first, std::int32_t second) {
    if (first == second) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    const auto low = static_cast<std::uint64_t>(std::min(first, second));
    const auto high = static_cast<std::uint64_t>(std::max(first, second));
    return high * (high - 1U) / 2U + low;
}

std::uint32_t count_at(
    const std::vector<std::uint8_t>& base,
    const std::unordered_map<std::uint64_t, std::uint32_t>& overflow,
    std::uint64_t cell) {
    if (cell == std::numeric_limits<std::uint64_t>::max()) {
        return 0;
    }
    const auto value = base[cell];
    if (value < std::numeric_limits<std::uint8_t>::max()) {
        return value;
    }
    const auto found = overflow.find(cell);
    return static_cast<std::uint32_t>(value)
        + (found == overflow.end() ? 0U : found->second);
}

void increment_count(
    std::vector<std::uint8_t>& base,
    std::unordered_map<std::uint64_t, std::uint32_t>& overflow,
    std::uint64_t cell) {
    auto& value = base[cell];
    if (value < std::numeric_limits<std::uint8_t>::max()) {
        ++value;
        return;
    }
    ++overflow[cell];
}

void decrement_count(
    std::vector<std::uint8_t>& base,
    std::unordered_map<std::uint64_t, std::uint32_t>& overflow,
    std::uint64_t cell) {
    auto& value = base[cell];
    if (value < std::numeric_limits<std::uint8_t>::max()) {
        if (value == 0) {
            throw std::runtime_error("short cooccurrence count underflow");
        }
        --value;
        return;
    }
    const auto found = overflow.find(cell);
    if (found == overflow.end()) {
        --value;
        return;
    }
    if (--found->second == 0) {
        overflow.erase(found);
    }
}

struct SourceState {
    std::vector<std::int32_t> cooccur_recent;
    std::vector<std::int32_t> latest_unique;
};

struct CooccurEvent {
    std::uint64_t cell;
    std::int32_t time;
};

struct PopularityEvent {
    std::int32_t destination;
    std::int32_t time;
};

class Materializer {
public:
    Materializer(
        std::vector<std::int32_t> train_src,
        std::vector<std::int32_t> train_dst,
        std::vector<std::int32_t> train_time,
        std::vector<std::int32_t> query_src,
        std::vector<std::int32_t> query_candidates,
        std::vector<std::int32_t> query_dst,
        std::vector<std::int32_t> query_time,
        std::vector<std::int32_t> query_availability_time,
        std::size_t candidate_count,
        double short_window,
        std::int32_t maximum_source,
        std::int32_t maximum_destination,
        std::string progress_path)
        : train_src_(std::move(train_src)),
          train_dst_(std::move(train_dst)),
          train_time_(std::move(train_time)),
          query_src_(std::move(query_src)),
          query_candidates_(std::move(query_candidates)),
          query_dst_(std::move(query_dst)),
          query_time_(std::move(query_time)),
          query_availability_time_(std::move(query_availability_time)),
          candidate_count_(candidate_count),
          short_window_(short_window),
          states_(static_cast<std::size_t>(maximum_source) + 1U),
          full_popularity_(static_cast<std::size_t>(maximum_destination) + 1U),
          short_popularity_(static_cast<std::size_t>(maximum_destination) + 1U),
          progress_path_(std::move(progress_path)) {
        const auto destination_count =
            static_cast<std::uint64_t>(maximum_destination) + 1U;
        const auto cells = destination_count * (destination_count - 1U) / 2U;
        full_counts_.resize(cells);
        short_counts_.resize(cells);
        lift_.resize(query_src_.size() * candidate_count_ * 2U);
        positive_popularity_.resize(query_src_.size());
        active_seen_.reserve(states_.size() * 32U);
    }

    void run() {
        if (query_time_.empty()) {
            return;
        }
        const auto started = std::chrono::steady_clock::now();
        std::size_t train_position = 0;
        std::size_t query_position = 0;
        while (query_position < query_time_.size()) {
            const auto availability_time =
                query_availability_time_[query_position];
            while (train_position < train_time_.size()
                   && train_time_[train_position] < availability_time) {
                process_train_event(train_position);
                ++train_position;
            }
            const auto availability_group_start = query_position;
            while (
                query_position < query_time_.size()
                && query_availability_time_[query_position]
                    == availability_time) {
                const auto query_time = query_time_[query_position];
                expire(query_time);
                const auto query_group_start = query_position;
                while (
                    query_position < query_time_.size()
                    && query_availability_time_[query_position]
                        == availability_time
                    && query_time_[query_position] == query_time) {
                    ++query_position;
                }
                process_query_group(query_group_start, query_position);
            }
            while (train_position < train_time_.size()
                   && train_time_[train_position] == availability_time) {
                process_train_event(train_position);
                ++train_position;
            }
            if (query_position <= availability_group_start) {
                throw std::runtime_error("query availability group did not advance");
            }
            write_progress(query_position, started);
        }
    }

    const std::vector<float>& lift() const { return lift_; }
    const std::vector<std::int32_t>& positive_popularity() const {
        return positive_popularity_;
    }

private:
    void expire(std::int32_t current_time) {
        const auto lower_time =
            static_cast<double>(current_time) - short_window_;
        while (!short_cooccurrences_.empty()
               && static_cast<double>(short_cooccurrences_.front().time)
                   <= lower_time) {
            const auto event = short_cooccurrences_.front();
            short_cooccurrences_.pop_front();
            decrement_count(short_counts_, short_overflow_, event.cell);
        }
        while (!short_popularity_events_.empty()
               && static_cast<double>(short_popularity_events_.front().time)
                   <= lower_time) {
            const auto event = short_popularity_events_.front();
            short_popularity_events_.pop_front();
            auto& count = short_popularity_[event.destination];
            if (count == 0) {
                throw std::runtime_error("short popularity count underflow");
            }
            --count;
        }
    }

    void process_train_event(std::size_t position) {
        const auto src = train_src_[position];
        const auto dst = train_dst_[position];
        const auto time = train_time_[position];
        auto& state = states_[src];
        const auto active_key = directed_key(src, dst);
        if (active_seen_.find(active_key) == active_seen_.end()) {
            for (const auto other : state.cooccur_recent) {
                const auto cell = triangular_cell(other, dst);
                increment_count(full_counts_, full_overflow_, cell);
                increment_count(short_counts_, short_overflow_, cell);
                short_cooccurrences_.push_back({cell, time});
            }
            active_seen_.insert(active_key);
            state.cooccur_recent.push_back(dst);
            if (state.cooccur_recent.size() > 256U) {
                const auto expired = state.cooccur_recent.front();
                state.cooccur_recent.erase(state.cooccur_recent.begin());
                active_seen_.erase(directed_key(src, expired));
            }
        }

        auto& latest = state.latest_unique;
        const auto found = std::find(latest.begin(), latest.end(), dst);
        if (found != latest.end()) {
            latest.erase(found);
        }
        latest.push_back(dst);
        if (latest.size() > 64U) {
            latest.erase(latest.begin());
        }

        ++full_popularity_[dst];
        ++short_popularity_[dst];
        short_popularity_events_.push_back({dst, time});
    }

    void process_query_group(std::size_t start, std::size_t stop) {
#pragma omp parallel for schedule(static)
        for (std::int64_t signed_row = static_cast<std::int64_t>(start);
             signed_row < static_cast<std::int64_t>(stop);
             ++signed_row) {
            const auto row = static_cast<std::size_t>(signed_row);
            const auto& history = states_[query_src_[row]].latest_unique;
            const auto candidate_offset = row * candidate_count_;
            const auto output_offset = candidate_offset * 2U;
            for (std::size_t column = 0; column < candidate_count_; ++column) {
                const auto candidate =
                    query_candidates_[candidate_offset + column];
                std::uint64_t full_cooccurrences = 0;
                std::uint64_t short_cooccurrences = 0;
                for (const auto seen : history) {
                    const auto cell = triangular_cell(seen, candidate);
                    full_cooccurrences +=
                        count_at(full_counts_, full_overflow_, cell);
                    short_cooccurrences +=
                        count_at(short_counts_, short_overflow_, cell);
                }
                lift_[output_offset + column * 2U] =
                    static_cast<float>(
                        std::log1p(static_cast<double>(full_cooccurrences))
                        - std::log1p(
                            static_cast<double>(full_popularity_[candidate])));
                lift_[output_offset + column * 2U + 1U] =
                    static_cast<float>(
                        std::log1p(static_cast<double>(short_cooccurrences))
                        - std::log1p(
                            static_cast<double>(short_popularity_[candidate])));
            }
            positive_popularity_[row] =
                static_cast<std::int32_t>(
                    full_popularity_[query_dst_[row]]);
        }
    }

    void write_progress(
        std::size_t completed,
        std::chrono::steady_clock::time_point started) const {
        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::ofstream output(progress_path_, std::ios::trunc);
        output << "{\"candidate_metrics_read\":false,"
               << "\"completed_rows\":" << completed << ','
               << "\"elapsed_seconds\":" << elapsed << ','
               << "\"external_scores_read\":false,"
               << "\"rows_per_second\":"
               << static_cast<double>(completed) / std::max(elapsed, 1e-9)
               << ",\"status\":\"materializing\","
               << "\"total_rows\":" << query_src_.size() << "}\n";
        std::cout << "materialized_rows=" << completed
                  << " elapsed_seconds=" << elapsed << std::endl;
    }

    std::vector<std::int32_t> train_src_;
    std::vector<std::int32_t> train_dst_;
    std::vector<std::int32_t> train_time_;
    std::vector<std::int32_t> query_src_;
    std::vector<std::int32_t> query_candidates_;
    std::vector<std::int32_t> query_dst_;
    std::vector<std::int32_t> query_time_;
    std::vector<std::int32_t> query_availability_time_;
    std::size_t candidate_count_;
    double short_window_;
    std::vector<SourceState> states_;
    std::unordered_set<std::uint64_t> active_seen_;
    std::vector<std::uint8_t> full_counts_;
    std::vector<std::uint8_t> short_counts_;
    std::unordered_map<std::uint64_t, std::uint32_t> full_overflow_;
    std::unordered_map<std::uint64_t, std::uint32_t> short_overflow_;
    std::vector<std::uint32_t> full_popularity_;
    std::vector<std::uint32_t> short_popularity_;
    std::deque<CooccurEvent> short_cooccurrences_;
    std::deque<PopularityEvent> short_popularity_events_;
    std::vector<float> lift_;
    std::vector<std::int32_t> positive_popularity_;
    std::string progress_path_;
};

std::int64_t parse_integer(const char* value, const char* label) {
    try {
        return std::stoll(value);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid ") + label);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 18) {
            throw std::runtime_error(
                "usage: materializer train-src train-dst train-time train-rows "
                "query-src query-candidates query-dst query-time "
                "query-availability-time query-rows "
                "candidate-count short-window max-src max-dst lift-out pop-out "
                "progress");
        }
        const auto train_rows =
            static_cast<std::size_t>(parse_integer(argv[4], "train rows"));
        const auto query_rows =
            static_cast<std::size_t>(parse_integer(argv[10], "query rows"));
        const auto candidate_count =
            static_cast<std::size_t>(parse_integer(argv[11], "candidate count"));
        const auto short_window = std::stod(argv[12]);
        const auto maximum_source =
            static_cast<std::int32_t>(parse_integer(argv[13], "max source"));
        const auto maximum_destination =
            static_cast<std::int32_t>(parse_integer(argv[14], "max destination"));
        if (short_window <= 0.0 || maximum_source < 0
            || maximum_destination < 1) {
            throw std::runtime_error("invalid materialization dimensions");
        }

        auto train_src = read_binary<std::int32_t>(argv[1], train_rows);
        auto train_dst = read_binary<std::int32_t>(argv[2], train_rows);
        auto train_time = read_binary<std::int32_t>(argv[3], train_rows);
        auto query_src = read_binary<std::int32_t>(argv[5], query_rows);
        auto query_candidates = read_binary<std::int32_t>(
            argv[6], query_rows * candidate_count);
        auto query_dst = read_binary<std::int32_t>(argv[7], query_rows);
        auto query_time = read_binary<std::int32_t>(argv[8], query_rows);
        auto query_availability_time =
            read_binary<std::int32_t>(argv[9], query_rows);

        if (!std::is_sorted(train_time.begin(), train_time.end())
            || !std::is_sorted(query_time.begin(), query_time.end())
            || !std::is_sorted(
                query_availability_time.begin(),
                query_availability_time.end())) {
            throw std::runtime_error("input rows must be chronological");
        }
        for (std::size_t row = 0; row < query_rows; ++row) {
            if (query_availability_time[row] > query_time[row]) {
                throw std::runtime_error(
                    "query availability time exceeds observation time");
            }
        }
        Materializer materializer(
            std::move(train_src),
            std::move(train_dst),
            std::move(train_time),
            std::move(query_src),
            std::move(query_candidates),
            std::move(query_dst),
            std::move(query_time),
            std::move(query_availability_time),
            candidate_count,
            short_window,
            maximum_source,
            maximum_destination,
            argv[17]);
        materializer.run();
        write_binary(argv[15], materializer.lift());
        write_binary(argv[16], materializer.positive_popularity());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cooccur-lift materializer failed: "
                  << error.what() << std::endl;
        return 1;
    }
}
