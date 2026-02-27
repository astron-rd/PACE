#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
static std::string format_j2000(double coord, bool is_ra = false) {
  bool negative = coord < 0;
  int64_t abs_coord = static_cast<int64_t>(std::abs(coord));

  int hours_or_deg = abs_coord / 10000;
  int minutes = (abs_coord % 10000) / 100;
  int seconds = abs_coord % 100;

  std::ostringstream oss;
  if (negative)
    oss << "-";

  oss << std::setw(2) << std::setfill('0') << hours_or_deg << ":"
      << std::setw(2) << std::setfill('0') << minutes << ":" << std::setw(2)
      << std::setfill('0') << seconds;

  return oss.str();
}

std::string format_mjd(double mjd) {
  // Convert MJD to calendar date
  double jd = mjd + 2400000.5; // Convert to Julian Date
  long z = static_cast<long>(jd + 0.5);
  double f = (jd + 0.5) - z;

  long a = z + 1 +
           (z < 2299161 ? 0
                        : (static_cast<long>((z - 1867216.25) / 36524.25) -
                           static_cast<long>((z - 1867216.25) / 36524.25) / 4));
  long b = a + 1524;
  long c = static_cast<long>((b - 122.1) / 365.25);
  long d = static_cast<long>(365.25 * c);
  long e = static_cast<long>((b - d) / 30.6001);

  int day = b - d - static_cast<long>(30.6001 * e);
  int month = e < 14 ? e - 1 : e - 13;
  int year = month > 2 ? c - 4716 : c - 4715;

  int hours = static_cast<int>(f * 24);
  int mins = static_cast<int>((f * 24 - hours) * 60);
  double secs = ((f * 24 - hours) * 60 - mins) * 60;

  std::ostringstream oss;
  oss << year << "-" << std::setw(2) << std::setfill('0') << month << "-"
      << std::setw(2) << std::setfill('0') << day << " " << std::setw(2)
      << std::setfill('0') << hours << ":" << std::setw(2) << std::setfill('0')
      << mins << ":" << std::setw(5) << std::setfill('0') << std::fixed
      << std::setprecision(2) << secs << " UTC";
  return oss.str();
}

} // namespace

// SIGPROC filterbank header structure
struct FilterbankHeader {
  // Sizes
  int64_t header_size = 0;
  int64_t data_size = 0;

  // Data dimensions
  int nchans = 0;   // Number of channels
  int nsamples = 0; // Number of time samples
  int nbits = 0;    // Bits per sample
  int npols = 0;    // Number of polarizations

  // Instrument info
  int machine_id = 0;
  int telescope_id = 0;
  int nbeams = 0;
  int ibeam = 0;

  // Timing and frequency
  double tstart = 0.0; // Start MJD
  double tsamp = 0.0;  // Sampling time (seconds)
  double fch1 = 0.0;   // Frequency of channel 1 (MHz)
  double foff = 0.0;   // Channel bandwidth (MHz)

  // Source coordinates
  double src_raj = 0.0;       // Source RA (J2000)
  double src_dej = 0.0;       // Source Dec (J2000)
  double azimuth_angle = 0.0; // Azimuth at start
  double zenith_angle = 0.0;  // Zenith angle at start

  // Source name
  char source_name[80] = {};
};

static std::ostream &operator<<(std::ostream &os, const FilterbankHeader &h) {
  os << "=== SIGPROC Filterbank Header ===\n";
  os << "Header size:     " << h.header_size << " bytes\n";
  os << "Data size:       " << h.data_size << " bytes\n";
  os << "\n--- Data Dimensions ---\n";
  os << "Channels:        " << h.nchans << "\n";
  os << "Samples:         " << h.nsamples << "\n";
  os << "Bits/sample:     " << h.nbits << "\n";
  os << "Polarizations:   " << h.npols << "\n";
  os << "\n--- Instrument ---\n";
  os << "Machine ID:      " << h.machine_id << "\n";
  os << "Telescope ID:    " << h.telescope_id << "\n";
  os << "Beams:           " << h.nbeams << "\n";
  os << "Beam index:      " << h.ibeam << "\n";
  os << "\n--- Timing & Frequency ---\n";
  os << "Start MJD:       " << format_mjd(h.tstart) << "\n";
  os << "Sampling (s):    " << h.tsamp << "\n";
  os << "Freq ch1 (MHz):  " << h.fch1 << "\n";
  os << "Bandwidth (MHz): " << h.foff << "\n";
  os << "\n--- Coordinates ---\n";
  os << "RA (J2000):      " << format_j2000(h.src_raj, true) << "\n";
  os << "Dec (J2000):     " << format_j2000(h.src_dej) << "\n";
  os << "Azimuth angle:   " << h.azimuth_angle << "\n";
  os << "Zenith angle:    " << h.zenith_angle << "\n";
  os << "=================================\n";
  return os;
}

class FilterbankFile {
public:
  FilterbankFile(const std::string &filepath) {
    file_ = std::fopen(filepath.c_str(), "rb");
    if (!file_) {
      throw std::runtime_error("Cannot open file: " + filepath);
    }

    read_header();
    read_data();
  }

  ~FilterbankFile() {
    if (file_) {
      std::fclose(file_);
    }
  }

  void read_header() {
    int nbytes = 0;

    while (true) {
      int nchar;
      if (std::fread(&nchar, sizeof(int), 1, file_) != 1) {
        throw std::runtime_error("Unexpected end of file while reading header");
      }

      // Skip invalid strings
      if (nchar <= 1 || nchar >= 80) {
        continue;
      }

      nbytes += sizeof(int) + nchar;

      // Read the string
      std::string key(nchar, '\0');
      if (std::fread(&key[0], 1, nchar, file_) != static_cast<size_t>(nchar)) {
        throw std::runtime_error("Failed to read header key");
      }

      // Remove null terminator
      if (!key.empty() && key.back() == '\0') {
        key.pop_back();
      }

      // Check for end of header
      if (key == "HEADER_END") {
        break;
      }

      // Read parameter values based on key
      if (key == "tsamp") {
        header_.tsamp = read_value<double>();
      } else if (key == "tstart") {
        header_.tstart = read_value<double>();
      } else if (key == "fch1") {
        header_.fch1 = read_value<double>();
      } else if (key == "foff") {
        header_.foff = read_value<double>();
      } else if (key == "nchans") {
        header_.nchans = read_value<int>();
      } else if (key == "nifs") {
        header_.npols = read_value<int>();
      } else if (key == "nbits") {
        header_.nbits = read_value<int>();
      } else if (key == "nsamples") {
        header_.nsamples = read_value<int>();
      } else if (key == "az_start") {
        header_.azimuth_angle = read_value<double>();
      } else if (key == "za_start") {
        header_.zenith_angle = read_value<double>();
      } else if (key == "src_raj") {
        header_.src_raj = read_value<double>();
      } else if (key == "src_dej") {
        header_.src_dej = read_value<double>();
      } else if (key == "telescope_id") {
        header_.telescope_id = read_value<int>();
      } else if (key == "machine_id") {
        header_.machine_id = read_value<int>();
      } else if (key == "nbeams") {
        header_.nbeams = read_value<int>();
      } else if (key == "ibeam") {
        header_.ibeam = read_value<int>();
      }
    }

    // Calculate header size
    header_.header_size = std::ftell(file_);

    // Get file size and calculate data size
    std::fseek(file_, 0, SEEK_END);
    int64_t file_size = std::ftell(file_);
    header_.data_size = file_size - header_.header_size;

    // Calculate samples if not provided
    if (header_.nsamples == 0 && header_.nchans > 0 && header_.npols > 0 &&
        header_.nbits > 0) {
      int64_t sample_size =
          (header_.nchans * header_.npols * header_.nbits) / 8;
      if (sample_size > 0) {
        header_.nsamples = static_cast<int>(header_.data_size / sample_size);
      }
    }
  }

  void read_data() {
    // Seek to start of data
    std::fseek(file_, header_.header_size, SEEK_SET);

    // Read all data into memory
    data_.resize(header_.data_size);
    if (header_.data_size > 0) {
      size_t read = std::fread(data_.data(), 1, header_.data_size, file_);
      if (read != static_cast<size_t>(header_.data_size)) {
        throw std::runtime_error("Failed to read complete data buffer");
      }
    }
  }

  // Accessors
  const FilterbankHeader &header() const { return header_; }
  const std::vector<uint8_t> &data() const { return data_; }
  std::vector<uint8_t> &data() { return data_; }

  // Get pointer to raw data
  const uint8_t *data_ptr() const { return data_.data(); }
  uint8_t *data_ptr() { return data_.data(); }

  // Get data size
  size_t data_size() const { return data_.size(); }

private:
  std::FILE *file_ = nullptr;
  FilterbankHeader header_;
  std::vector<uint8_t> data_;

  // Read a string from file
  std::string read_string() {
    int nchar;
    if (std::fread(&nchar, sizeof(int), 1, file_) != 1) {
      throw std::runtime_error("Failed to read string length");
    }

    if (nchar < 1 || nchar >= 80) {
      return ""; // Invalid length
    }

    std::string str(nchar, '\0');
    if (std::fread(&str[0], 1, nchar, file_) != static_cast<size_t>(nchar)) {
      throw std::runtime_error("Failed to read string data");
    }

    // Remove null terminator if present
    if (!str.empty() && str.back() == '\0') {
      str.pop_back();
    }

    return str;
  }

  // Template to read a value of type T
  template <typename T> T read_value() {
    T value;
    if (std::fread(&value, sizeof(T), 1, file_) != 1) {
      throw std::runtime_error("Failed to read value");
    }
    return value;
  }
};

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <filterbank_file>" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    FilterbankFile fil(argv[1]);
    std::cout << fil.header();
    const uint8_t *ptr = fil.data_ptr();
    size_t size = fil.data_size();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}