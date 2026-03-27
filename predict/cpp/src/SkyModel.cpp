#include <predict/SkyModel.h>

#include <fstream>
#include <iostream>

#include <filesystem>
#include <predict/Spectrum.h>
#include <regex>
#include <vector>

std::string line_regex =
    //  1. Source name (letters, digits, underscore, dot, plus, minus)
    R"(([A-Za-z0-9_.+\-]+),\s*)"
    //  2. Source type
    R"((\w+),\s*)"
    //  3. Patch name
    R"(([\w_]+),\s*)"
    //  4. Right ascension (sexagesimal, with optional decimal seconds)
    R"(([-+]?\d+:\d+:\d+(?:\.\d+)?),\s*)"
    //  5. Declination (sexagesimal, with optional sign)
    R"(([-+]?\d+.\d+.\d+(?:\.\d+)?),\s*)"
    //  6. I in Jy
    R"(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*)"
    //  7. Spectral parameters (array)
    R"(\[([^\]]*)\],\s*)"
    //  8. Are spectral parameter logarithmic? (true/false)
    R"((true|false),\s*)"
    //  9. ReferenceFrequency
    R"(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*)"
    // 10. MajorAxis
    R"(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*)"
    // 11. MinorAxis
    R"(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*)"
    // 12. Orientation
    R"(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*)"
    // 13. Q in Jy
    R"(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*)"
    // 14. U in Jy
    R"(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*)"
    // 15. V in Jy
    R"(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))";

namespace {

std::string LTrim(std::string s) { return s.erase(0, s.find(' ') + 1); }

void SplitLineFast(std::string line, std::vector<std::string> &tokens) {
  tokens.clear();
  std::string subloop_string = "";
  while (!line.empty()) {
    auto found = line.find(',');
    bool is_subloop = false;
    if (found < line.size()) {
      std::string token = line.substr(0, found);
      line = line.substr(found + 1);
      token = LTrim(token);

      if (token[0] == '[') {
        is_subloop = true;
        token = token.erase(0, 1); // Remove the opening bracket
      }

      if (token[token.size() - 1] == ']') {
        is_subloop = false;
        token = token.substr(0, token.size() - 2); // Remove the closing bracket
        subloop_string += token;

        tokens.push_back(subloop_string);
        subloop_string.clear();
        continue;
      }

      if (!is_subloop) {
        tokens.push_back(token);
      } else {
        subloop_string += token + ",";
      }
    } else {
      tokens.push_back(LTrim(line));
      break;
    }
  }
}

double ParseHourAngle(const std::string &value) {
  std::string hour_angle_regex_string = R"(([-]?\d+):(\d+):(\d+\.?\d+))";
  std::regex hour_angle_regex(hour_angle_regex_string);
  std::smatch match;
  std::regex_match(value, match, hour_angle_regex);
  if (match.size() > 1) {
    double hours = std::stod(match[1]);
    double minutes = std::stod(match[2]);
    double seconds = std::stod(match[3]);
    return (hours * 3600.0 + minutes * 60.0 + seconds) / 24.0 / 3600.0 * 2.0 *
           M_PI;
  } else {
    std::cerr << "Invalid hour angle format: " << value << std::endl;
    return NAN;
  }
}

double ParseDeclination(const std::string &value) {
  std::string hour_angle_regex_string = R"(([-]?\d+).(\d+).(\d+\.?\d+))";
  std::regex hour_angle_regex(hour_angle_regex_string);
  std::smatch match;
  std::regex_match(value, match, hour_angle_regex);
  if (match.size() > 1) {
    double degrees = std::stod(match[1]);
    double primes = std::stod(match[2]);
    double seconds = std::stod(match[3]);
    return (degrees + primes / 60.0 + seconds / 3600.0) / 180.0 * M_PI;
  } else {
    std::cerr << "Invalid hour angle format: " << value << std::endl;
    return NAN;
  }
}

} // namespace

namespace predict {
void ParseSkyModel(const std::string &skymodel_path,
                   GaussianSourceCollection &gaussians_sources,
                   PointSourceCollection &point_sources) {
  std::cout << "Parsing sky model from: " << skymodel_path << std::endl;

  std::filesystem::path path(skymodel_path);
  if (!std::filesystem::exists(path)) {
    std::cerr << "Sky model file does not exist: " << skymodel_path
              << std::endl;
    throw std::runtime_error("Sky model file not found");
  }
  std::ifstream file(skymodel_path);
  std::vector<string> stash;
  std::map<std::string, int> patch_to_beam;
  int current_beam = 0;

  std::cout << "Reading file in memory: " << std::endl;
  while (!file.eof()) {
    std::string line;
    std::vector<std::string> tokens;
    std::getline(file, line);

    if (line[0] == ' ' && line[1] == ',') {
      std::vector<std::string> tokens;
      SplitLineFast(line, tokens);
      const std::string patch_name = tokens[2];
      double ra = ParseHourAngle(tokens[3]);
      double dec = ParseDeclination(tokens[4]);

      point_sources.AddBeamDirection(current_beam, Direction(ra, dec));
      gaussians_sources.AddBeamDirection(current_beam, Direction(ra, dec));
      patch_to_beam[patch_name] = current_beam++;
    }
    if (line.empty() || line[0] == '#' || line[0] == ' ' || line[0] == 'F') {
      continue; // Skip empty lines and comments
    }
    stash.emplace_back(line);
  }
  std::cout << "File read " << std::endl;
  std::cout << "Start parsing" << std::endl;

  for (size_t l = 0; l < stash.size(); l++) {
    std::string &line = stash[l];
    std::vector<std::string> tokens;

    SplitLineFast(line, tokens);

    if (tokens.size() < 15) {
      std::cerr << "Invalid line format: " << line << std::endl;
      throw std::runtime_error("Invalid line format");
    }
    // Parse the tokens
    // std::string source_name = tokens[0];
    std::string source_type = tokens[1];
    std::string patch_name = tokens[2];
    int beam_id = patch_to_beam[patch_name];
    double ra = ParseHourAngle(tokens[3]);
    double dec = ParseDeclination(tokens[4]);

    double I = std::stod(tokens[5]);
    std::vector<double> spectral_terms;
    if (tokens[6] != "") {
      std::stringstream ss(tokens[6]);
      std::string term;
      while (std::getline(ss, term, ',')) {
        spectral_terms.push_back(std::stod(term));
      }
    }

    bool is_logarithmic = (tokens[7] == "true");
    double reference_frequency = std::stod(tokens[8]);
    double major_axis = std::stod(tokens[9]);
    double minor_axis = std::stod(tokens[10]);
    double orientation = std::stod(tokens[11]);
    double Q = std::stod(tokens[12]);
    double U = std::stod(tokens[13]);
    double V = std::stod(tokens[14]);
    Spectrum spectrum;

    spectrum.SetReferenceFlux(Stokes(I, Q, U, V));
    spectrum.SetSpectralTerms(reference_frequency, is_logarithmic,
                              spectral_terms);
    if (source_type == "GAUSSIAN") {
      const GaussianSource source(Direction(ra, dec), spectrum, orientation,
                                  false, minor_axis, major_axis, beam_id);
      gaussians_sources.Add(source);
    } else if (source_type == "POINT") {
      const PointSource source(Direction(ra, dec), spectrum, beam_id);

      point_sources.Add(source);
    } else {
      std::cerr << "Source type " << source_type << " not supported"
                << std::endl;
      std::string exception_string =
          "Source type " + source_type + " not supported";
      throw std::runtime_error(exception_string);
    }
  }
  point_sources.UpdateBeams();
  gaussians_sources.UpdateBeams();
}
} // namespace predict