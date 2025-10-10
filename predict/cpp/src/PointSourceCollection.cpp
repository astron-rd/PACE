#include <predict/PointSourceCollection.h>

namespace {
size_t UpdateDirection(predict::Direction &direction, double ra, double dec,
                       size_t count) {
  direction.ra *= count;
  direction.dec *= count;

  direction.ra += ra;
  direction.dec += dec;

  direction.ra /= ++count;
  direction.dec /= count;
  return count;
}

inline int FindGroup(const std::vector<predict::Direction> &groups_direction,
                     double ra, double dec, double distance_square) {
  for (size_t i = 0; i < groups_direction.size(); i++) {
    if (groups_direction[i].angular_distance(ra, dec) <= distance_square) {
      return i;
    }
  }
  return -1;
}

} // namespace

namespace predict {
void PointSourceCollection::GroupSources(double max_angular_separation) {
  // This algorithm assumes the source are sorted by directions.
  // It should be extended to consider another situation.
  size_t current_group_id = 0;
  int current_sources_count = 1;
  std::vector<Direction> groups_directions;
  std::vector<size_t> groups_count;

  Direction current_group_direction =
      Direction(direction_vector[0].ra, direction_vector[0].dec);
  const double max_separation_square =
      max_angular_separation * max_angular_separation;
  for (size_t source_id = 0; source_id < Size(); source_id++) {
    const double angular_distance_square =
        current_group_direction.angular_distance(
            direction_vector.ra[source_id], direction_vector.dec[source_id]);
    // Direction is too far from current grousp direction
    if (angular_distance_square > max_angular_separation) {
      const int direction_id =
          FindGroup(groups_directions, direction_vector.ra[source_id],
                    direction_vector.dec[source_id], max_separation_square);
      if (direction_id >= 0) {
        groups_count[direction_id] = UpdateDirection(
            groups_directions[direction_id], direction_vector.ra[source_id],
            direction_vector.dec[source_id], groups_count[direction_id]);
        beam_id[source_id] = direction_id;
        current_group_direction = groups_directions[direction_id];
        current_sources_count = groups_count[direction_id];
      } else {
        // New group
        groups_directions.push_back(current_group_direction);
        groups_count.push_back(current_sources_count);
        current_group_id++;
        current_sources_count = 1;
        beam_id[source_id] = current_group_id;
        current_group_direction = direction_vector[source_id];
      }
    } else {
      beam_id[source_id] = current_group_id;
      //  Update current groups
      current_sources_count = UpdateDirection(
          current_group_direction, direction_vector.ra[source_id],
          direction_vector.dec[source_id], current_sources_count);
    }
  }

  UpdateBeams();
  // beam_directions = groups_directions;

  for (size_t i = 0; i < groups_directions.size(); ++i) {
    beam_directions[i] = groups_directions[i];
  }
}
} // namespace predict