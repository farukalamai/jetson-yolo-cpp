#pragma once

#include <unordered_map>
#include <unordered_set>
#include <Eigen/Dense>

struct CountLine {
    float x1, y1, x2, y2;
};

struct Counts {
    int entered = 0;
    int exited  = 0;
};

class LineCounter {
public:
    LineCounter(const CountLine& line) : line_(line) {}

    // Call each frame with the tracker output matrix (rows: x1,y1,x2,y2,id,conf,cls,det_idx)
    void update(const Eigen::MatrixXf& tracks) {
        for (int i = 0; i < tracks.rows(); ++i) {
            int id = static_cast<int>(tracks(i, 4));
            float cx = (tracks(i, 0) + tracks(i, 2)) / 2.0f;
            float cy = (tracks(i, 1) + tracks(i, 3)) / 2.0f;

            float side = sideOfLine(cx, cy);

            auto it = prev_side_.find(id);
            if (it == prev_side_.end()) {
                prev_side_[id] = side;
                continue;
            }

            float prev = it->second;
            // Crossed: sign changed and this id hasn't been counted yet
            if (prev * side < 0 && counted_.find(id) == counted_.end()) {
                counted_.insert(id);
                if (prev < 0) counts_.entered++;
                else          counts_.exited++;
            }
            it->second = side;
        }
    }

    const Counts& counts() const { return counts_; }

    void reset() {
        counts_ = {};
        prev_side_.clear();
        counted_.clear();
    }

private:
    // Returns positive/negative depending on which side of the line (cx,cy) is on.
    // Uses the cross product of the line vector with the point vector.
    float sideOfLine(float px, float py) const {
        float dx = line_.x2 - line_.x1;
        float dy = line_.y2 - line_.y1;
        return (px - line_.x1) * dy - (py - line_.y1) * dx;
    }

    CountLine line_;
    Counts counts_;
    std::unordered_map<int, float> prev_side_;
    std::unordered_set<int> counted_;
};
