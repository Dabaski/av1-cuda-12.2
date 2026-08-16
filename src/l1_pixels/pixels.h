#pragma once

#include <cstddef>
#include <vector>

#include <core.h>

namespace pixels {

class Plane {
public:
    Plane(int width, int height, int padding)
        : width_(width),
          height_(height),
          padding_(padding),
          data_(static_cast<std::size_t>(stride()) *
                static_cast<std::size_t>(height + 2 * padding)) {}

    int width() const { return width_; }
    int height() const { return height_; }
    int stride() const { return width_ + 2 * padding_; }

    core::Sample& at(int x, int y) {
        return data_[static_cast<std::size_t>((y + padding_) * stride() + (x + padding_))];
    }

    const core::Sample& at(int x, int y) const {
        return data_[static_cast<std::size_t>((y + padding_) * stride() + (x + padding_))];
    }

private:
    int width_;
    int height_;
    int padding_;
    std::vector<core::Sample> data_;
};

}  // namespace pixels
