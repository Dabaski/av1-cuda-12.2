#include <doctest.h>
#include <pixels.h>

TEST_CASE("plane reports its width") {
    pixels::Plane p(8, 4, 4);
    CHECK(p.width() == 8);
}

TEST_CASE("plane stride includes left and right padding") {
    pixels::Plane p(8, 4, 4);
    CHECK(p.stride() == 16);
}

TEST_CASE("plane reports its height") {
    pixels::Plane p(8, 4, 4);
    CHECK(p.height() == 4);
}

TEST_CASE("plane stores a sample at (0,0)") {
    pixels::Plane p(8, 4, 4);
    p.at(0, 0) = 42;
    CHECK(p.at(0, 0) == 42);
}
