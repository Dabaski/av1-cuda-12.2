#pragma once

#include <cstdint>
#include <cstddef>

#include <pixels.h>
#include <intra.h>
#include <transform.h>

namespace pipeline {

// Compose the 4x4 host block path, mirroring the SVT chain:
//   build_intra_predictors -> av1_subtract_block (int16, no clamping)
//   -> av1_tranform_two_d_core_c (4x4, shift {2,0,0})
// src window is taken from the luma Plane at (px, py); edges are supplied by
// the caller following buildIntraPredictors' convention.
void encodeBlock4x4(const pixels::Plane& plane, int px, int py, const std::uint8_t* aboveRef, int nTopPx,
                    int nTopRightPx, const std::uint8_t* leftRef, int nLeftPx, int nBottomLeftPx,
                    std::uint8_t aboveLeft, intra::PredictionMode mode, int angleDelta,
                    transforms::TxType txType, std::int32_t coeffs[16]);

std::string subtractCuSource();

}  // namespace pipeline