/**
 * @file versions.h
 * @brief Function declarations for all panorama stitching pipeline versions (V0, V1, V2...).
 *
 * Course: CSC14120 - Parallel Programming (Final Project)
 * Authors: Group XX
 */

#ifndef VERSIONS_H
#define VERSIONS_H

#include "common.h"

/**
 * @brief V0: Single-threaded Sequential CPU baseline implementation.
 *
 * @param imgA      Left (reference) RGB image.
 * @param imgB      Right (target) RGB image to be warped into imgA's frame.
 * @param params    Pipeline hyperparameters.
 * @param timings   Output per-stage execution times (in milliseconds).
 * @param stats     Output pipeline statistics (keypoint counts, matches, inliers, Homography).
 * @return          Stitched panorama RGB image (caller is responsible for freeing via free_data()).
 */
Image stitch_v0(
    const Image& imgA,
    const Image& imgB,
    const StitchParams& params,
    StageTimings& timings,
    StitchStats& stats
);

#endif // VERSIONS_H
