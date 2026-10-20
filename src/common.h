/**
 * @file common.h
 * @brief Common data structures, parameters, and timing definitions for Panorama Image Stitching.
 *
 * Course: CSC14120 - Parallel Programming (Final Project)
 * Authors: Group XX
 */

#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/**
 * @brief Represents an RGB image in row-major interleaved format (R, G, B).
 */
struct Image {
    int width = 0;
    int height = 0;
    int channels = 3;
    uint8_t* data = nullptr;

    static Image allocate(int w, int h, int c = 3) {
        Image img;
        img.width = w;
        img.height = h;
        img.channels = c;
        size_t total = static_cast<size_t>(w) * h * c;
        img.data = static_cast<uint8_t*>(std::calloc(total, sizeof(uint8_t)));
        return img;
    }

    void free_data() {
        if (data) {
            std::free(data);
            data = nullptr;
        }
    }
};

/**
 * @brief Represents a detected 2D feature keypoint.
 */
struct Keypoint {
    float x = 0.0f;
    float y = 0.0f;
    float response = 0.0f;
    float angle = 0.0f; // Orientation in radians (for Rotated BRIEF)
};

/**
 * @brief 256-bit binary feature descriptor (8 x 32-bit unsigned integers).
 */
struct Descriptor {
    uint32_t bits[8];
};

/**
 * @brief Represents a matched pair of keypoints between Image A and Image B.
 */
struct Match {
    int idxA = -1;
    int idxB = -1;
    int distance = 256;
};

/**
 * @brief 3x3 Projective Homography matrix mapping points in Image B to Image A:
 *        p_A ~ H * p_B
 */
struct Homography {
    double h[9] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
};

/**
 * @brief Configurable pipeline parameters (kept identical across all versions V0, V1, V2...).
 */
struct StitchParams {
    int gaussian_radius = 2;        // 5x5 Gaussian kernel (radius = 2)
    float gaussian_sigma = 1.4f;    // Standard deviation for Gaussian blur
    float harris_k = 0.04f;         // Harris detector sensitivity factor
    float harris_thresh = 1e4f;     // Minimum Harris corner response threshold
    int nms_radius = 2;             // 5x5 Non-Maximum Suppression window (radius = 2)
    int max_keypoints = 3000;       // Top-K keypoints retained per image
    int patch_radius = 15;          // 31x31 patch radius for Rotated BRIEF
    float ratio_thresh = 0.75f;     // Lowe's ratio test threshold
    int max_hamming_dist = 64;      // Maximum acceptable Hamming distance
    int ransac_iters = 2000;        // Number of RANSAC iterations
    float ransac_thresh = 3.0f;     // RANSAC inlier reprojection threshold (pixels)
    uint32_t random_seed = 42;      // Fixed PRNG seed for reproducibility across versions
};

/**
 * @brief Execution timings (in milliseconds) for each of the 6 pipeline stages.
 */
struct StageTimings {
    double preprocess_ms = 0.0;     // Stage 1: RGB to Grayscale + Gaussian Blur
    double detect_ms = 0.0;         // Stage 2: Harris Corner + NMS + Top-K
    double describe_ms = 0.0;       // Stage 3: Rotated BRIEF 256-bit Description
    double match_ms = 0.0;          // Stage 4: Brute-Force Hamming Matching + Ratio Test
    double homography_ms = 0.0;     // Stage 5: RANSAC + 4-point Normalized DLT + Refinement
    double warp_blend_ms = 0.0;     // Stage 6: Inverse Warping + Bilinear + Feathering
    double h2d_ms = 0.0;            // Host-to-Device transfer time (0 for V0)
    double d2h_ms = 0.0;            // Device-to-Host transfer time (0 for V0)
    double total_ms = 0.0;          // End-to-end pipeline execution time
};

/**
 * @brief Diagnostic statistics from a pipeline execution.
 */
struct StitchStats {
    int num_kp_A = 0;
    int num_kp_B = 0;
    int num_matches = 0;
    int num_inliers = 0;
    Homography H;
};

#endif // COMMON_H
