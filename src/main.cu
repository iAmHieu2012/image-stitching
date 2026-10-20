/**
 * @file main.cu
 * @brief Command-line entry point and benchmarking harness for Panorama Image Stitching.
 *
 * Usage:
 *   ./stitch <left_image> <right_image> <output_image> --version <0|1|2|...> [--runs N] [--gt <gt_file>]
 *
 * Course: CSC14120 - Parallel Programming (Final Project)
 * Authors: Group XX
 */

#include "common.h"
#include "cuda_utils.cuh"
#include "versions.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::fprintf(
        stderr,
        "Usage: %s <left_image> <right_image> <output_image> --version <0|1|2|...> [--runs N] [--gt <gt_homography.txt>]\n",
        prog
    );
}

bool load_image_rgb(const std::string& path, Image& img) {
    int w = 0, h = 0, c = 0;
    uint8_t* raw = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!raw) {
        std::fprintf(stderr, "Error: Failed to load image '%s': %s\n", path.c_str(), stbi_failure_reason());
        return false;
    }
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data = raw;
    return true;
}

bool save_image_rgb(const std::string& path, const Image& img) {
    if (!img.data || img.width <= 0 || img.height <= 0) {
        return false;
    }
    const size_t dot = path.find_last_of('.');
    std::string ext = (dot != std::string::npos) ? path.substr(dot + 1) : "png";
    for (char& ch : ext) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    int ok = 0;
    if (ext == "jpg" || ext == "jpeg") {
        ok = stbi_write_jpg(path.c_str(), img.width, img.height, 3, img.data, 95);
    } else {
        ok = stbi_write_png(path.c_str(), img.width, img.height, 3, img.data, img.width * 3);
    }
    return ok != 0;
}

bool load_gt_homography(const std::string& path, Homography& H_gt) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return false;
    }
    for (int i = 0; i < 9; ++i) {
        if (!(ifs >> H_gt.h[i])) {
            return false;
        }
    }
    if (std::fabs(H_gt.h[8]) > 1e-12) {
        for (int i = 0; i < 9; ++i) {
            H_gt.h[i] /= H_gt.h[8];
        }
    }
    return true;
}

void evaluate_homography_error(const Homography& H_est, const Homography& H_gt, int wB, int hB) {
    double frob_sq = 0.0;
    for (int i = 0; i < 9; ++i) {
        const double diff = H_est.h[i] - H_gt.h[i];
        frob_sq += diff * diff;
    }
    const double frob_err = std::sqrt(frob_sq);

    // Compute mean reprojection error across 4 corners + center of Image B
    const double pts[5][2] = {
        {0.0, 0.0},
        {static_cast<double>(wB - 1), 0.0},
        {static_cast<double>(wB - 1), static_cast<double>(hB - 1)},
        {0.0, static_cast<double>(hB - 1)},
        {0.5 * (wB - 1), 0.5 * (hB - 1)}
    };

    double total_px_err = 0.0;
    double max_px_err = 0.0;
    for (int i = 0; i < 5; ++i) {
        const double x = pts[i][0];
        const double y = pts[i][1];

        const double w_est = H_est.h[6] * x + H_est.h[7] * y + H_est.h[8];
        const double x_est = (H_est.h[0] * x + H_est.h[1] * y + H_est.h[2]) / w_est;
        const double y_est = (H_est.h[3] * x + H_est.h[4] * y + H_est.h[5]) / w_est;

        const double w_gt = H_gt.h[6] * x + H_gt.h[7] * y + H_gt.h[8];
        const double x_gt = (H_gt.h[0] * x + H_gt.h[1] * y + H_gt.h[2]) / w_gt;
        const double y_gt = (H_gt.h[3] * x + H_gt.h[4] * y + H_gt.h[5]) / w_gt;

        const double err = std::hypot(x_est - x_gt, y_est - y_gt);
        total_px_err += err;
        max_px_err = std::max(max_px_err, err);
    }

    std::printf("\n--- Ground-Truth Homography Verification ---\n");
    std::printf("Frobenius Norm Error ||H_est - H_gt||_F : %.6f\n", frob_err);
    std::printf("Mean Point Reprojection Error           : %.4f px\n", total_px_err / 5.0);
    std::printf("Max  Point Reprojection Error           : %.4f px\n", max_px_err);
}

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::string left_path = argv[1];
    std::string right_path = argv[2];
    std::string out_path = argv[3];
    int version = 0;
    int num_runs = 5;
    std::string gt_path = "";

    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0 && i + 1 < argc) {
            version = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            num_runs = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--gt") == 0 && i + 1 < argc) {
            gt_path = argv[++i];
        } else {
            std::fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

#ifdef __CUDACC__
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::printf("GPU Device : %s\n", prop.name);
    } else {
        std::printf("GPU Device : None (CPU execution)\n");
    }
#else
    std::printf("Build Mode : CPU Sequential Build\n");
#endif

    Image imgA, imgB;
    if (!load_image_rgb(left_path, imgA) || !load_image_rgb(right_path, imgB)) {
        return EXIT_FAILURE;
    }

    std::printf("Input A    : %s (%d x %d)\n", left_path.c_str(), imgA.width, imgA.height);
    std::printf("Input B    : %s (%d x %d)\n", right_path.c_str(), imgB.width, imgB.height);
    std::printf("Version    : V%d (%d timed runs + 1 warm-up run)\n", version, num_runs);

    StitchParams params;
    StageTimings warmup_timings;
    StitchStats stats;

    auto run_version = [&](StageTimings& t, StitchStats& s) -> Image {
        switch (version) {
            case 0:
                return stitch_v0(imgA, imgB, params, t, s);
            default:
                std::fprintf(stderr, "Error: Version V%d is not implemented yet (Midterm includes V0).\n", version);
                std::exit(EXIT_FAILURE);
        }
    };

    // 1. Warm-up run (excluded from timing measurements)
    Image warmup_out = run_version(warmup_timings, stats);
    warmup_out.free_data();

    // 2. Timed runs
    StageTimings avg_t{};
    StageTimings min_t{};
    min_t.preprocess_ms = 1e18;
    min_t.detect_ms = 1e18;
    min_t.describe_ms = 1e18;
    min_t.match_ms = 1e18;
    min_t.homography_ms = 1e18;
    min_t.warp_blend_ms = 1e18;
    min_t.total_ms = 1e18;

    Image final_pano;
    for (int r = 0; r < num_runs; ++r) {
        StageTimings cur_t{};
        StitchStats cur_s{};
        Image out = run_version(cur_t, cur_s);

        avg_t.preprocess_ms += cur_t.preprocess_ms;
        avg_t.detect_ms     += cur_t.detect_ms;
        avg_t.describe_ms   += cur_t.describe_ms;
        avg_t.match_ms      += cur_t.match_ms;
        avg_t.homography_ms += cur_t.homography_ms;
        avg_t.warp_blend_ms += cur_t.warp_blend_ms;
        avg_t.total_ms      += cur_t.total_ms;

        min_t.preprocess_ms = std::min(min_t.preprocess_ms, cur_t.preprocess_ms);
        min_t.detect_ms     = std::min(min_t.detect_ms, cur_t.detect_ms);
        min_t.describe_ms   = std::min(min_t.describe_ms, cur_t.describe_ms);
        min_t.match_ms      = std::min(min_t.match_ms, cur_t.match_ms);
        min_t.homography_ms = std::min(min_t.homography_ms, cur_t.homography_ms);
        min_t.warp_blend_ms = std::min(min_t.warp_blend_ms, cur_t.warp_blend_ms);
        min_t.total_ms      = std::min(min_t.total_ms, cur_t.total_ms);

        if (r == num_runs - 1) {
            final_pano = out;
            stats = cur_s;
        } else {
            out.free_data();
        }
    }

    const double inv_runs = 1.0 / static_cast<double>(num_runs);
    avg_t.preprocess_ms *= inv_runs;
    avg_t.detect_ms     *= inv_runs;
    avg_t.describe_ms   *= inv_runs;
    avg_t.match_ms      *= inv_runs;
    avg_t.homography_ms *= inv_runs;
    avg_t.warp_blend_ms *= inv_runs;
    avg_t.total_ms      *= inv_runs;

    std::printf("\n--- Pipeline Statistics ---\n");
    std::printf("Keypoints (Image A / Image B) : %d / %d\n", stats.num_kp_A, stats.num_kp_B);
    std::printf("Matched Keypoint Pairs        : %d\n", stats.num_matches);
    std::printf("RANSAC Inliers                : %d\n", stats.num_inliers);
    std::printf("Output Panorama Size          : %d x %d\n", final_pano.width, final_pano.height);
    std::printf("Estimated Homography H (B -> A):\n");
    for (int r = 0; r < 3; ++r) {
        std::printf("  [ %12.6f  %12.6f  %12.6f ]\n",
                    stats.H.h[r * 3 + 0], stats.H.h[r * 3 + 1], stats.H.h[r * 3 + 2]);
    }

    if (!gt_path.empty()) {
        Homography H_gt;
        if (load_gt_homography(gt_path, H_gt)) {
            evaluate_homography_error(stats.H, H_gt, imgB.width, imgB.height);
        } else {
            std::fprintf(stderr, "Warning: Could not read ground-truth homography from '%s'\n", gt_path.c_str());
        }
    }

    const double sum_stages = avg_t.preprocess_ms + avg_t.detect_ms + avg_t.describe_ms +
                              avg_t.match_ms + avg_t.homography_ms + avg_t.warp_blend_ms;
    const double denom = (avg_t.total_ms > 0.0) ? avg_t.total_ms : 1.0;

    std::printf("\n--- Stage Timing Breakdown (Average of %d runs) ---\n", num_runs);
    std::printf("%-30s | %12s | %12s | %9s\n", "Stage", "Avg (ms)", "Min (ms)", "Share (%)");
    std::printf("------------------------------------------------------------------------\n");
    std::printf("%-30s | %12.3f | %12.3f | %8.2f%%\n",
                "1. Preprocessing", avg_t.preprocess_ms, min_t.preprocess_ms, 100.0 * avg_t.preprocess_ms / denom);
    std::printf("%-30s | %12.3f | %12.3f | %8.2f%%\n",
                "2. Keypoint Detection", avg_t.detect_ms, min_t.detect_ms, 100.0 * avg_t.detect_ms / denom);
    std::printf("%-30s | %12.3f | %12.3f | %8.2f%%\n",
                "3. Description (rBRIEF)", avg_t.describe_ms, min_t.describe_ms, 100.0 * avg_t.describe_ms / denom);
    std::printf("%-30s | %12.3f | %12.3f | %8.2f%%\n",
                "4. Feature Matching", avg_t.match_ms, min_t.match_ms, 100.0 * avg_t.match_ms / denom);
    std::printf("%-30s | %12.3f | %12.3f | %8.2f%%\n",
                "5. Homography (RANSAC)", avg_t.homography_ms, min_t.homography_ms, 100.0 * avg_t.homography_ms / denom);
    std::printf("%-30s | %12.3f | %12.3f | %8.2f%%\n",
                "6. Warping & Blending", avg_t.warp_blend_ms, min_t.warp_blend_ms, 100.0 * avg_t.warp_blend_ms / denom);
    std::printf("------------------------------------------------------------------------\n");
    std::printf("%-30s | %12.3f | %12.3f | %8.2f%%\n",
                "Total End-to-End Time", avg_t.total_ms, min_t.total_ms, 100.0);

    // Amdahl's Law Estimate:
    // Scenario A: Stages 1, 2, 3, 4, 6 on GPU (leaving Stage 5 on CPU or refining on CPU)
    // Scenario B: All 6 stages on GPU (only tiny host orchestration overhead remains)
    const double p_stages_12346 = (avg_t.preprocess_ms + avg_t.detect_ms + avg_t.describe_ms +
                                   avg_t.match_ms + avg_t.warp_blend_ms) / denom;
    const double p_all_6 = std::min(0.9999, sum_stages / denom);

    std::printf("\n--- Amdahl's Law Analysis ---\n");
    std::printf("Parallelizable fraction (Stages 1,2,3,4,6) : p = %.4f -> Max Speedup = %.2fx\n",
                p_stages_12346, 1.0 / std::max(1e-6, 1.0 - p_stages_12346));
    std::printf("Parallelizable fraction (All 6 Stages)     : p = %.4f -> Max Speedup = %.2fx\n",
                p_all_6, 1.0 / std::max(1e-6, 1.0 - p_all_6));

    if (!save_image_rgb(out_path, final_pano)) {
        std::fprintf(stderr, "Error: Failed to write output panorama to '%s'\n", out_path.c_str());
        final_pano.free_data();
        stbi_image_free(imgA.data);
        stbi_image_free(imgB.data);
        return EXIT_FAILURE;
    }
    std::printf("\nSaved stitched panorama to '%s'\n", out_path.c_str());

    final_pano.free_data();
    stbi_image_free(imgA.data);
    stbi_image_free(imgB.data);
    return EXIT_SUCCESS;
}
