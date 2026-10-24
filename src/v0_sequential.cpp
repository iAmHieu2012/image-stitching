/**
 * @file v0_sequential.cpp
 * @brief Phiên bản V0: Cài đặt tuần tự trên 1 luồng CPU cho toàn bộ 6 bước ghép ảnh Panorama.
 *
 * Tác giả: Group 18
 *
 * Nguồn tham khảo thuật toán:
 * 1. Harris, C., & Stephens, M. (1988). "A Combined Corner and Edge Detector." Alvey Vision Conference.
 * 2. Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). "ORB: An efficient alternative to SIFT or SURF." ICCV.
 * 3. Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints." IJCV.
 * 4. Hartley, R., & Zisserman, A. (2004). "Multiple View Geometry in Computer Vision" (2nd ed.).
 *    Cambridge University Press (Normalized DLT & RANSAC Homography estimation).
 * 5. Szeliski, R. (2022). "Computer Vision: Algorithms and Applications" (2nd ed.). Springer.
 * 6. Sơn T. Trần, Việt .H Võ, Slide môn Thị giác máy tính (HCMUS)
 */

#include "versions.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

// Tính khoảng thời gian chạy (đơn vị: mili-giây) giữa 2 mốc thời gian
inline double elapsed_ms(const Clock::time_point& t0, const Clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Kẹp giá trị chỉ số v nằm trong đoạn [lo, hi] để xử lý biên ảnh an toàn
inline int clamp_idx(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// ============================================================================
// STAGE 1: TIỀN XỬ LÝ (Chuyển RGB sang ảnh xám Grayscale + Làm mờ Gaussian 2D)
// ============================================================================

/**
 * @brief Chuyển ảnh màu RGB (uint8_t) sang ảnh đa mức xám (float trong đoạn [0, 255]).
 */
void rgb_to_grayscale(const Image& img, std::vector<float>& gray) {
    const int w = img.width;
    const int h = img.height;
    gray.resize(static_cast<size_t>(w) * h);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = (static_cast<size_t>(y) * w + x) * img.channels;
            const float r = static_cast<float>(img.data[idx + 0]);
            const float g = static_cast<float>(img.data[idx + 1]);
            const float b = static_cast<float>(img.data[idx + 2]);
            // Công thức độ chói chuẩn ITU-R BT.601
            gray[static_cast<size_t>(y) * w + x] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
}

/**
 * @brief Tạo ma trận nhân lọc (kernel) Gaussian 2D kích thước (2*radius + 1) x (2*radius + 1),
 *        chuẩn hóa sao cho tổng các trọng số bằng 1.0f.
 */
std::vector<float> make_gaussian_kernel_2d(int radius, float sigma) {
    const int ksize = 2 * radius + 1;
    std::vector<float> kernel(ksize * ksize);
    float sum = 0.0f;
    const float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float val = std::exp(-static_cast<float>(dx * dx + dy * dy) * inv_2sigma2);
            kernel[(dy + radius) * ksize + (dx + radius)] = val;
            sum += val;
        }
    }
    for (float& v : kernel) {
        v /= sum;
    }
    return kernel;
}

/**
 * @brief Tích chập 2D làm mờ ảnh xám bằng bộ lọc Gaussian để giảm nhiễu hạt trước khi tính đạo hàm.
 */
void gaussian_blur_2d(
    const std::vector<float>& src,
    std::vector<float>& dst,
    int w,
    int h,
    int radius,
    const std::vector<float>& kernel
) {
    const int ksize = 2 * radius + 1;
    dst.resize(static_cast<size_t>(w) * h);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy) {
                const int sy = clamp_idx(y + dy, 0, h - 1);
                const int ky = (dy + radius) * ksize;
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int sx = clamp_idx(x + dx, 0, w - 1);
                    acc += src[static_cast<size_t>(sy) * w + sx] * kernel[ky + (dx + radius)];
                }
            }
            dst[static_cast<size_t>(y) * w + x] = acc;
        }
    }
}

// ============================================================================
// STAGE 2: PHÁT HIỆN ĐIỂM ĐẶC TRƯNG (Harris Corner Detector + NMS + Top-K)
// ============================================================================

/**
 * @brief Phát hiện các góc đặc trưng bằng bộ dò góc Harris, triệt tiêu phi cực đại (NMS)
 *        và chọn ra tối đa Top-K điểm có độ phản hồi góc lớn nhất.
 */
void detect_harris_keypoints(
    const std::vector<float>& blurred,
    int w,
    int h,
    const StitchParams& params,
    std::vector<Keypoint>& keypoints
) {
    const size_t total_pixels = static_cast<size_t>(w) * h;
    std::vector<float> ix2(total_pixels, 0.0f);
    std::vector<float> iy2(total_pixels, 0.0f);
    std::vector<float> ixy(total_pixels, 0.0f);

    // Bước 2.1: Tính đạo hàm bậc nhất theo chiều ngang (gx) và dọc (gy) bằng bộ lọc Sobel 3x3
    // đồng thời tính các tích Ix^2, Iy^2, Ix*Iy tại từng pixel
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const float p00 = blurred[static_cast<size_t>(y - 1) * w + (x - 1)];
            const float p01 = blurred[static_cast<size_t>(y - 1) * w + x];
            const float p02 = blurred[static_cast<size_t>(y - 1) * w + (x + 1)];
            const float p10 = blurred[static_cast<size_t>(y) * w + (x - 1)];
            const float p12 = blurred[static_cast<size_t>(y) * w + (x + 1)];
            const float p20 = blurred[static_cast<size_t>(y + 1) * w + (x - 1)];
            const float p21 = blurred[static_cast<size_t>(y + 1) * w + x];
            const float p22 = blurred[static_cast<size_t>(y + 1) * w + (x + 1)];

            const float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
            const float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);

            const size_t idx = static_cast<size_t>(y) * w + x;
            ix2[idx] = gx * gx;
            iy2[idx] = gy * gy;
            ixy[idx] = gx * gy;
        }
    }

    // Bước 2.2: Cộng dồn ma trận cấu trúc M trong cửa sổ 3x3 và tính độ phản hồi Harris:
    // R = det(M) - k * (trace(M))^2
    std::vector<float> response(total_pixels, 0.0f);
    // Chừa lề biên tối thiểu 16 pixel để bước trích xuất vùng 31x31 ở Stage 3 không tràn ra ngoài ảnh
    const int border = std::max(params.patch_radius + 1, 16);

    for (int y = border; y < h - border; ++y) {
        for (int x = border; x < w - border; ++x) {
            float sxx = 0.0f;
            float syy = 0.0f;
            float sxy = 0.0f;

            for (int dy = -1; dy <= 1; ++dy) {
                const size_t row_offset = static_cast<size_t>(y + dy) * w;
                for (int dx = -1; dx <= 1; ++dx) {
                    const size_t idx = row_offset + (x + dx);
                    sxx += ix2[idx];
                    syy += iy2[idx];
                    sxy += ixy[idx];
                }
            }

            const float det = sxx * syy - sxy * sxy;
            const float trace = sxx + syy;
            response[static_cast<size_t>(y) * w + x] = det - params.harris_k * trace * trace;
        }
    }

    // Bước 2.3: Triệt tiêu phi cực đại (Non-Maximum Suppression - NMS) trong lân cận (2*r+1)x(2*r+1)
    keypoints.clear();
    const int r = params.nms_radius;

    for (int y = border; y < h - border; ++y) {
        for (int x = border; x < w - border; ++x) {
            const float val = response[static_cast<size_t>(y) * w + x];
            if (val <= params.harris_thresh) {
                continue;
            }

            bool is_max = true;
            for (int dy = -r; dy <= r && is_max; ++dy) {
                const size_t row_offset = static_cast<size_t>(y + dy) * w;
                for (int dx = -r; dx <= r; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (response[row_offset + (x + dx)] >= val) {
                        is_max = false;
                        break;
                    }
                }
            }

            if (is_max) {
                Keypoint kp;
                kp.x = static_cast<float>(x);
                kp.y = static_cast<float>(y);
                kp.response = val;
                kp.angle = 0.0f;
                keypoints.push_back(kp);
            }
        }
    }

    // Bước 2.4: Sắp xếp giảm dần theo độ phản hồi R và chỉ giữ lại tối đa Top-K keypoints mạnh nhất
    if (static_cast<int>(keypoints.size()) > params.max_keypoints) {
        std::partial_sort(
            keypoints.begin(),
            keypoints.begin() + params.max_keypoints,
            keypoints.end(),
            [](const Keypoint& a, const Keypoint& b) {
                return a.response > b.response;
            }
        );
        keypoints.resize(params.max_keypoints);
    } else {
        std::sort(
            keypoints.begin(),
            keypoints.end(),
            [](const Keypoint& a, const Keypoint& b) {
                return a.response > b.response;
            }
        );
    }
}

// ============================================================================
// STAGE 3: MÔ TẢ ĐẶC TRƯNG (Rotated BRIEF 256-bit theo chuẩn ORB)
// ============================================================================

struct BriefPair {
    int8_t dx1, dy1;
    int8_t dx2, dy2;
};

/**
 * @brief Khởi tạo cố định 256 cặp tọa độ lấy mẫu trong [-10, 10] x [-10, 10] bằng seed cố định.
 *        Khi xoay theo bất kỳ góc theta nào, tọa độ sau xoay vẫn nằm gọn trong bán kính 15 (patch 31x31).
 */
std::vector<BriefPair> generate_brief_pattern(uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 4.5f);
    std::vector<BriefPair> pattern(256);

    for (int i = 0; i < 256; ++i) {
        auto sample_coord = [&]() -> int8_t {
            int v = static_cast<int>(std::round(dist(rng)));
            return static_cast<int8_t>(clamp_idx(v, -10, 10));
        };
        pattern[i].dx1 = sample_coord();
        pattern[i].dy1 = sample_coord();
        pattern[i].dx2 = sample_coord();
        pattern[i].dy2 = sample_coord();
        if (pattern[i].dx1 == pattern[i].dx2 && pattern[i].dy1 == pattern[i].dy2) {
            pattern[i].dx2 = static_cast<int8_t>((pattern[i].dx1 < 10) ? (pattern[i].dx1 + 1) : (pattern[i].dx1 - 1));
        }
    }
    return pattern;
}

/**
 * @brief Tính góc hướng chủ đạo (Intensity Centroid) và trích xuất descriptor nhị phân 256-bit (8 x uint32_t)
 *        cho từng keypoint.
 */
void compute_rotated_brief_descriptors(
    const std::vector<float>& blurred,
    int w,
    int h,
    const std::vector<BriefPair>& pattern,
    int patch_radius,
    std::vector<Keypoint>& keypoints,
    std::vector<Descriptor>& descriptors
) {
    descriptors.resize(keypoints.size());
    const int r2 = patch_radius * patch_radius;

    for (size_t i = 0; i < keypoints.size(); ++i) {
        const int cx = static_cast<int>(keypoints[i].x);
        const int cy = static_cast<int>(keypoints[i].y);

        // Bước 3.1: Tính góc hướng của keypoint bằng trọng tâm cường độ sáng (Intensity Centroid)
        float m10 = 0.0f;
        float m01 = 0.0f;
        for (int dy = -patch_radius; dy <= patch_radius; ++dy) {
            const int py = clamp_idx(cy + dy, 0, h - 1);
            for (int dx = -patch_radius; dx <= patch_radius; ++dx) {
                if (dx * dx + dy * dy <= r2) {
                    const int px = clamp_idx(cx + dx, 0, w - 1);
                    const float val = blurred[static_cast<size_t>(py) * w + px];
                    m10 += static_cast<float>(dx) * val;
                    m01 += static_cast<float>(dy) * val;
                }
            }
        }
        const float angle = std::atan2(m01, m10);
        keypoints[i].angle = angle;

        const float cos_t = std::cos(angle);
        const float sin_t = std::sin(angle);

        // Bước 3.2: Xoay 256 cặp điểm mẫu theo góc `angle` và so sánh độ sáng để tạo 256-bit descriptor
        Descriptor desc{};
        for (int k = 0; k < 8; ++k) {
            desc.bits[k] = 0u;
        }

        for (int b = 0; b < 256; ++b) {
            const BriefPair& bp = pattern[b];

            const int rx1 = static_cast<int>(std::round(bp.dx1 * cos_t - bp.dy1 * sin_t));
            const int ry1 = static_cast<int>(std::round(bp.dx1 * sin_t + bp.dy1 * cos_t));
            const int rx2 = static_cast<int>(std::round(bp.dx2 * cos_t - bp.dy2 * sin_t));
            const int ry2 = static_cast<int>(std::round(bp.dx2 * sin_t + bp.dy2 * cos_t));

            const int x1 = clamp_idx(cx + rx1, 0, w - 1);
            const int y1 = clamp_idx(cy + ry1, 0, h - 1);
            const int x2 = clamp_idx(cx + rx2, 0, w - 1);
            const int y2 = clamp_idx(cy + ry2, 0, h - 1);

            const float v1 = blurred[static_cast<size_t>(y1) * w + x1];
            const float v2 = blurred[static_cast<size_t>(y2) * w + x2];

            if (v1 < v2) {
                // Bật bit thứ (b % 32) trong từ 32-bit thứ (b / 32)
                desc.bits[b >> 5] |= (1u << (b & 31));
            }
        }
        descriptors[i] = desc;
    }
}

// ============================================================================
// STAGE 4: SO KHỚP ĐẶC TRƯNG (Brute-Force Hamming + Lowe's Ratio Test + Cross-Check)
// ============================================================================

/**
 * @brief Tính khoảng cách Hamming (số bit khác nhau) giữa hai descriptor 256-bit.
 */
inline int hamming_distance_256(const Descriptor& a, const Descriptor& b) {
    int dist = 0;
    for (int k = 0; k < 8; ++k) {
        dist += __builtin_popcount(a.bits[k] ^ b.bits[k]);
    }
    return dist;
}

/**
 * @brief So khớp vét cạn (Brute-Force) giữa tập descriptor của ảnh A và ảnh B,
 *        kết hợp kiểm định tỉ số Lowe (Ratio Test) và kiểm tra chéo 2 chiều (Mutual Cross-Check).
 */
void match_descriptors(
    const std::vector<Descriptor>& descA,
    const std::vector<Descriptor>& descB,
    const StitchParams& params,
    std::vector<Match>& matches
) {
    matches.clear();
    if (descA.empty() || descB.size() < 2) {
        return;
    }

    const int nA = static_cast<int>(descA.size());
    const int nB = static_cast<int>(descB.size());

    // Chiều thuận A -> B: Tìm điểm gần nhất (best_d) và gần nhì (second_d), áp dụng Lowe's Ratio Test
    std::vector<int> best_idx_AtoB(nA, -1);
    std::vector<int> best_dist_AtoB(nA, 256);

    for (int i = 0; i < nA; ++i) {
        int best_d = 256;
        int second_d = 256;
        int best_j = -1;

        for (int j = 0; j < nB; ++j) {
            const int d = hamming_distance_256(descA[i], descB[j]);
            if (d < best_d) {
                second_d = best_d;
                best_d = d;
                best_j = j;
            } else if (d < second_d) {
                second_d = d;
            }
        }

        if (best_j >= 0 &&
            best_d <= params.max_hamming_dist &&
            static_cast<float>(best_d) < params.ratio_thresh * static_cast<float>(second_d)) {
            best_idx_AtoB[i] = best_j;
            best_dist_AtoB[i] = best_d;
        }
    }

    // Chiều ngược B -> A: Tìm láng giềng gần nhất trên A cho mỗi điểm trên B để kiểm tra chéo (Cross-Check)
    std::vector<int> best_idx_BtoA(nB, -1);
    for (int j = 0; j < nB; ++j) {
        int best_d = 256;
        int best_i = -1;
        for (int i = 0; i < nA; ++i) {
            const int d = hamming_distance_256(descA[i], descB[j]);
            if (d < best_d) {
                best_d = d;
                best_i = i;
            }
        }
        best_idx_BtoA[j] = best_i;
    }

    // Chỉ giữ lại các cặp (i, j) khớp đồng thuận cả 2 chiều
    for (int i = 0; i < nA; ++i) {
        const int j = best_idx_AtoB[i];
        if (j >= 0 && best_idx_BtoA[j] == i) {
            Match m;
            m.idxA = i;
            m.idxB = j;
            m.distance = best_dist_AtoB[i];
            matches.push_back(m);
        }
    }
}

// ============================================================================
// STAGE 5: ƯỚC LƯỢNG MA TRẬN HOMOGRAPHY (RANSAC + Normalized 4-Point DLT + Refinement)
// ============================================================================

/**
 * @brief Giải hệ phương trình tuyến tính 8 ẩn (ma trận mở rộng 8x9) bằng phương pháp
 *        Khử Gauss có chọn phần tử trụ lớn nhất theo cột (Partial Pivoting).
 */
bool solve_linear_8x8(double A[8][9], double x_out[8]) {
    for (int col = 0; col < 8; ++col) {
        // Tìm hàng có trị tuyệt đối lớn nhất ở cột `col` làm phần tử trụ (pivot)
        int pivot_row = col;
        double max_val = std::fabs(A[col][col]);
        for (int row = col + 1; row < 8; ++row) {
            const double v = std::fabs(A[row][col]);
            if (v > max_val) {
                max_val = v;
                pivot_row = row;
            }
        }

        if (max_val < 1e-11) {
            return false; // Ma trận suy biến (4 điểm thẳng hàng hoặc trùng nhau)
        }

        if (pivot_row != col) {
            for (int k = col; k <= 8; ++k) {
                std::swap(A[col][k], A[pivot_row][k]);
            }
        }

        // Khử xuôi các hàng bên dưới hàng trụ
        const double pivot = A[col][col];
        for (int row = col + 1; row < 8; ++row) {
            const double factor = A[row][col] / pivot;
            for (int k = col; k <= 8; ++k) {
                A[row][k] -= factor * A[col][k];
            }
        }
    }

    // Thế ngược từ hàng 7 về hàng 0 để tìm nghiệm x_out[0..7]
    for (int row = 7; row >= 0; --row) {
        double sum = A[row][8];
        for (int col = row + 1; col < 8; ++col) {
            sum -= A[row][col] * x_out[col];
        }
        x_out[row] = sum / A[row][row];
    }
    return true;
}

/**
 * @brief Ước lượng ma trận Homography 3x3 H (ánh xạ p_A ~ H * p_B) từ N >= 4 cặp điểm tương ứng
 *        bằng phương pháp Chuẩn hóa Hartley (Hartley Normalization) kết hợp hệ phương trình chuẩn DLT.
 */
bool compute_normalized_dlt_homography(
    const std::vector<Keypoint>& kpA,
    const std::vector<Keypoint>& kpB,
    const std::vector<Match>& matches,
    const int* sample_indices,
    int num_samples,
    Homography& H_out
) {
    if (num_samples < 4) return false;

    // 1. Tính tọa độ trọng tâm của tập điểm trên ảnh A và ảnh B
    double mean_xA = 0.0, mean_yA = 0.0;
    double mean_xB = 0.0, mean_yB = 0.0;
    for (int k = 0; k < num_samples; ++k) {
        const Match& m = matches[sample_indices[k]];
        mean_xA += kpA[m.idxA].x;
        mean_yA += kpA[m.idxA].y;
        mean_xB += kpB[m.idxB].x;
        mean_yB += kpB[m.idxB].y;
    }
    mean_xA /= num_samples;
    mean_yA /= num_samples;
    mean_xB /= num_samples;
    mean_yB /= num_samples;

    // 2. Tính khoảng cách trung bình từ các điểm tới trọng tâm để co giãn về sqrt(2)
    double dist_A = 0.0, dist_B = 0.0;
    for (int k = 0; k < num_samples; ++k) {
        const Match& m = matches[sample_indices[k]];
        const double dxA = kpA[m.idxA].x - mean_xA;
        const double dyA = kpA[m.idxA].y - mean_yA;
        const double dxB = kpB[m.idxB].x - mean_xB;
        const double dyB = kpB[m.idxB].y - mean_yB;
        dist_A += std::sqrt(dxA * dxA + dyA * dyA);
        dist_B += std::sqrt(dxB * dxB + dyB * dyB);
    }
    dist_A /= num_samples;
    dist_B /= num_samples;

    if (dist_A < 1e-6 || dist_B < 1e-6) {
        return false;
    }

    const double scale_A = std::sqrt(2.0) / dist_A;
    const double scale_B = std::sqrt(2.0) / dist_B;

    // 3. Lập hệ phương trình chuẩn bình phương tối thiểu (A^T * A) * h = A^T * b cho 8 ẩn (cố định h_8 = 1)
    double ATA[8][9] = {{0.0}};

    for (int k = 0; k < num_samples; ++k) {
        const Match& m = matches[sample_indices[k]];
        const double uA = (kpA[m.idxA].x - mean_xA) * scale_A;
        const double vA = (kpA[m.idxA].y - mean_yA) * scale_A;
        const double uB = (kpB[m.idxB].x - mean_xB) * scale_B;
        const double vB = (kpB[m.idxB].y - mean_yB) * scale_B;

        // Phương trình ngang: [ uB, vB, 1,  0,  0, 0, -uA*uB, -uA*vB ] * h = uA
        const double r1[8] = { uB, vB, 1.0, 0.0, 0.0, 0.0, -uA * uB, -uA * vB };
        const double b1 = uA;

        // Phương trình dọc:   [  0,  0, 0, uB, vB, 1, -vA*uB, -vA*vB ] * h = vA
        const double r2[8] = { 0.0, 0.0, 0.0, uB, vB, 1.0, -vA * uB, -vA * vB };
        const double b2 = vA;

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                ATA[i][j] += r1[i] * r1[j] + r2[i] * r2[j];
            }
            ATA[i][8] += r1[i] * b1 + r2[i] * b2;
        }
    }

    double h_norm[8];
    if (!solve_linear_8x8(ATA, h_norm)) {
        return false;
    }

    const double Hn[9] = {
        h_norm[0], h_norm[1], h_norm[2],
        h_norm[3], h_norm[4], h_norm[5],
        h_norm[6], h_norm[7], 1.0
    };

    // 4. Khử chuẩn hóa (Denormalization): H = T_A^{-1} * Hn * T_B
    const double TB[9] = {
        scale_B, 0.0, -scale_B * mean_xB,
        0.0, scale_B, -scale_B * mean_yB,
        0.0, 0.0, 1.0
    };
    const double TA_inv[9] = {
        1.0 / scale_A, 0.0, mean_xA,
        0.0, 1.0 / scale_A, mean_yA,
        0.0, 0.0, 1.0
    };

    double temp[9] = {0.0};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            temp[r * 3 + c] = Hn[r * 3 + 0] * TB[0 * 3 + c] +
                              Hn[r * 3 + 1] * TB[1 * 3 + c] +
                              Hn[r * 3 + 2] * TB[2 * 3 + c];
        }
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            H_out.h[r * 3 + c] = TA_inv[r * 3 + 0] * temp[0 * 3 + c] +
                                 TA_inv[r * 3 + 1] * temp[1 * 3 + c] +
                                 TA_inv[r * 3 + 2] * temp[2 * 3 + c];
        }
    }

    if (std::fabs(H_out.h[8]) < 1e-12) {
        return false;
    }
    // Chuẩn hóa lại ma trận sao cho phần tử cuối H[2][2] (tức h[8]) = 1.0
    const double inv_h8 = 1.0 / H_out.h[8];
    for (int i = 0; i < 9; ++i) {
        H_out.h[i] *= inv_h8;
    }
    return true;
}

/**
 * @brief Tìm ma trận Homography tốt nhất bằng thuật toán RANSAC và tinh chỉnh lại trên tập Inliers.
 */
void estimate_homography_ransac(
    const std::vector<Keypoint>& kpA,
    const std::vector<Keypoint>& kpB,
    const std::vector<Match>& matches,
    const StitchParams& params,
    Homography& best_H,
    int& best_inliers
) {
    best_H = Homography{};
    best_inliers = 0;

    const int num_matches = static_cast<int>(matches.size());
    if (num_matches < 4) {
        return;
    }

    std::mt19937 rng(params.random_seed);
    std::uniform_int_distribution<int> uni(0, num_matches - 1);

    const double thresh2 = static_cast<double>(params.ransac_thresh) * params.ransac_thresh;
    double best_error_sum = std::numeric_limits<double>::max();

    for (int iter = 0; iter < params.ransac_iters; ++iter) {
        // Chọn ngẫu nhiên 4 chỉ số cặp điểm phân biệt
        int sample[4];
        for (int s = 0; s < 4; ++s) {
            bool unique = false;
            while (!unique) {
                sample[s] = uni(rng);
                unique = true;
                for (int prev = 0; prev < s; ++prev) {
                    if (sample[prev] == sample[s]) {
                        unique = false;
                        break;
                    }
                }
            }
        }

        Homography H_cand;
        if (!compute_normalized_dlt_homography(kpA, kpB, matches, sample, 4, H_cand)) {
            continue;
        }

        // Đếm số lượng điểm khớp thuận (Inliers) thỏa mãn sai số chiếu lại < ransac_thresh
        int inlier_count = 0;
        double error_sum = 0.0;

        for (int i = 0; i < num_matches; ++i) {
            const Keypoint& pA = kpA[matches[i].idxA];
            const Keypoint& pB = kpB[matches[i].idxB];

            const double w = H_cand.h[6] * pB.x + H_cand.h[7] * pB.y + H_cand.h[8];
            if (w <= 1e-5) {
                continue; // Loại bỏ các phép chiếu bị lật ra sau mặt phẳng camera
            }
            const double inv_w = 1.0 / w;
            const double proj_x = (H_cand.h[0] * pB.x + H_cand.h[1] * pB.y + H_cand.h[2]) * inv_w;
            const double proj_y = (H_cand.h[3] * pB.x + H_cand.h[4] * pB.y + H_cand.h[5]) * inv_w;

            const double dx = proj_x - pA.x;
            const double dy = proj_y - pA.y;
            const double d2 = dx * dx + dy * dy;

            if (d2 < thresh2) {
                ++inlier_count;
                error_sum += d2;
            }
        }

        // Cập nhật giả thuyết tốt nhất nếu có nhiều inliers hơn (hoặc bằng nhưng tổng sai số nhỏ hơn)
        if (inlier_count > best_inliers ||
            (inlier_count == best_inliers && inlier_count > 0 && error_sum < best_error_sum)) {
            best_inliers = inlier_count;
            best_error_sum = error_sum;
            best_H = H_cand;
        }
    }

    // Bước tinh chỉnh (Least-Squares Refinement): giải lại DLT trên toàn bộ tập Inliers của giả thuyết tốt nhất
    if (best_inliers >= 4) {
        std::vector<int> inlier_indices;
        inlier_indices.reserve(best_inliers);

        for (int i = 0; i < num_matches; ++i) {
            const Keypoint& pA = kpA[matches[i].idxA];
            const Keypoint& pB = kpB[matches[i].idxB];

            const double w = best_H.h[6] * pB.x + best_H.h[7] * pB.y + best_H.h[8];
            if (w <= 1e-5) continue;
            const double inv_w = 1.0 / w;
            const double proj_x = (best_H.h[0] * pB.x + best_H.h[1] * pB.y + best_H.h[2]) * inv_w;
            const double proj_y = (best_H.h[3] * pB.x + best_H.h[4] * pB.y + best_H.h[5]) * inv_w;

            const double dx = proj_x - pA.x;
            const double dy = proj_y - pA.y;
            if (dx * dx + dy * dy < thresh2) {
                inlier_indices.push_back(i);
            }
        }

        Homography H_refined;
        if (static_cast<int>(inlier_indices.size()) >= 4 &&
            compute_normalized_dlt_homography(
                kpA, kpB, matches, inlier_indices.data(),
                static_cast<int>(inlier_indices.size()), H_refined)) {
            int refined_inliers = 0;
            for (int i = 0; i < num_matches; ++i) {
                const Keypoint& pA = kpA[matches[i].idxA];
                const Keypoint& pB = kpB[matches[i].idxB];
                const double w = H_refined.h[6] * pB.x + H_refined.h[7] * pB.y + H_refined.h[8];
                if (w <= 1e-5) continue;
                const double inv_w = 1.0 / w;
                const double proj_x = (H_refined.h[0] * pB.x + H_refined.h[1] * pB.y + H_refined.h[2]) * inv_w;
                const double proj_y = (H_refined.h[3] * pB.x + H_refined.h[4] * pB.y + H_refined.h[5]) * inv_w;
                const double dx = proj_x - pA.x;
                const double dy = proj_y - pA.y;
                if (dx * dx + dy * dy < thresh2) {
                    ++refined_inliers;
                }
            }
            if (refined_inliers >= best_inliers - 2) {
                best_H = H_refined;
                best_inliers = refined_inliers;
            }
        }
    }
}

// ============================================================================
// STAGE 6: BIẾN ĐỔI HÌNH HỌC & HÒA TRỘN ẢNH (Inverse Warping + Bilinear + Feathering)
// ============================================================================

/**
 * @brief Tính ma trận nghịch đảo H^{-1} kích thước 3x3 bằng định thức và ma trận phụ hợp (adjugate).
 */
bool invert_homography_3x3(const Homography& H, Homography& H_inv) {
    const double* m = H.h;
    const double c00 =  (m[4] * m[8] - m[5] * m[7]);
    const double c01 = -(m[3] * m[8] - m[5] * m[6]);
    const double c02 =  (m[3] * m[7] - m[4] * m[6]);

    const double c10 = -(m[1] * m[8] - m[2] * m[7]);
    const double c11 =  (m[0] * m[8] - m[2] * m[6]);
    const double c12 = -(m[0] * m[7] - m[1] * m[6]);

    const double c20 =  (m[1] * m[5] - m[2] * m[4]);
    const double c21 = -(m[0] * m[5] - m[2] * m[3]);
    const double c22 =  (m[0] * m[4] - m[1] * m[3]);

    const double det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    if (std::fabs(det) < 1e-12) {
        return false;
    }

    const double inv_det = 1.0 / det;
    H_inv.h[0] = c00 * inv_det;
    H_inv.h[1] = c10 * inv_det;
    H_inv.h[2] = c20 * inv_det;
    H_inv.h[3] = c01 * inv_det;
    H_inv.h[4] = c11 * inv_det;
    H_inv.h[5] = c21 * inv_det;
    H_inv.h[6] = c02 * inv_det;
    H_inv.h[7] = c12 * inv_det;
    H_inv.h[8] = c22 * inv_det;
    return true;
}

/**
 * @brief Lấy mẫu màu RGB tại tọa độ thực (u, v) bằng nội suy song tuyến tính (Bilinear Interpolation).
 */
inline void sample_bilinear_rgb(const Image& img, float u, float v, float rgb_out[3]) {
    const int x0 = static_cast<int>(std::floor(u));
    const int y0 = static_cast<int>(std::floor(v));
    const int x1 = std::min(x0 + 1, img.width - 1);
    const int y1 = std::min(y0 + 1, img.height - 1);

    const float ax = u - static_cast<float>(x0);
    const float ay = v - static_cast<float>(y0);
    const float w00 = (1.0f - ax) * (1.0f - ay);
    const float w10 = ax * (1.0f - ay);
    const float w01 = (1.0f - ax) * ay;
    const float w11 = ax * ay;

    const uint8_t* p00 = img.data + (static_cast<size_t>(y0) * img.width + x0) * 3;
    const uint8_t* p10 = img.data + (static_cast<size_t>(y0) * img.width + x1) * 3;
    const uint8_t* p01 = img.data + (static_cast<size_t>(y1) * img.width + x0) * 3;
    const uint8_t* p11 = img.data + (static_cast<size_t>(y1) * img.width + x1) * 3;

    for (int c = 0; c < 3; ++c) {
        rgb_out[c] = w00 * p00[c] + w10 * p10[c] + w01 * p01[c] + w11 * p11[c];
    }
}

/**
 * @brief Biến đổi ảnh B về hệ tọa độ của ảnh A bằng Inverse Warping và hòa trộn vùng chồng lắp
 *        bằng trọng số khoảng cách tới biên (Linear Feathering).
 */
Image warp_and_blend(const Image& imgA, const Image& imgB, const Homography& H) {
    Homography H_inv;
    if (!invert_homography_3x3(H, H_inv)) {
        H_inv = Homography{};
    }

    // Chiếu 4 góc của ảnh B qua ma trận H sang hệ quy chiếu của ảnh A để tìm khung bao (Bounding Box)
    const double corners_B[4][2] = {
        {0.0, 0.0},
        {static_cast<double>(imgB.width - 1), 0.0},
        {static_cast<double>(imgB.width - 1), static_cast<double>(imgB.height - 1)},
        {0.0, static_cast<double>(imgB.height - 1)}
    };

    double min_x = 0.0;
    double max_x = static_cast<double>(imgA.width - 1);
    double min_y = 0.0;
    double max_y = static_cast<double>(imgA.height - 1);

    for (int i = 0; i < 4; ++i) {
        const double x = corners_B[i][0];
        const double y = corners_B[i][1];
        const double w = H.h[6] * x + H.h[7] * y + H.h[8];
        if (w > 1e-4) {
            const double px = (H.h[0] * x + H.h[1] * y + H.h[2]) / w;
            const double py = (H.h[3] * x + H.h[4] * y + H.h[5]) / w;
            min_x = std::min(min_x, px);
            max_x = std::max(max_x, px);
            min_y = std::min(min_y, py);
            max_y = std::max(max_y, py);
        }
    }

    // Giới hạn kích thước khung ảnh tối đa để tránh tràn bộ nhớ khi gặp cặp ảnh lỗi suy biến
    const double max_span_w = static_cast<double>(imgA.width) * 2.5;
    const double max_span_h = static_cast<double>(imgA.height) * 2.0;
    min_x = std::max(min_x, -max_span_w);
    max_x = std::min(max_x,  max_span_w);
    min_y = std::max(min_y, -max_span_h);
    max_y = std::min(max_y,  max_span_h);

    const int offset_x = static_cast<int>(std::floor(min_x));
    const int offset_y = static_cast<int>(std::floor(min_y));
    const int out_w = std::max(1, static_cast<int>(std::ceil(max_x)) - offset_x + 1);
    const int out_h = std::max(1, static_cast<int>(std::ceil(max_y)) - offset_y + 1);

    Image pano = Image::allocate(out_w, out_h, 3);

    const float wA_max = static_cast<float>(imgA.width - 1);
    const float hA_max = static_cast<float>(imgA.height - 1);
    const float wB_max = static_cast<float>(imgB.width - 1);
    const float hB_max = static_cast<float>(imgB.height - 1);

    for (int y_out = 0; y_out < out_h; ++y_out) {
        const int yA = y_out + offset_y;
        const bool in_A_y = (yA >= 0 && yA < imgA.height);

        for (int x_out = 0; x_out < out_w; ++x_out) {
            const int xA = x_out + offset_x;
            const bool in_A = in_A_y && (xA >= 0 && xA < imgA.width);

            // Ánh xạ ngược (Inverse Warp) từ tọa độ (xA, yA) về tọa độ thực (uB, vB) trên ảnh B
            const double denom = H_inv.h[6] * xA + H_inv.h[7] * yA + H_inv.h[8];
            bool in_B = false;
            float uB = 0.0f;
            float vB = 0.0f;

            if (denom > 1e-5) {
                const double inv_d = 1.0 / denom;
                uB = static_cast<float>((H_inv.h[0] * xA + H_inv.h[1] * yA + H_inv.h[2]) * inv_d);
                vB = static_cast<float>((H_inv.h[3] * xA + H_inv.h[4] * yA + H_inv.h[5]) * inv_d);
                if (uB >= 0.0f && uB <= wB_max && vB >= 0.0f && vB <= hB_max) {
                    in_B = true;
                }
            }

            uint8_t* dst_pixel = pano.data + (static_cast<size_t>(y_out) * out_w + x_out) * 3;

            if (in_A && in_B) {
                // Vùng chồng lắp: tính trọng số feathering dựa trên khoảng cách tới mép gần nhất của mỗi ảnh
                const float dist_A = std::min({
                    static_cast<float>(xA),
                    wA_max - static_cast<float>(xA),
                    static_cast<float>(yA),
                    hA_max - static_cast<float>(yA)
                });
                const float dist_B = std::min({
                    uB,
                    wB_max - uB,
                    vB,
                    hB_max - vB
                });

                const float alpha = (dist_A + dist_B > 1e-5f)
                    ? (dist_A / (dist_A + dist_B))
                    : 0.5f;

                float rgbB[3];
                sample_bilinear_rgb(imgB, uB, vB, rgbB);
                const uint8_t* rgbA = imgA.data + (static_cast<size_t>(yA) * imgA.width + xA) * 3;

                for (int c = 0; c < 3; ++c) {
                    const float blended = alpha * static_cast<float>(rgbA[c]) + (1.0f - alpha) * rgbB[c];
                    dst_pixel[c] = static_cast<uint8_t>(clamp_idx(static_cast<int>(std::round(blended)), 0, 255));
                }
            } else if (in_A) {
                const uint8_t* rgbA = imgA.data + (static_cast<size_t>(yA) * imgA.width + xA) * 3;
                dst_pixel[0] = rgbA[0];
                dst_pixel[1] = rgbA[1];
                dst_pixel[2] = rgbA[2];
            } else if (in_B) {
                float rgbB[3];
                sample_bilinear_rgb(imgB, uB, vB, rgbB);
                for (int c = 0; c < 3; ++c) {
                    dst_pixel[c] = static_cast<uint8_t>(clamp_idx(static_cast<int>(std::round(rgbB[c])), 0, 255));
                }
            }
        }
    }

    return pano;
}

} // anonymous namespace

// ============================================================================
// HÀM GIAO DIỆN CHÍNH CỦA PHIÊN BẢN V0
// ============================================================================

Image stitch_v0(
    const Image& imgA,
    const Image& imgB,
    const StitchParams& params,
    StageTimings& timings,
    StitchStats& stats
) {
    const auto t_pipeline_start = Clock::now();

    // ------------------------------------------------------------------------
    // Stage 1: Tiền xử lý (Chuyển ảnh xám + Làm mờ Gaussian 2D)
    // ------------------------------------------------------------------------
    const auto t1_start = Clock::now();
    const std::vector<float> gauss_kernel = make_gaussian_kernel_2d(params.gaussian_radius, params.gaussian_sigma);
    std::vector<float> grayA, grayB, blurA, blurB;
    rgb_to_grayscale(imgA, grayA);
    rgb_to_grayscale(imgB, grayB);
    gaussian_blur_2d(grayA, blurA, imgA.width, imgA.height, params.gaussian_radius, gauss_kernel);
    gaussian_blur_2d(grayB, blurB, imgB.width, imgB.height, params.gaussian_radius, gauss_kernel);
    const auto t1_end = Clock::now();
    timings.preprocess_ms = elapsed_ms(t1_start, t1_end);

    // ------------------------------------------------------------------------
    // Stage 2: Phát hiện điểm đặc trưng (Harris Corner + NMS + Top-K)
    // ------------------------------------------------------------------------
    const auto t2_start = Clock::now();
    std::vector<Keypoint> kpA, kpB;
    detect_harris_keypoints(blurA, imgA.width, imgA.height, params, kpA);
    detect_harris_keypoints(blurB, imgB.width, imgB.height, params, kpB);
    const auto t2_end = Clock::now();
    timings.detect_ms = elapsed_ms(t2_start, t2_end);

    // ------------------------------------------------------------------------
    // Stage 3: Mô tả đặc trưng (Rotated BRIEF 256-bit)
    // ------------------------------------------------------------------------
    const auto t3_start = Clock::now();
    const std::vector<BriefPair> brief_pattern = generate_brief_pattern(params.random_seed);
    std::vector<Descriptor> descA, descB;
    compute_rotated_brief_descriptors(blurA, imgA.width, imgA.height, brief_pattern, params.patch_radius, kpA, descA);
    compute_rotated_brief_descriptors(blurB, imgB.width, imgB.height, brief_pattern, params.patch_radius, kpB, descB);
    const auto t3_end = Clock::now();
    timings.describe_ms = elapsed_ms(t3_start, t3_end);

    // ------------------------------------------------------------------------
    // Stage 4: So khớp đặc trưng (Brute-Force Hamming + Lowe's Ratio Test + Cross-Check)
    // ------------------------------------------------------------------------
    const auto t4_start = Clock::now();
    std::vector<Match> matches;
    match_descriptors(descA, descB, params, matches);
    const auto t4_end = Clock::now();
    timings.match_ms = elapsed_ms(t4_start, t4_end);

    // ------------------------------------------------------------------------
    // Stage 5: Ước lượng Homography (RANSAC + Normalized 4-Point DLT + Refinement)
    // ------------------------------------------------------------------------
    const auto t5_start = Clock::now();
    Homography best_H;
    int best_inliers = 0;
    estimate_homography_ransac(kpA, kpB, matches, params, best_H, best_inliers);
    const auto t5_end = Clock::now();
    timings.homography_ms = elapsed_ms(t5_start, t5_end);

    // ------------------------------------------------------------------------
    // Stage 6: Biến đổi & Hòa trộn ảnh (Inverse Warping + Bilinear + Feathering)
    // ------------------------------------------------------------------------
    const auto t6_start = Clock::now();
    Image pano = warp_and_blend(imgA, imgB, best_H);
    const auto t6_end = Clock::now();
    timings.warp_blend_ms = elapsed_ms(t6_start, t6_end);

    timings.h2d_ms = 0.0;
    timings.d2h_ms = 0.0;
    timings.total_ms = elapsed_ms(t_pipeline_start, t6_end);

    stats.num_kp_A = static_cast<int>(kpA.size());
    stats.num_kp_B = static_cast<int>(kpB.size());
    stats.num_matches = static_cast<int>(matches.size());
    stats.num_inliers = best_inliers;
    stats.H = best_H;

    return pano;
}
