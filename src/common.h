/**
 * @file common.h
 * @brief Định nghĩa các cấu trúc dữ liệu, tham số cấu hình và bộ đo thời gian dùng chung cho bài toán ghép ảnh Panorama.
 *
 * Tác giả: Group 18
 */

#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/**
 * @brief Biểu diễn ảnh màu RGB lưu trữ theo thứ tự hàng (row-major interleaved: R, G, B).
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
 * @brief Biểu diễn một điểm đặc trưng 2D (Keypoint) trên ảnh.
 */
struct Keypoint {
    float x = 0.0f;
    float y = 0.0f;
    float response = 0.0f;
    float angle = 0.0f; // Góc hướng tính bằng radian (dùng cho Rotated BRIEF)
};

/**
 * @brief Vector mô tả nhị phân 256-bit (gồm 8 số nguyên không dấu 32-bit).
 */
struct Descriptor {
    uint32_t bits[8];
};

/**
 * @brief Biểu diễn một cặp keypoint khớp nhau giữa Ảnh A và Ảnh B.
 */
struct Match {
    int idxA = -1;
    int idxB = -1;
    int distance = 256;
};

/**
 * @brief Ma trận biến đổi phối cảnh Homography 3x3 ánh xạ điểm từ Ảnh B sang Ảnh A:
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
 * @brief Bộ siêu tham số của pipeline (giữ cố định giống nhau ở mọi phiên bản V0, V1, V2...).
 */
struct StitchParams {
    int gaussian_radius = 2;        // Bán kính bộ lọc Gaussian 5x5 (radius = 2)
    float gaussian_sigma = 1.4f;    // Độ lệch chuẩn sigma cho làm mờ Gaussian
    float harris_k = 0.04f;         // Hệ số nhạy của bộ dò góc Harris
    float harris_thresh = 1e4f;     // Ngưỡng phản hồi góc Harris tối thiểu
    int nms_radius = 2;             // Bán kính cửa sổ NMS 5x5 (radius = 2)
    int max_keypoints = 3000;       // Số lượng keypoints tối đa giữ lại (Top-K) trên mỗi ảnh
    int patch_radius = 15;          // Bán kính vùng lân cận 31x31 cho Rotated BRIEF
    float ratio_thresh = 0.75f;     // Ngưỡng kiểm định tỉ số Lowe (Lowe's ratio test)
    int max_hamming_dist = 64;      // Khoảng cách Hamming tối đa chấp nhận được
    int ransac_iters = 2000;        // Số vòng lặp RANSAC
    float ransac_thresh = 3.0f;     // Ngưỡng sai số chiếu lại (pixel) để tính là Inlier trong RANSAC
    uint32_t random_seed = 42;      // Seed cố định để tái lập kết quả đồng nhất giữa các phiên bản
};

/**
 * @brief Thời gian chạy (đơn vị: mili-giây) của từng bước trong 6 bước của pipeline.
 */
struct StageTimings {
    double preprocess_ms = 0.0;     // Stage 1: Chuyển RGB sang Grayscale + Làm mờ Gaussian
    double detect_ms = 0.0;         // Stage 2: Dò góc Harris + NMS + Chọn Top-K
    double describe_ms = 0.0;       // Stage 3: Mô tả đặc trưng Rotated BRIEF 256-bit
    double match_ms = 0.0;          // Stage 4: So khớp Brute-Force Hamming + Ratio Test + Cross-Check
    double homography_ms = 0.0;     // Stage 5: RANSAC + 4-point Normalized DLT + Refinement
    double warp_blend_ms = 0.0;     // Stage 6: Inverse Warping + Bilinear + Linear Feathering
    double h2d_ms = 0.0;            // Thời gian copy dữ liệu Host -> Device (bằng 0 ở V0)
    double d2h_ms = 0.0;            // Thời gian copy dữ liệu Device -> Host (bằng 0 ở V0)
    double total_ms = 0.0;          // Tổng thời gian thực thi toàn bộ pipeline (End-to-End)
};

/**
 * @brief Các thông số thống kê kết quả sau khi chạy pipeline.
 */
struct StitchStats {
    int num_kp_A = 0;
    int num_kp_B = 0;
    int num_matches = 0;
    int num_inliers = 0;
    Homography H;
};

#endif // COMMON_H
