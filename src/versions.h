/**
 * @file versions.h
 * @brief Khai báo hàm giao diện cho các phiên bản pipeline ghép ảnh Panorama (V0, V1, V2...).
 *
 * Tác giả: Group 18
 */

#ifndef VERSIONS_H
#define VERSIONS_H

#include "common.h"

/**
 * @brief Phiên bản V0: Cài đặt tuần tự chuẩn (baseline) trên 1 luồng CPU.
 *
 * @param imgA      Ảnh RGB bên trái (ảnh gốc tham chiếu).
 * @param imgB      Ảnh RGB bên phải (ảnh cần biến đổi về hệ quy chiếu của imgA).
 * @param params    Bộ siêu tham số cấu hình pipeline.
 * @param timings   Đầu ra thời gian thực thi của từng stage (mili-giây).
 * @param stats     Đầu ra thông số thống kê (số keypoints, matches, inliers, ma trận Homography).
 * @return          Ảnh RGB Panorama đã ghép (người gọi hàm chịu trách nhiệm giải phóng bằng free_data()).
 */
Image stitch_v0(
    const Image& imgA,
    const Image& imgB,
    const StitchParams& params,
    StageTimings& timings,
    StitchStats& stats
);

#endif // VERSIONS_H
