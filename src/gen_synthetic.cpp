/**
 * @file gen_synthetic.cpp
 * @brief Sinh bộ ảnh kiểm thử giả lập (1280x720, 1920x1080, 3840x2160)
 *        kèm ma trận Homography chuẩn (Ground-Truth H_gt) để kiểm tra độ chính xác và đo thời gian V0.
 */

#include "common.h"
#include "stb_image_write.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

struct RectShape {
    float cx, cy, w, h, angle;
    uint8_t r, g, b;
};

// Vẽ bức cảnh tổng thể chứa các hình khối kiến trúc/cửa sổ có góc cạnh sắc nét để kiểm thử
void render_canvas(Image& canvas) {
    const int W = canvas.width;
    const int H = canvas.height;

    // Nền chuyển màu (gradient) bầu trời và mặt đất
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t* p = canvas.data + (static_cast<size_t>(y) * W + x) * 3;
            p[0] = static_cast<uint8_t>(40 + (120 * y) / H);
            p[1] = static_cast<uint8_t>(70 + (100 * x) / W);
            p[2] = static_cast<uint8_t>(140 - (60 * y) / H);
        }
    }

    // Sinh cố định 280 khối hình chữ nhật xoay góc có họa tiết lưới ô cờ bên trong
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> ux(0.05f * W, 0.95f * W);
    std::uniform_real_distribution<float> uy(0.08f * H, 0.92f * H);
    std::uniform_real_distribution<float> uw(0.015f * W, 0.06f * W);
    std::uniform_real_distribution<float> uang(-0.6f, 0.6f);
    std::uniform_int_distribution<int> ucol(30, 245);

    const int num_shapes = 280;
    std::vector<RectShape> shapes(num_shapes);
    for (int i = 0; i < num_shapes; ++i) {
        shapes[i].cx = ux(rng);
        shapes[i].cy = uy(rng);
        shapes[i].w = uw(rng);
        shapes[i].h = uw(rng) * (0.5f + (i % 3) * 0.4f);
        shapes[i].angle = uang(rng);
        shapes[i].r = static_cast<uint8_t>(ucol(rng));
        shapes[i].g = static_cast<uint8_t>(ucol(rng));
        shapes[i].b = static_cast<uint8_t>(ucol(rng));
    }

    for (const auto& s : shapes) {
        const float cos_a = std::cos(s.angle);
        const float sin_a = std::sin(s.angle);
        const float rad = std::hypot(s.w, s.h);
        const int x0 = std::max(0, static_cast<int>(s.cx - rad));
        const int x1 = std::min(W - 1, static_cast<int>(s.cx + rad));
        const int y0 = std::max(0, static_cast<int>(s.cy - rad));
        const int y1 = std::min(H - 1, static_cast<int>(s.cy + rad));

        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const float dx = x - s.cx;
                const float dy = y - s.cy;
                const float lx =  dx * cos_a + dy * sin_a;
                const float ly = -dx * sin_a + dy * cos_a;
                if (std::fabs(lx) <= s.w * 0.5f && std::fabs(ly) <= s.h * 0.5f) {
                    const int gx = static_cast<int>((lx + s.w * 0.5f) / std::max(4.0f, s.w * 0.25f));
                    const int gy = static_cast<int>((ly + s.h * 0.5f) / std::max(4.0f, s.h * 0.25f));
                    const bool parity = ((gx + gy) & 1) != 0;
                    uint8_t* p = canvas.data + (static_cast<size_t>(y) * W + x) * 3;
                    p[0] = parity ? s.r : static_cast<uint8_t>(s.r * 0.55f);
                    p[1] = parity ? s.g : static_cast<uint8_t>(s.g * 0.55f);
                    p[2] = parity ? s.b : static_cast<uint8_t>(s.b * 0.55f);
                }
            }
        }
    }
}

// Nội suy song tuyến tính (Bilinear) để sinh ảnh B từ khung cảnh gốc
void sample_bilinear(const Image& img, double u, double v, uint8_t out[3]) {
    const int x0 = std::max(0, std::min(img.width - 1, static_cast<int>(std::floor(u))));
    const int y0 = std::max(0, std::min(img.height - 1, static_cast<int>(std::floor(v))));
    const int x1 = std::min(img.width - 1, x0 + 1);
    const int y1 = std::min(img.height - 1, y0 + 1);
    const double ax = u - x0;
    const double ay = v - y0;

    const uint8_t* p00 = img.data + (static_cast<size_t>(y0) * img.width + x0) * 3;
    const uint8_t* p10 = img.data + (static_cast<size_t>(y0) * img.width + x1) * 3;
    const uint8_t* p01 = img.data + (static_cast<size_t>(y1) * img.width + x0) * 3;
    const uint8_t* p11 = img.data + (static_cast<size_t>(y1) * img.width + x1) * 3;

    for (int c = 0; c < 3; ++c) {
        const double val = (1.0 - ax) * (1.0 - ay) * p00[c] +
                           ax * (1.0 - ay) * p10[c] +
                           (1.0 - ax) * ay * p01[c] +
                           ax * ay * p11[c];
        out[c] = static_cast<uint8_t>(std::max(0.0, std::min(255.0, std::round(val))));
    }
}

// Sinh cặp ảnh Trái (A) và Phải (B) ở độ phân giải W x H kèm file Ground-Truth Homography
void generate_pair(const std::string& prefix, int W, int H) {
    const int CW = static_cast<int>(W * 1.8);
    const int CH = static_cast<int>(H * 1.3);
    Image canvas = Image::allocate(CW, CH, 3);
    render_canvas(canvas);

    const int ox = static_cast<int>(W * 0.1);
    const int oy = static_cast<int>(H * 0.15);

    // Cắt ảnh A trực tiếp từ khung cảnh tổng tại vị trí (ox, oy)
    Image imgA = Image::allocate(W, H, 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const uint8_t* src = canvas.data + (static_cast<size_t>(y + oy) * CW + (x + ox)) * 3;
            uint8_t* dst = imgA.data + (static_cast<size_t>(y) * W + x) * 3;
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
        }
    }

    // Ma trận Ground-Truth Homography H_gt ánh xạ p_B -> p_A:
    // Dịch ngang 35% chiều rộng, xoay ~2 độ và biến dạng phối cảnh nhẹ
    const double theta = 0.035;
    const double cos_t = std::cos(theta);
    const double sin_t = std::sin(theta);
    const double tx = 0.35 * W;
    const double ty = 0.02 * H;
    const double p1 = -1.5e-5 / (W / 1280.0);
    const double p2 =  1.0e-5 / (H / 720.0);

    double H_gt[9] = {
        cos_t, -sin_t, tx,
        sin_t,  cos_t, ty,
        p1,     p2,    1.0
    };

    Image imgB = Image::allocate(W, H, 3);
    for (int yB = 0; yB < H; ++yB) {
        for (int xB = 0; xB < W; ++xB) {
            const double w = H_gt[6] * xB + H_gt[7] * yB + H_gt[8];
            const double xA = (H_gt[0] * xB + H_gt[1] * yB + H_gt[2]) / w;
            const double yA = (H_gt[3] * xB + H_gt[4] * yB + H_gt[5]) / w;

            uint8_t* dst = imgB.data + (static_cast<size_t>(yB) * W + xB) * 3;
            sample_bilinear(canvas, xA + ox, yA + oy, dst);
        }
    }

    const std::string pathA = prefix + "_left.png";
    const std::string pathB = prefix + "_right.png";
    const std::string pathGT = prefix + "_gt.txt";

    stbi_write_png(pathA.c_str(), W, H, 3, imgA.data, W * 3);
    stbi_write_png(pathB.c_str(), W, H, 3, imgB.data, W * 3);

    std::ofstream ofs(pathGT);
    for (int r = 0; r < 3; ++r) {
        ofs << H_gt[r * 3 + 0] << " " << H_gt[r * 3 + 1] << " " << H_gt[r * 3 + 2] << "\n";
    }
    ofs.close();

    std::printf("Generated %s (%dx%d): %s, %s, %s\n",
                prefix.c_str(), W, H, pathA.c_str(), pathB.c_str(), pathGT.c_str());

    imgB.free_data();
    imgA.free_data();
    canvas.free_data();
}

} // anonymous namespace

int main() {
    generate_pair("dataset/synthetic_small",  1280, 720);
    generate_pair("dataset/synthetic_medium", 1920, 1080);
    generate_pair("dataset/synthetic_large",  3840, 2160);
    return 0;
}
