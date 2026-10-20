# Luồng Xử Lý Ghép Ảnh Panorama (`V0` Sequential Pipeline)

Tài liệu này mô tả chi tiết luồng xử lý của chương trình ghép ảnh Panorama (`stitch_v0` trong [v0_sequential.cpp](file:///home/iamhieu2012/Dev/image-stitching/src/v0_sequential.cpp)) bằng sơ đồ **Mermaid**.

---

## 1. Sơ Đồ Tổng Quan Pipeline (6 Stages)

Các bước **Stage 1 $\rightarrow$ Stage 3** xử lý hoàn toàn độc lập trên từng ảnh (`Image A` và `Image B`). Từ **Stage 4 $\rightarrow$ Stage 6**, dữ liệu của cả hai ảnh được kết hợp lại để tìm phép biến đổi hình học và tạo ra ảnh Panorama cuối cùng.

```mermaid
flowchart TD
    subgraph Input["Đầu vào (main.cu)"]
        ImgA["Ảnh Trái (Image A - RGB)"]
        ImgB["Ảnh Phải (Image B - RGB)"]
    end

    subgraph Stage1["Stage 1: Preprocessing (Độc lập từng ảnh)"]
        GrayA["RGB sang Grayscale (float)"]
        BlurA["Gaussian Blur 2D (5x5, sigma = 1.4)"]
        GrayB["RGB sang Grayscale (float)"]
        BlurB["Gaussian Blur 2D (5x5, sigma = 1.4)"]
    end

    subgraph Stage2["Stage 2: Keypoint Detection (Harris Corner)"]
        SobelA["Đạo hàm Sobel 3x3 (Ix, Iy)"]
        TensorA["Tổng cửa sổ 3x3 (Sxx, Syy, Sxy) và Harris R"]
        NMSA["NMS 5x5 + Lọc biên 16px + Top-K (3000 điểm)"]

        SobelB["Đạo hàm Sobel 3x3 (Ix, Iy)"]
        TensorB["Tổng cửa sổ 3x3 (Sxx, Syy, Sxy) và Harris R"]
        NMSB["NMS 5x5 + Lọc biên 16px + Top-K (3000 điểm)"]
    end

    subgraph Stage3["Stage 3: Description (Rotated BRIEF 256-bit)"]
        AngleA["Tính góc hướng theta (Intensity Centroid r=15)"]
        DescA["Xoay 256 cặp mẫu và đóng gói 8 x uint32_t"]

        AngleB["Tính góc hướng theta (Intensity Centroid r=15)"]
        DescB["Xoay 256 cặp mẫu và đóng gói 8 x uint32_t"]
    end

    subgraph Stage4["Stage 4: Feature Matching (Kết hợp 2 ảnh)"]
        Hamming["Tính khoảng cách Hamming Brute-Force (XOR + popcount)"]
        Ratio["Lọc Lowe Ratio Test (d1 < 0.75 * d2, d1 <= 64)"]
        CrossCheck["Kiểm tra chéo 2 chiều (Mutual Cross-Check A <-> B)"]
    end

    subgraph Stage5["Stage 5: Homography Estimation (RANSAC + DLT)"]
        Sample["Lặp 2000 lần: Lấy ngẫu nhiên 4 cặp Match"]
        NormDLT["Hartley Normalization + Giải hệ 8x8 (Gaussian Elimination)"]
        CountInliers["Chiếu p_B sang A và đếm Inliers (sai số < 3.0 px)"]
        Refine["Tinh chỉnh bình phương tối thiểu (Least-Squares) trên toàn bộ Inliers tốt nhất"]
    end

    subgraph Stage6["Stage 6: Warping and Blending"]
        BBox["Chiếu 4 góc ảnh B qua H để tính khung ảnh Panorama (Bounding Box)"]
        InvWarp["Tính H^-1 và ánh xạ ngược từng pixel đầu ra (x_out, y_out)"]
        Bilinear["Lấy mẫu nội suy Bilinear trên ảnh B"]
        Feather["Hòa trộn vùng chồng lắp bằng trọng số khoảng cách (Linear Feathering)"]
    end

    Output["Ảnh Panorama Đầu Ra (output_image)"]

    ImgA --> GrayA --> BlurA --> SobelA --> TensorA --> NMSA --> AngleA --> DescA
    ImgB --> GrayB --> BlurB --> SobelB --> TensorB --> NMSB --> AngleB --> DescB

    DescA --> Hamming
    DescB --> Hamming
    Hamming --> Ratio --> CrossCheck

    NMSA --> Sample
    NMSB --> Sample
    CrossCheck --> Sample
    Sample --> NormDLT --> CountInliers --> Refine

    ImgA --> Feather
    ImgB --> Bilinear
    Refine --> BBox --> InvWarp --> Bilinear --> Feather --> Output
```

---

## 2. Sơ Đồ Chi Tiết Luồng Thực Thi & Đo Thời Gian (`main.cu` $\rightarrow$ `stitch_v0`)

Sơ đồ trình tự (Sequence Diagram) dưới đây thể hiện cách chương trình chính [`main.cu`](file:///home/iamhieu2012/Dev/image-stitching/src/main.cu) đọc ảnh, chạy **Warm-up**, lặp **$N$ lần đo thời gian**, kiểm tra sai số với Ground-Truth và lưu ảnh kết quả:

```mermaid
sequenceDiagram
    autonumber
    participant CLI as User / CLI
    participant Main as main.cu
    participant STB as stb_image / write
    participant V0 as stitch_v0 (v0_sequential.cpp)

    CLI->>Main: ./stitch left right out --version 0 --runs 5 [--gt gt.txt]
    Main->>STB: stbi_load(left_image), stbi_load(right_image)
    STB-->>Main: Image A (RGB), Image B (RGB)

    Note over Main,V0: Bước 1: Chạy 1 lần Warm-up (Không tính vào thời gian đo)
    Main->>V0: stitch_v0(imgA, imgB, params, warmup_timings, stats)
    V0-->>Main: warmup_out (giải phóng bộ nhớ ngay)

    Note over Main,V0: Bước 2: Lặp N lần đo thời gian (Mặc định N = 5)
    loop r = 1 .. N runs
        Main->>V0: stitch_v0(imgA, imgB, params, cur_timings, cur_stats)
        Note right of V0: Đo riêng từng Stage 1..6 bằng<br/>std::chrono::high_resolution_clock
        V0-->>Main: trả về ảnh Panorama + cur_timings + cur_stats
        Main->>Main: Cộng dồn Avg time & cập nhật Min time cho từng Stage
    end

    opt Nếu có truyền cờ --gt <gt_homography.txt>
        Main->>Main: Đọc ma trận H_gt và tính sai số Frobenius + Sai số chiếu 5 điểm (px)
    end

    Main->>Main: In bảng Timing 6 Stages + Tỷ lệ % + Phân tích định luật Amdahl (p)
    Main->>STB: stbi_write_png / stbi_write_jpg (out_path)
    STB-->>CLI: Lưu thành công file ảnh Panorama
```

---

## 3. Sơ Đồ Chi Tiết Thuật Toán RANSAC + Normalized DLT (Stage 5)

Stage 5 là bước quyết định độ chính xác hình học của ảnh ghép. Luồng xử lý chi tiết bên trong hàm `estimate_homography_ransac` như sau:

```mermaid
flowchart LR
    Matches["Tập Matches từ Stage 4 (M cặp điểm)"] --> CheckM{"M >= 4?"}
    CheckM -- "Không" --> Identity["Trả về ma trận đơn vị I"]
    CheckM -- "Có" --> LoopStart["Vòng lặp RANSAC (iter = 1..2000)"]

    subgraph RansacLoop["Mỗi vòng lặp RANSAC"]
        LoopStart --> Pick4["Chọn ngẫu nhiên 4 cặp điểm phân biệt"]
        Pick4 --> Hartley["Chuẩn hóa Hartley: Dịch tâm về (0,0), co giãn khoảng cách TB = sqrt(2)"]
        Hartley --> Solve8x8["Lập hệ phương trình chuẩn 8x8 và giải Khử Gauss (Partial Pivoting)"]
        Solve8x8 --> Denorm["Khử chuẩn hóa: H = T_A^-1 * H_norm * T_B"]
        Denorm --> EvalInliers["Chiếu toàn bộ M điểm từ B sang A, đếm số điểm có sai số Euclid < 3.0 px"]
        EvalInliers --> UpdateBest["Cập nhật best_H nếu số Inliers lớn hơn (hoặc bằng nhưng tổng sai số nhỏ hơn)"]
    end

    UpdateBest --> RefineCheck{"best_inliers >= 4?"}
    RefineCheck -- "Có" --> LeastSquares["Gom tất cả Inliers tốt nhất -> Giải lại DLT Bình phương tối thiểu (A^T A h = A^T b)"]
    LeastSquares --> FinalH["Ma trận Homography tối ưu H (3x3)"]
    RefineCheck -- "Không" --> FinalH
```

---

## 4. Tóm Tắt Input / Output & Độ Phức Tạp Từng Stage

| Stage | Hàm trong `v0_sequential.cpp` | Input | Output | Độ phức tạp tuần tự |
| :--- | :--- | :--- | :--- | :--- |
| **1. Preprocessing** | `rgb_to_grayscale`<br>`gaussian_blur_2d` | `Image A, B` ($W \times H \times 3$ `uint8_t`) | `blurA, blurB` ($W \times H$ `float`) | $O(W \cdot H \cdot k^2)$ với $k=5$ |
| **2. Keypoint Detection** | `detect_harris_keypoints` | `blurA, blurB` ($W \times H$ `float`) | `kpA, kpB` ($\le 3000$ `Keypoint`) | $O(W \cdot H + C \log K)$ |
| **3. Description** | `compute_rotated_brief_descriptors` | `blurA, blurB` + `kpA, kpB` | `descA, descB` ($\le 3000 \times 256\text{-bit}$) | $O(K \cdot (r^2 + 256))$ với $r=15$ |
| **4. Matching** | `match_descriptors` | `descA` ($N_A$), `descB` ($N_B$) | `matches` ($M$ cặp điểm khớp) | $O(N_A \cdot N_B)$ |
| **5. Homography** | `estimate_homography_ransac`<br>`compute_normalized_dlt_homography` | `kpA, kpB, matches` | Ma trận `Homography H` ($3\times 3$) | $O(I_{\text{ransac}} \cdot (8^3 + M))$ |
| **6. Warping & Blending** | `warp_and_blend` | `imgA, imgB, H` | Ảnh Panorama `pano` ($W_{out} \times H_{out} \times 3$) | $O(W_{out} \cdot H_{out})$ |
