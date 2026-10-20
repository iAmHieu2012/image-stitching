# Giải Thích Lựa Chọn Thuật Toán & Thiết Kế Pipeline (`EXPLAIN.md`)

Tài liệu này giải thích chi tiết lý do lựa chọn từng thuật toán trong 6 bước của pipeline ([`src/v0_sequential.cpp`](file:///home/iamhieu2012/Dev/image-stitching/src/v0_sequential.cpp)), lý do loại bỏ các phương án thay thế trong đề bài, và các luận điểm bảo vệ khi **vấn đáp (Oral Defense)**.

---

## 1. Nguyên Tắc Của Đề Bài Về Việc Chọn Thuật Toán

Theo mục **2.2 Suggested Methods** và mục **3 Implementation Guideline** trong [`PROJECT_SPEC.md`](file:///home/iamhieu2012/Dev/image-stitching/PROJECT_SPEC.md):
1. **Bảng phương pháp trong đề là "thực đơn gợi ý" (menu), không phải yêu cầu cài đặt tất cả:**
   > *"Teams may choose any method for each stage, if the choice is reasonable, produces a good panorama, and can be implemented efficiently. The table lists common options as a starting point, not a requirement."*
2. **Giữ cố định 1 bộ thuật toán và tham số xuyên suốt mọi phiên bản (`V0`, `V1`, `V2`, `V3`, ...):**
   > *"Every team submits at least 4 versions of the same pipeline, using the same methods and parameters in every version so that timing comparisons are fair."*

Do đó, ở mỗi bước (stage), nhóm chỉ chọn **1 phương pháp tối ưu nhất** sao cho cân bằng giữa hai yếu tố:
- **Chất lượng ảnh ghép (Output Quality):** Độ chính xác ma trận Homography cao, ảnh ghép sắc nét, không bị lệch hình hay lộ vết cắt (seam).
- **Hiệu năng song song trên GPU (Parallel Efficiency):** Phép tính đồng nhất trên từng pixel/keypoint/match, ít rẽ nhánh (`warp divergence`), tiết kiệm băng thông bộ nhớ và tận dụng tốt phần cứng CUDA.

---

## 2. Bảng Đối Chiếu: Đề Bài Gợi Ý vs. Phương Pháp Nhóm Chọn vs. Lý Do Loại Bỏ

| Stage | Các lựa chọn trong đề bài | Phương pháp ĐÃ DÙNG trong `V0` | Phương pháp KHÔNG DÙNG & Lý do tại sao không chọn |
| :--- | :--- | :--- | :--- |
| **1. Preprocessing** | - Grayscale<br>- Gaussian blur (2D / separable)<br>- Image pyramid | **Grayscale** + **Gaussian Blur 2D ($5\times 5, \sigma=1.4$)** *(sang V2 nâng cấp thành Separable 1D)* | - **Image pyramid (Tháp ảnh đa tỉ lệ):** Chỉ cần thiết khi 2 ảnh chụp có độ phóng đại (zoom/scale) chênh lệch lớn. Khi chụp panorama đứng yên xoay máy, tỉ lệ giữa 2 ảnh gần như không đổi nên không cần tốn bộ nhớ và thời gian dựng pyramid. |
| **2. Keypoint Detection** | *Bộ phát hiện góc:*<br>- Harris<br>- Shi–Tomasi<br>- FAST<br>- DoG (SIFT-style)<br>*Hậu xử lý:*<br>- Non-maximum suppression (NMS)<br>- Top-K selection | **Harris Corner Detector** + **NMS ($5\times 5$)** + **Top-K ($K=3000$)** | - **Shi–Tomasi:** Phải tính căn bậc hai để tìm trị riêng nhỏ nhất $\lambda_{\min}$ tại mọi pixel, nặng hơn phép nhân/trừ của Harris mà chất lượng góc tương đương.<br>- **FAST:** Dùng chuỗi câu lệnh rẽ nhánh `if-else` trên vòng tròn 16 pixel $\rightarrow$ khi đưa lên GPU sẽ gây **warp divergence** (phân kỳ luồng) nghiêm trọng.<br>- **DoG (SIFT-style):** Phải xây dựng tháp Gaussian nhiều tầng (octaves) và tìm cực trị không gian 3 chiều $3\times 3\times 3$, rất phức tạp khi tự cài đặt bằng CUDA thuần và khó đồng bộ kết quả giữa CPU/GPU. |
| **3. Description** | - BRIEF<br>- ORB (rotated BRIEF)<br>- SIFT-like histograms | **ORB (Rotated BRIEF 256-bit)** kèm hướng **Intensity Centroid** | - **BRIEF thuần:** Không có bước ước lượng góc hướng $\theta$, nên nếu người chụp cầm tay hơi nghiêng máy giữa 2 lần bấm sẽ bị trượt khớp (mismatch).<br>- **SIFT-like histograms:** Mỗi descriptor gồm 128 số `float` (tốn **512 bytes**, lớn gấp **16 lần** so với **32 bytes** của Rotated BRIEF 256-bit), làm nghẽn băng thông bộ nhớ toàn cục (Global Memory) ở bước Matching. |
| **4. Matching** | - Brute-force nearest neighbor (Hamming hoặc L2)<br>- Lowe's ratio test<br>- Cross-check | **Dùng cả 3:**<br>**Brute-Force Hamming** + **Lowe's Ratio Test ($0.75$)** + **Mutual Cross-Check** | - **Khoảng cách L2 (Euclid):** Chỉ dùng cho vector số thực (`float`) như SIFT. Vì bước 3 dùng descriptor nhị phân 256-bit nên khoảng cách **Hamming** (phép XOR + đếm bit `__popc`) nhanh hơn hàng chục lần. |
| **5. Homography** | - RANSAC with 4-point DLT<br>- Optional refinement on inliers<br>- Variants such as PROSAC | **RANSAC (2000 vòng) với Hartley-Normalized 4-point DLT** + **Least-Squares Refinement trên toàn bộ Inliers** | - **PROSAC:** Lấy mẫu có thứ tự ưu tiên khi tỉ lệ nhiễu quá cao. Nhờ bước 4 đã kết hợp cả *Lowe's Ratio Test* và *Mutual Cross-Check*, tỉ lệ Inliers của nhóm đạt tới **60%–85%**, nên RANSAC chuẩn hội tụ cực nhanh và cho phép tất cả các luồng GPU lấy mẫu độc lập hoàn toàn. |
| **6. Warping & Blending** | *Warping:*<br>- Inverse warping với Nearest hoặc Bilinear sampling<br>*Blending:*<br>- Overwrite<br>- Feathering<br>- Multi-band blending | **Inverse Warping với Bilinear Interpolation** + **Linear Feathering Blending** | - **Nearest sampling:** Gây hiện tượng răng cưa (aliasing) ở các đường chéo sau khi biến đổi phối cảnh.<br>- **Overwrite (Ghi đè):** Để lộ rõ đường cắt (seam) khi 2 ảnh có độ sáng/cân bằng trắng hơi lệch nhau.<br>- **Multi-band blending:** Phải xây dựng tháp Laplacian đa tầng trên toàn bộ khung ảnh Panorama, làm phình bộ nhớ và phức tạp hóa pipeline. **Feathering** theo khoảng cách biên xóa mượt vết ghép chỉ với vài phép toán trên mỗi pixel. |

---

## 3. Giải Thích Chi Tiết Từng Bước Trong Code `V0` (`src/v0_sequential.cpp`)

### Stage 1: Preprocessing (`rgb_to_grayscale` & `gaussian_blur_2d`)
1. **Chuyển sang ảnh xám (Luminance):**
   $$Y(x, y) = 0.299 R(x, y) + 0.587 G(x, y) + 0.114 B(x, y)$$
   Lưu dưới kiểu `float` để tránh mất mát độ chính xác hoặc tràn số nguyên khi tính đạo hàm bậc hai ở bước sau.
2. **Làm mờ Gaussian 2D ($5\times 5$, $\sigma = 1.4$):**
   $$G(u, v) = \frac{1}{Z} \exp\left(-\frac{u^2 + v^2}{2\sigma^2}\right), \quad u, v \in [-2, 2]$$
   Giúp triệt tiêu nhiễu hạt (sensor noise) để bộ lọc đạo hàm Sobel ở Stage 2 không bắt nhầm nhiễu thành góc.

---

### Stage 2: Keypoint Detection (`detect_harris_keypoints`)
1. **Đạo hàm bậc nhất bằng bộ lọc Sobel $3\times 3$:**
   Tính gradient ngang $I_x$ và dọc $I_y$, sau đó tính 3 thành phần tích $I_{xx} = I_x^2$, $I_{yy} = I_y^2$, $I_{xy} = I_x I_y$.
2. **Ma trận cấu trúc (Structure Tensor) & Độ phản hồi Harris $R$:**
   Cộng dồn $S_{xx}, S_{yy}, S_{xy}$ trong cửa sổ $3\times 3$ quanh mỗi pixel để tạo ma trận:
   $$M = \begin{bmatrix} S_{xx} & S_{xy} \\ S_{xy} & S_{yy} \end{bmatrix}$$
   Độ phản hồi góc Harris:
   $$R = \det(M) - k \cdot (\text{trace}(M))^2 = (S_{xx} S_{yy} - S_{xy}^2) - 0.04 (S_{xx} + S_{yy})^2$$
3. **Lọc biên, Non-Maximum Suppression (NMS $5\times 5$) & Top-$K$:**
   - Bỏ qua các điểm cách mép ảnh $< 16\text{ px}$ để đảm bảo vùng patch $31\times 31$ ở Stage 3 luôn nằm gọn trong ảnh.
   - Chỉ giữ các điểm có $R > 10^4$ và lớn hơn toàn bộ 24 điểm lân cận trong cửa sổ $5\times 5$.
   - Sắp xếp giảm dần theo $R$ (`std::partial_sort`) và giữ lại tối đa $K = 3000$ điểm mạnh nhất.

---

### Stage 3: Description (`compute_rotated_brief_descriptors`)
1. **Tính góc hướng $\theta$ bằng Intensity Centroid (chuẩn ORB):**
   Trong hình tròn bán kính $r = 15$ quanh keypoint $(c_x, c_y)$, tính moment bậc một:
   $$m_{10} = \sum_{u^2 + v^2 \le 15^2} u \cdot I(c_x + u, c_y + v), \quad m_{01} = \sum_{u^2 + v^2 \le 15^2} v \cdot I(c_x + u, c_y + v)$$
   Góc hướng của keypoint là $\theta = \text{atan2}(m_{01}, m_{10})$.
2. **Tạo chuỗi nhị phân 256-bit (`uint32_t bits[8]`):**
   - Dùng 256 cặp tọa độ mẫu cố định $(u_1, v_1), (u_2, v_2)$ sinh từ seed cố định (`42`).
   - Xoay từng cặp tọa độ theo góc $\theta$ bằng ma trận quay 2D $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$.
   - So sánh cường độ sáng tại 2 điểm sau khi xoay: nếu $I(\mathbf{p}_1) < I(\mathbf{p}_2)$ thì bật bit tương ứng lên `1`.

---

### Stage 4: Feature Matching (`match_descriptors`)
1. **Khoảng cách Hamming 256-bit:**
   Số bit khác nhau giữa hai descriptor `a` và `b` được tính bằng phép XOR và đếm số bit `1`:
   $$d_{\text{Hamming}}(a, b) = \sum_{k=0}^{7} \text{popcount}(\texttt{a.bits}[k] \oplus \texttt{b.bits}[k])$$
2. **Lọc nhiễu kép (Lowe's Ratio Test + Mutual Cross-Check):**
   - **Chiều thuận ($A \rightarrow B$):** Với mỗi điểm $i \in A$, tìm điểm gần nhất $j_1 \in B$ (khoảng cách $d_1$) và gần nhì $j_2 \in B$ (khoảng cách $d_2$). Chỉ giữ nếu $d_1 \le 64$ và $d_1 < 0.75 \times d_2$.
   - **Kiểm tra chéo ($B \rightarrow A$):** Điểm $j_1 \in B$ khi tìm điểm gần nhất bên ảnh $A$ cũng phải trỏ ngược về đúng điểm $i \in A$. Bước này loại bỏ gần như toàn bộ các điểm khớp nhập nhằng trên bề mặt lặp lại (như gạch, cửa sổ).

---

### Stage 5: Homography Estimation (`estimate_homography_ransac`)
1. **Chuẩn hóa Hartley (Hartley Normalization) — Quyết định độ chính xác số học:**
   - Ở độ phân giải Full HD / 4K, tọa độ pixel $(x, y)$ lên tới hàng nghìn, nên các số hạng $x \cdot x'$ trong ma trận DLT lớn gấp hàng triệu lần số hạng $1$, làm hệ phương trình bị suy biến (ill-conditioned).
   - Trước khi giải DLT, tọa độ các điểm được dịch trọng tâm về $(0,0)$ và co giãn sao cho khoảng cách trung bình tới tâm bằng $\sqrt{2}$ thông qua hai ma trận biến đổi $T_A, T_B$.
2. **Giải hệ phương trình tuyến tính $8\times 8$ (Normalized DLT):**
   - Đặt $h_{33} = 1$, mỗi cặp điểm đóng góp 2 phương trình tuyến tính cho 8 ẩn $(h_{11}, h_{12}, h_{13}, h_{21}, h_{22}, h_{23}, h_{31}, h_{32})$.
   - Thay vì dùng thư viện SVD bên ngoài, code lập hệ phương trình chuẩn $(\mathbf{A}^T \mathbf{A}) \mathbf{h} = \mathbf{A}^T \mathbf{b}$ kích thước $8\times 8$ và giải trực tiếp bằng **Khử Gauss có chọn phần tử trụ lớn nhất (Gaussian Elimination with Partial Pivoting)**. Cách này vừa dùng được cho đúng 4 điểm ($N=4$ trong vòng lặp RANSAC), vừa dùng lại nguyên vẹn cho bước tinh chỉnh với $N \ge 4$ điểm Inliers!
   - Khử chuẩn hóa: $\mathbf{H} = T_A^{-1} \tilde{\mathbf{H}} T_B$.
3. **Tinh chỉnh trên tập Inliers (Inlier Least-Squares Refinement):**
   - Sau 2000 vòng lặp RANSAC, lấy toàn bộ tập Inliers tốt nhất (các điểm có sai số chiếu $< 3.0\text{ px}$) đưa vào giải lại hệ bình phương tối thiểu $\mathbf{A}^T \mathbf{A} \mathbf{h} = \mathbf{A}^T \mathbf{b}$ để thu được ma trận $\mathbf{H}$ chính xác tới mức dưới 0.15 pixel (`0.0968 px` ở 720p).

---

### Stage 6: Warping & Blending (`warp_and_blend`)
1. **Xác định khung ảnh Panorama (Bounding Box):**
   Chiếu 4 góc của ảnh B qua $\mathbf{H}$ sang hệ tọa độ ảnh A, kết hợp với 4 góc của ảnh A để tìm `[min_x, max_x] × [min_y, max_y]`.
2. **Biến đổi ngược (Inverse Warping) & Nội suy Bilinear:**
   - Tính ma trận nghịch đảo $\mathbf{H}^{-1}$.
   - Với mỗi pixel $(x_{out}, y_{out})$ trên khung Panorama, đổi về tọa độ $(x_A, y_A)$ và nhân với $\mathbf{H}^{-1}$ để tìm tọa độ thực $(u_B, v_B)$ trên ảnh B.
   - Nếu $(u_B, v_B)$ nằm trong ảnh B, lấy màu từ 4 pixel nguyên lân cận bằng nội suy song tuyến tính (**Bilinear Interpolation**).
3. **Hòa trộn mượt bằng khoảng cách tới biên (Distance-to-Border Linear Feathering):**
   - Nếu pixel chỉ thuộc ảnh A $\rightarrow$ lấy màu ảnh A.
   - Nếu pixel chỉ thuộc ảnh B $\rightarrow$ lấy màu ảnh B.
   - Nếu nằm trong vùng giao nhau của cả A và B: tính khoảng cách từ điểm đó tới mép gần nhất của ảnh A ($d_A$) và mép gần nhất của ảnh B ($d_B$):
     $$\alpha = \frac{d_A}{d_A + d_B}, \quad C_{\text{out}} = \alpha \cdot C_A + (1 - \alpha) \cdot C_B$$
   - Khi đi sát về phía biên của ảnh B ($d_B \to 0$), trọng số $\alpha \to 1$ (nghiêng hẳn về ảnh A) và ngược lại, giúp vết nối giữa hai ảnh biến mất hoàn toàn.
