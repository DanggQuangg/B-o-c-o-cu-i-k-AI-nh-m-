Trình mô phỏng Thuật toán Tìm đường AI (AI Pathfinding Visualizer)

Đây là một dự án Python sử dụng thư viện Pygame để mô phỏng và trực quan hóa hoạt động của nhiều thuật toán tìm đường trong lĩnh vực Trí tuệ Nhân tạo. Người dùng có thể điều khiển một chiếc xe vượt qua một bản đồ địa hình 2D với các độ cao khác nhau, đồng thời phải đối mặt với các ràng buộc về nhiên liệu và độ dốc.

Tính năng chính

Bản đồ địa hình tương tác: Một lưới ô vuông nơi mỗi ô có độ cao riêng, ảnh hưởng đến chi phí di chuyển.

Hệ thống ràng buộc thực tế:

Nhiên liệu (Fuel): Mỗi bước di chuyển tiêu tốn một lượng xăng nhất định.

Độ dốc (Slope): Xe không thể di chuyển giữa hai ô có chênh lệch độ cao quá lớn.

Điều khiển đa dạng:

Thủ công: Người dùng có thể tự lái xe bằng các phím mũi tên hoặc WASD.

Tự động: Chọn một thuật toán từ danh sách, và xe sẽ tự động di chuyển theo đường đi được tìm thấy.

Bộ sưu tập thuật toán AI phong phú: Triển khai hơn 15 thuật toán tìm đường và giải quyết vấn đề kinh điển.

Giao diện trực quan:

Thanh bên trái chứa các nút để kích hoạt thuật toán.

Khu vực trung tâm hiển thị bản đồ, xe, và đường đi đã qua.

Thanh bên phải ghi lại nhật ký (log) về các hành động và kết quả tìm đường.

Tùy chỉnh dễ dàng: Bản đồ (RAW_MAP) và các tham số vật lý (chi phí di chuyển, độ dốc tối đa) có thể dễ dàng thay đổi ngay trong mã nguồn.

Yêu cầu

Python 3.x

Thư viện Pygame

Cài đặt & Chạy chương trình

Cài đặt Pygame: Mở terminal hoặc command prompt và chạy lệnh sau:

pip install pygame


Chuẩn bị tài nguyên (Assets):
Chương trình sẽ hoạt động tốt nhất nếu có các file hình ảnh sau trong cùng thư mục với file Python:

car_up.png, car_down.png, car_left.png, car_right.png (Hình ảnh xe ở 4 hướng)

h0.png, h1.png, h2.png, h3.png, h4.png (Hình ảnh cho các loại địa hình theo độ cao)

water.png (Hình ảnh cho các ô "hố" - 'X')

home.png (Hình ảnh cho ô đích)

Lưu ý: Nếu không tìm thấy các file ảnh này, chương trình vẫn sẽ chạy bằng cách vẽ các hình khối đơn giản để thay thế.

Chạy chương trình:
Lưu mã nguồn vào một file (ví dụ: main.py) và chạy nó từ terminal:

python main.py


Cách sử dụng

Nhập nhiên liệu: Khi chương trình khởi động, một màn hình sẽ yêu cầu bạn nhập vào lượng xăng ban đầu. Hãy nhập một số dương và nhấn Enter.

Giao diện chính:

Bên trái: Bảng điều khiển chứa danh sách các thuật toán. Nhấn vào một nút để yêu cầu thuật toán đó tìm đường đi từ vị trí hiện tại của xe.

Giữa: Bản đồ trò chơi.

Bên phải: Bảng ghi lại trạng thái, thông báo kết quả tìm đường, chi phí di chuyển, và các lỗi (nếu có).

Điều khiển thủ công:

Sử dụng các phím mũi tên (Lên, Xuống, Trái, Phải) hoặc W, A, S, D để di chuyển xe.

Chạy thuật toán:

Nhấn vào tên một thuật toán ở bảng bên trái.

Chương trình sẽ tìm đường đi. Nếu thành công, xe sẽ tự động di chuyển theo lộ trình đã vạch ra.

Kết quả và độ dài đường đi sẽ được in ra bảng trạng thái.

Các phím tắt:

R: Reset lại toàn bộ màn chơi (đưa xe về vị trí xuất phát, phục hồi nhiên liệu).

Esc: Thoát khỏi chương trình.

Các thuật toán được triển khai

Chương trình mô phỏng một loạt các thuật toán từ cơ bản đến nâng cao.

Tìm kiếm thông thường (Uninformed/Informed Search)

A*: Thuật toán tìm kiếm ưu tiên tốt nhất, sử dụng heuristic để tối ưu.

UCS (Uniform Cost Search): Tìm đường đi với tổng chi phí thấp nhất.

BFS (Breadth-First Search): Tìm đường đi ngắn nhất về số bước.

IDS (Iterative Deepening Search): Kết hợp ưu điểm của BFS và DFS.

DFS (Depth-First Search): Tìm kiếm theo chiều sâu.

Greedy Best-First Search: Tìm kiếm dựa hoàn toàn vào heuristic.

Beam Search: Một biến thể của BFS chỉ giữ lại một số trạng thái tốt nhất ở mỗi bước.

Tìm kiếm cục bộ (Local Search)

Hill Climbing: Luôn di chuyển đến hàng xóm tốt nhất (gần đích hơn).

Simulated Annealing (Luyện kim mô phỏng): Cho phép các bước đi "tệ hơn" một cách có xác suất để thoát khỏi các cực tiểu cục bộ.

Ràng buộc & Quay lui (Constraint & Backtracking)

AC-3 Reduce: Thuật toán thỏa mãn ràng buộc để loại bỏ các ô không hợp lệ khỏi bản đồ trước khi tìm kiếm.

Forward Checking: Tìm kiếm quay lui kết hợp "nhìn trước" để cắt tỉa các nhánh vô vọng.

Backtracking: Thuật toán quay lui cơ bản.

Lập kế hoạch (Planning)

Uncertain Action (AND-OR Search): Lập kế hoạch cho các hành động có kết quả không chắc chắn (ví dụ: đi thẳng nhưng có thể bị trượt sang trái).

Belief Search (Conformant Planning): Tìm một kế hoạch chắc chắn thành công khi vị trí ban đầu không hoàn toàn chắc chắn.

Belief-Partial Search: Một biến thể linh hoạt hơn của Belief Search, cho phép hành động thành công với một tỷ lệ nhất định.

Tùy chỉnh

Bạn có thể dễ dàng thay đổi bản đồ bằng cách chỉnh sửa biến RAW_MAP ở đầu file mã nguồn.

"S": Điểm xuất phát (Start).

"G": Điểm đích (Goal).

"X": Ô hố, không thể đi vào.

0-5: Các số nguyên đại diện cho độ cao của địa hình.
