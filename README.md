Chương trình nhận dạng biển số xe trong kho bãi, được dùng cho biển số xe Việt Nam cả 1 và 2 hàng. Sử dụng xử lý ảnh OpenCV và thuật toán KNN



HOW TO USE:
- "GenData.py" để
- "Image_detect.py" để nhận diện, phát hiện dữ liệu đầu vào là ảnh
- "Video_detect.py" để nhận diện, phát hiện dữ liệu đầu vào là ảnh



## CÁC BƯỚC CHÍNH TRONG CỦA 1 BÀI TOÁN NHẬN DẠNG BIỂN SỐ XE

1. Phát hiện vị trí và tách biển số xe
2. Phân đoạn kí tự
3. Nhận diện kí tự


## PHÁT HIỆN VÀ TÁCH BIỂN SỐ:
1. Chụp ảnh từ camera
2. Chuyển ảnh sang ảnh xám
3. Tăng độ tương phảm
4. Giảm nhiễu bằng bộ lọc Gauss
5. Nhị phân hóa ảnh với ngưỡng động
6. Phát hiện biên bằng phương pháp Canny
7. Tìm vị trí và lọc biển số bằng Contour

## Phân tách kí tự:

 1. xoay biển số về đúng chính diện
    Phương pháp xoay ảnh sử dụng ở đây là:
        * Lọc ra tọa độ 2 đỉnh A,B nằm dưới cùng của biển số
        *	Từ 2 đỉnh có tọa độ lần lượt là A(x1, y1) và B(x2,y2) ta có thể tính được cạnh đối và cạnh kề của tam giác ABC
        *	Tính góc quay bằng hàm tan()
        *	Xoay ảnh theo góc quay đã tính. Nếu ngược lại điểm A nằm cao hơn điểm B ta cho góc quay âm


    2. Từ ảnh nhị phân, ta lại tìm contour cho các kí tự (phần màu trắng). Sau đó vẽ những hình chữ nhật bao quanh các kí tự đó. Tuy nhiên việc tìm contour này cũng bị nhiễu dẫn đến việc máy xử lý sai mà tìm ra những hình ảnh không phải kí tự. Ta sẽ áp dụng các đặc điểm về tỉ lệ chiều cao/rộng của kí tự, diện tích của kí tự so với biển số 


## Nhận dạng kí tự ##

KNN là một trong những thuật toán học có giám sát đơn giản nhất trong Machine Learning, hoạt động theo quy trình gồm 4 bước chính:
1.	Xác định tham số K (số láng giềng gần nhất). 
2.	Tính khoảng cách từ điểm đang xét đến tất cả các điểm trong tập dữ liệu cho trước
3.	Sắp xếp các khoảng cách đó theo thứ tự tăng dần
4.	Xét trong tập K điểm gần nhất với điểm đang xét, nếu số lượng điểm của loại nào cao hơn thì coi như điểm đang xét thuộc loại đó

Vì mỗi kí tự có kích thước khác nhau xử lý phức tạp nên cần chuẩn hóa hình ảnh lại với kích thước cao:rộng là 30:20 pixels. Thay vì mỗi kí tự đưa vào mô hình để máy nhận diện thì những kí tự này sẽ được ta gắn nhãn bằng những phím bấm trên máy tính. Sau khi gắn nhãn hết các kí tự ta sẽ lưu hai file `.txt` là `classifications.txt` và `flattened_images.txt`. File `classifications.txt` có nhiệm vụ lưu các mã ASCII của các kí tự đó và file `flattened_images.txt` sẽ lưu giá trị các điểm ảnh có trong hình ảnh kí tự (hình 20x30 pixel có tổng cộng 600 điểm ảnh có giá trị 0 hoặc 255)

Tiếp đó ta thực hiện đưa ảnh đang xét vào và tính khoảng cách đến tất cả các điểm trong mẫu, kết quả sẽ là mã ASCII đại điện cho hình ảnh đó. Cuối cùng ta in biển số xe ra hình. 

## Nhận dạng kí tự ##
**Result**

| Category |	Total number of plates	| Nunmber of found plates | Percentage(%) |
|:---------------------:|:----------------------:|:-------------------------:|:-------:|
|1 row plate  |	370     |	182                  |	49,2          | 
|2 row plate	| 2349 |	924 |	39,3 |

Khi ta quay theo nhiều góc độ, nhiều vị trí dẫn đến khi tính toán diện tích, tỉ lệ cao/rộng của biển số không còn thỏa điều kiện đặt ra nên đã bị loại. Biển số có thể bị ảnh hưởng bởi những chi tiết ngoài nên khi xấp xỉ contour không ra hình tứ giác, dẫn đến cũng gây mất biển số. Lỗi này đặc biệt xảy ra ở những xe ô tô vì ô tô thường có nền xung quanh biển số là những vật liệu phản chiếu ánh sáng mạnh, gây ảnh hưởng lớn đến quá trình xác định vùng biển số. 


Trong quá trình xử lý, việc xử lý nhị phân cũng đóng vai trò quan trọng, ảnh bị nhiễu và bản thân biển số bị tối, dính nhiều bụi dẫn đến khi xử lý nhị phân sẽ bị đứt đoạn và vẻ contour bị sai, để khắc phục cần sử dụng những phép toán hình thái học như phép nở, phép đóng để làm liền những đường màu trắng trong ảnh nhị phân.


Nhìn chung mô hình nhận diện KNN cũng khá tốt, có những kí tự dù bị mờ, bị nghiêng vẫn nhận diện đúng. Điều này một phần nhờ vào chương trình đã xoay biển số lại cho để tăng khả năng nhận diện, cho dù nghiêng thì kí tự cũng chỉ nghiêng từ 3° đến 7°. Tuy nhiên vẫn còn nhầm lẫn nhiều giữa các kí tự như số 1 với số 7. Chữ G, chữ D, số 6 với số 0. Chữ B với số 8...



