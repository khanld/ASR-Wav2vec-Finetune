# Vietnamese Speech Recognition
### Cách mà mình đã train bộ Tiếng Việt (Optional - Vì lần đầu làm nên còn nhiều xử lý không được chuẩn)
- chars_to_ignore: ```'[^\ a-zA-Z_àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]'```
- Bộ FOSD và VLSP mình xử lý như sau:
	- FOSD: Còn chứa các ký tự số, cần được chuyển về dạng chữ. Mình có sử dụng bộ converter [vietnam_number](https://pypi.org/project/vietnam-number/) nhưng sẽ có một vài hạn chế như: ̣94 có thể sẽ bị chuyển thành chín tư dù nngười đọc là chín bốn. Ngoài ra quan sát dữ liệu thì mình thấy có 1 vài độ đo bị viết tắt dạng như 4m, 5l,... các ký tự này mình xóa khi train (các bạn có thể giữ nguyên hoặc thay bằng ```<unk>```). 
	- VLSP (không nên theo): Mình xóa trước các từ ```<unk>``` xuất hiện trong dataset, do chars_to_ignore của mình sẽ xóa đi ký tự ```>``` và ```<```. Kết quả là nó sẽ thành ```unk``` và tokenizer nó không nhận ra được. Lúc train xong rồi thì mình mới phát hiện thì đã muộn :D, không khuyến khích làm theo nha!


- Đây là file [base_dataset.py](base_dataset_vietnamese.py) mà mình đã sử dụng để train bộ Tiếng Việt, nếu không có gì sai sót thì bạn sẽ nhận được một file [vocab.json](vocab.json) như này khi chạy.
- Ngoài ra lúc train thì mình có finetune learning rate một vài lần do ban đầu mình set lớn quá mô hình bị diverge (khuyến khích set learning rate trong khoảng 1e-6 -> 5e-5), nên không thể reproduce đúng mô hình mình public trên huggingface được. Tuy nhiên thì mình tin là chạy lại vẫn sẽ được kết quả tốt như vậy thôi.