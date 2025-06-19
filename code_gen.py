import os
import json
# Import the new OpenAI client
from openai import OpenAI

def generate_hf_inference_code(problem_txt_path: str,
                               hf_model_config: dict,
                               data_dir: str,
                               openai_api_key: str) -> str:
    """
    Đọc mô tả bài toán, cấu hình HF model và data dir, rồi gọi model OpenAI
    để sinh ra code Python hoàn chỉnh cho inference với transformers.

    Args:
        problem_txt_path (str): đường dẫn đến file .txt chứa mô tả bài toán.
        hf_model_config (dict): JSON chứa thông tin model HF (ví dụ {'model_name': ..., 'revision': ..., ...}).
        data_dir (str): đường dẫn đến thư mục chứa data.
        openai_api_key (str): API key để gọi OpenAI.

    Returns:
        str: Mã nguồn Python do GPT sinh ra.
    """
    # Đọc nội dung file bài toán
    if not os.path.isfile(problem_txt_path):
        raise FileNotFoundError(f"Không tìm thấy file: {problem_txt_path}")
    with open(problem_txt_path, 'r', encoding='utf-8') as f:
        problem_description = f.read()

    # Serialize HF model config
    hf_config_json = json.dumps(hf_model_config, indent=2, ensure_ascii=False)
    url = hf_model_config.get("url")
    model_name = hf_model_config.get("model_name_from_title")
    input_info = hf_model_config.get("headings_with_content", {}).get("📥 Đầu vào", "")
    output_info = hf_model_config.get("headings_with_content", {}).get("📤 Đầu ra", "")
    usage_example = hf_model_config.get("headings_with_content", {}).get("🧪 Sử dụng mô hình", "")

    # Tạo prompt cho GPT
    prompt = f"""
You are a senior ML engineer. 
Dựa vào các thông tin sau:

1. Problem description (context + yêu cầu đầu ra):
\"\"\"
{problem_description}
\"\"\"

2. HuggingFace model configuration (JSON):
- url: {url}
- model_name: {model_name}
- input_info: {input_info}
- output_info: {output_info}
- HƯỚNG DẪN SỬ DỤNG: {usage_example} 

3. Data directory:
{data_dir}

Hãy viết một script Python hoàn chỉnh (có thể chạy được) để:
- Dùng thư viện `transformers` load đúng model và tokenizer từ HuggingFace, nếu cách hướng dẫn sử dụng không dùng cách này thì làm theo hướng dẫn sử dụng (nhớ import các thư viện cần thiết của các module được đề cập đến trong hướng dẫn).
- Nếu input data (trong mô tả) không đúng shape mà model yêu cầu (ví dụ độ dài sequence, kích thước ảnh, v.v.), tự động detect và scale/crop/pad cho phù hợp.
- Đọc dữ liệu từ `{data_dir}` nếu có nhiều data như ảnh, hoặc một file csv duy nhất, thực hiện inference theo yêu cầu bài toán.
- Xuất kết quả cuối cùng theo đúng định dạng được mô tả trong problem description.

Yêu cầu:
- Comment rõ từng bước bằng tiếng Việt.
- LÀM ĐÚNG THEO HƯỚNG DẪN SỬ DỤNG MODEL VÀ IMPOR CÁC THƯ VIỆN CẦN THIẾT.
- Sử dụng best practices: device selection (CPU/GPU), batch processing nếu có thể.
- Import các thư viện cần thiết (NHỚ LÀM).
- Trả về code Python hoàn chỉnh duy nhất, không thêm giải thích.
"""

    # --- UPDATED OPENAI API CALL ---
    # 1. Instantiate the OpenAI client with your API key
    try:
        client = OpenAI(api_key=openai_api_key)

        # 2. Call the chat.completions.create method
        resp = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14", 
            messages=[
                {"role": "system", "content": "You are a helpful Python and Machine Learning expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            # max_tokens=4000 
        )

        # 3. Access the response content
        code = resp.choices[0].message.content.strip()
        
        # Clean up the response if it includes markdown code block fences
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        if code.endswith("```"):
            code = code[:-len("```")].strip()

        return code
    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        return ""


# Ví dụ gọi hàm:
if __name__ == "__main__":
    # Đường dẫn và cấu hình ví dụ
    problem_txt_path = "Task1/task.txt"
    data_folder = "Task1/test.csv"

    hf_config = {"url": "https://huggingface.co/zhaospei/Model_2", "model_name_from_title": "zhaospei/Model_2 · Hugging Face", "description_from_meta": "We’re on a journey to advance and democratize artificial intelligence through open source and open science.", "headings_with_content": {"zhaospei/Model_2like0": "Safetensors\nModel card Files Files and versions Community", "🏠 Mô hình Wide & Deep Neural Network - Dự đoán Giá Nhà California": "No content found", "📝 Mô tả": "Đây là một mô hình Wide & Deep Neural Network được huấn luyện trên tập dữ liệu California Housing để dự đoán giá nhà trung bình ( MedHouseVal ). Mô hình được xây dựng bằng PyTorch , dựa trên kiến trúc trong cuốn Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow của Aurélien Géron .", "📌 Nhiệm vụ": "Dự đoán giá nhà dựa trên dữ liệu bảng (tabular regression) với 8 đặc trưng đầu vào.", "📥 Đầu vào": "Số chiều : [batch_size, 8] Kiểu dữ liệu : torch.FloatTensor Các đặc trưng đầu vào : 'MedInc' – Thu nhập trung vị 'HouseAge' – Tuổi trung bình của căn nhà 'AveRooms' – Số phòng trung bình 'AveBedrms' – Số phòng ngủ trung bình 'Population' – Dân số 'AveOccup' – Số người trung bình trên mỗi hộ 'Latitude' – Vĩ độ 'Longitude' – Kinh độ", "📤 Đầu ra": "Kiểu : torch.FloatTensor có shape [batch_size, 1] Ý nghĩa : Giá nhà trung bình dự đoán (giá trị thực).", "🧪 Cách sử dụng mô hình": "Dưới đây là ví dụ về cách sử dụng mô hình với dữ liệu đầu vào giả lập:\nimport torch import torch.nn as nn from huggingface_hub import PyTorchModelHubMixin # Tạo dữ liệu đầu vào giả lập (batch 1, 8 features) x_input = torch.randn( 1 , 8 ) print ( \"Mock input:\" ) print (x_input) # Định nghĩa mô hình Wide & Deep Neural Network class WideAndDeepNet (nn.Module, PyTorchModelHubMixin): def __init__ ( self ): super ().__init__()\n        self.hidden1 = nn.Linear( 6 , 30 )\n        self.hidden2 = nn.Linear( 30 , 30 )\n        self.main_head = nn.Linear( 35 , 1 )\n        self.aux_head = nn.Linear( 30 , 1 )\n        self.main_loss_fn = nn.MSELoss(reduction= 'sum' )\n        self.aux_loss_fn = nn.MSELoss(reduction= 'sum' ) def forward ( self, input_wide, input_deep, label= None ):\n        act = torch.relu(self.hidden1(input_deep))\n        act = torch.relu(self.hidden2(act))\n        concat = torch.cat([input_wide, act], dim= 1 )\n        main_output = self.main_head(concat)\n        aux_output = self.aux_head(act) if label is not None :\n            main_loss = self.main_loss_fn(main_output.squeeze(), label)\n            aux_loss = self.aux_loss_fn(aux_output.squeeze(), label) return WideAndDeepNetOutput(main_output=main_output, aux_output=aux_output) # Tải mô hình từ Hugging Face Hub model = WideAndDeepNet.from_pretrained( \"sadhaklal/wide-and-deep-net-california-housing-v3\" )\nmodel. eval () # Dự đoán với mô hình with torch.no_grad():\n    prediction = model(x_input) print ( f\"Giá nhà dự đoán (mock input): {prediction.item(): .3 f} \" )"}}

    # Lấy API key từ biến môi trường để bảo mật hơn
    # For this example, we'll continue using the hardcoded key from your script.
    # It's highly recommended to use environment variables in production.
    # my_api_key = os.getenv("OPENAI_API_KEY") 
    my_api_key = "YOUR_OPENAI_API_KEY"

    if my_api_key == "YOUR_OPENAI_API_KEY":
        print("="*60)
        print("!!! CẢNH BÁO: Vui lòng thay 'YOUR_OPENAI_API_KEY' bằng API key thật của bạn.")
        print("="*60)
    else:
        print("Đang tạo mã nguồn Python, vui lòng đợi...")
        generated_code = generate_hf_inference_code(problem_txt_path, hf_config, data_folder, openai_api_key=my_api_key)
        
        if generated_code:
            print("\n--- MÃ NGUỒN ĐƯỢC TẠO RA ---\n")
            print(generated_code)
            
            # Optionally, save the generated code to a file
            with open("generated_inference_script.py", "w", encoding="utf-8") as f:
                f.write(generated_code)
            print("\n--- ĐÃ LƯU MÃ NGUỒN VÀO FILE 'generated_inference_script.py' ---")

