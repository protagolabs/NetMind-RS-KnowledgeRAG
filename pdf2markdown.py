from PyPDF2 import PdfReader, PdfWriter
import requests
import json
import os

MAX_SIZE = 20 * 1024 * 1024  # 20MB


url = "https://api.netmind.ai/inference-api/agent/v1/parse-pdf"

url2="https://github.com/protagolabs/NetMind-RS-KnowledgeRAG/raw/GuoCheng/pdf_to_markdown/year_reports_of_NVDA/"

headers = {
   'Authorization': 'Bearer ffcba38d4999436a986939e216a61a9f',
   'Content-Type': 'application/json'
}
def split_pdf_recursive(input_file, base_name, counter=None):
    """
    递归拆分 PDF 文件，使得每个部分 <= MAX_SIZE
    counter: 一个 list，用来存储全局编号
    """
    if counter is None:
        counter = [1]  # 用 list 包装，保证递归里共享引用
    

    if os.path.getsize(input_file) <= MAX_SIZE:
        return

    file_name = os.path.basename(input_file)
    if file_name[-5]==")":
        file_name=file_name[:-7]+".pdf"
    name_without_ext = os.path.splitext(file_name)[0]
    print(f"📄 拆分: {file_name}")

    reader = PdfReader(input_file)
    num_pages = len(reader.pages)

    # 如果只剩 1 页，无法继续拆分
    if num_pages <= 1:
        print(f"⚠️ 文件 {file_name} 只有 {num_pages} 页，无法继续拆分，但仍然超出大小限制。")
        return

    # 拆分点
    mid = num_pages // 2
    parts = [(0, mid), (mid, num_pages)]

    for start, end in parts:
        out_file = os.path.join(base_name, f"{name_without_ext}({counter[0]}).pdf")

        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        with open(out_file, "wb") as f:
            writer.write(f)
        print(f"📂 生成: {out_file}")

        counter[0] += 1  # 全局编号自增

        # 递归继续拆分
        split_pdf_recursive(out_file, base_name, counter)

    # 删除源文件
    os.remove(input_file)
    print(f"🗑️ 已删除源文件: {input_file}")

def process_file_number(floder_name):
    files=sorted(os.listdir(floder_name))
    file_number={}
    for file in files:
        if file[-5]!=")":
            continue
        else:
            original_file_name=file[:-7]
            if original_file_name not in file_number.keys():
                file_number[original_file_name]=1
                os.rename(os.path.join(floder_name,file),os.path.join(floder_name,original_file_name+"(1).pdf"))
            else:
                file_number[original_file_name]+=1
                os.rename(os.path.join(floder_name,file),os.path.join(floder_name,original_file_name+"("+str(file_number[original_file_name])+").pdf"))

def convert_pdf_to_markdown(input_folder,output_folder):
    for file in os.listdir(input_folder):
        if not file.endswith(".pdf"):
            continue
        file=file.replace(" ","%20")
        print(file)
        new_url=url2+file
    #  print(new_url)
        payload = json.dumps({
            "url":new_url,
            "format": "markdown",
            "vlm": True
        })
        file=file.replace("%20"," ")
        file=file.replace(".pdf",".md")
        response = requests.request("POST", url, headers=headers, data=payload)

        with open(os.path.join(output_folder, file), "w", encoding="utf-8") as f:
            f.write(response.text)

if __name__ == "__main__":
    # for file in os.listdir("pdf_to_markdown/year_reports_of_NVDA"):
    #     if not file.endswith(".pdf"):
    #         continue
    #     split_pdf_recursive(os.path.join("pdf_to_markdown/year_reports_of_NVDA",file), "pdf_to_markdown/year_reports_of_NVDA")
    # process_file_number("pdf_to_markdown/year_reports_of_NVDA")
    convert_pdf_to_markdown("pdf_to_markdown/year_reports_of_NVDA","pdf_to_markdown/year_reports_of_NVDA_markdown")

