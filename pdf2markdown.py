import requests
import json
import os

url = "https://api.netmind.ai/inference-api/agent/v1/parse-pdf"

url2="https://github.com/protagolabs/NetMind-RS-KnowledgeRAG/raw/GuoCheng/pdf_to_markdown/pdf/"

headers = {
   'Authorization': 'Bearer ffcba38d4999436a986939e216a61a9f',
   'Content-Type': 'application/json'
}

for file in os.listdir("pdf_to_markdown/pdf"):
    
    file=file.replace(" ","%20")
   #  print(file)
    new_url=url2+file
    # print(new_url)
    payload = json.dumps({
   "url":new_url,
   "format": "markdown",
   "vlm": True
    })
    file=file.replace("%20"," ")
    file=file.replace(".pdf",".md")
    response = requests.request("POST", url, headers=headers, data=payload)

    # with open("pdf_to_markdown/markdown/"+file, "w", encoding="utf-8") as f:
    #     f.write(response.text)
