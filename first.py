# 导入所需模块
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredImageLoader
import yaml  # 新增此行
from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.embeddings import Embeddings  # 导入基类
from langchain_community.vectorstores.utils import filter_complex_metadata
from transformers import AutoTokenizer

from typing import List
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from PIL import Image
import os
from accelerate import init_empty_weights
from transformers import CLIPProcessor, CLIPModel

# 标准化图像加载器
# class SimpleImageLoader:
#     def __init__(self, path: str):
#         self.image_paths = [os.path.join(path, f) for f in os.listdir(path) 
#                           if f.lower().endswith('.png')]

#     def load(self) -> List[Document]:
#         docs = []
#         for img_path in self.image_paths:
#             with Image.open(img_path) as img:
#                 docs.append(Document(
#                     page_content=f"PNG图像: {os.path.basename(img_path)}",
#                     metadata={
#                         "source": img_path,
#                         "width": img.width,
#                         "height": img.height,
#                         "content_type": "image"
#                     }
#                 ))
#         return docs

import torch
print(torch.backends.mps.is_available())  # 输出应为True
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"MPS可用: {torch.backends.mps.is_available()}")
from pathlib import Path

class CLIPImageLoader:
    def __init__(self, path: str):
        self.image_paths = [os.path.join(path, f) for f in os.listdir(path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def load(self) -> List[Document]:
        docs = []
        for img_path in self.image_paths:
            image = Image.open(img_path)
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                features = self.model.get_image_features(**inputs).numpy().tolist()[0]
            docs.append(Document(
                page_content="",  # 可保留空或添加描述
                metadata={
                    "source": img_path,
                    "vector": features,  # 存储CLIP特征向量
                    "content_type": "image"
                }
            ))
        return docs
    
# 加载本地文本文件（示例路径：data/sample.txt）
text_loader = DirectoryLoader(
    path="doc/",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"autodetect_encoding": True},  # 移除metadata_columns
    show_progress=True
)
text_docs = text_loader.load()
text_files = list(Path("doc/").glob("**/*.md")) + list(Path("doc/").glob("**/*.txt"))
print(f"匹配的文本文件: {[str(f) for f in text_files]}")


# 手动添加元数据类型标记
for doc in text_docs:
    doc.metadata["content_type"] = "text"

# 2. 修正文本分割逻辑
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10,
    separators=["\n\n", "\n", "。", "！", "？", "；", "——", "…"]  # 添加中文分隔符
)
splits = text_splitter.split_documents(text_docs)

# 3. 验证分割结果
print(f"原始文档数: {len(text_docs)}")
print(f"分割后文档数: {len(splits)}")


image_loader = CLIPImageLoader("doc/img/")
# 过滤元数据（保留默认支持的str/int/float/bool）


documents = splits + image_loader.load()
documents = filter_complex_metadata(documents)

print('documents',documents)


class CLIPEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 处理文本嵌入
        inputs = self.processor(text=texts,
                                 return_tensors="pt", 
                                 padding=True,
                                 max_length=77,
                                 truncation=True,  # 添加截断
)
        with torch.no_grad():
            return self.model.get_text_features(**inputs).tolist()
    
    def embed_query(self, query: str) -> List[float]:
        # 处理查询嵌入
        inputs = self.processor(text=query, return_tensors="pt",max_length=77, truncation=True,  # 添加截断
)
        with torch.no_grad():
            return self.model.get_text_features(**inputs).tolist()[0]
        
clip_embeddings = CLIPEmbeddings("openai/clip-vit-base-patch32")





# 使用多模态嵌入模型（如 Llama 3.2 或 Bakllava）
# embeddings = OllamaEmbeddings(model="llama3.2-vision")  # 支持图像与文本联合编码
# 生成跨模态向量
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=clip_embeddings,  # 使用自定义嵌入类
    persist_directory="./chroma_db"
)

# 动态计算k值（至少1，不超过文档总数的1/3）
valid_doc_count = len(documents)
retriever = vectorstore.as_retriever(    search_kwargs={
        "k": max(5, max(1, valid_doc_count // 3)),
        # "boost": [
        #     {"content_type": "text", "factor": 2.0},  # 文本权重加倍
        #     {"content_type": "image", "factor": 1.0}
        # ]
    })

print('retriever',retriever)


# 查询所有文档（示例）
docs = vectorstore.get(include=["documents", "metadatas"])
print("文档内容:", docs["documents"])
print("元数据:", docs["metadatas"])



# 配置多模态生成模型
llm = OllamaLLM(
    model="llama3.2-vision",  # 支持图文混合输入
    temperature=0.3,
    num_threads= 4
)

# 辅助函数定义（必须在链调用前）
def _format_docs(docs):
    formatted = []
    for doc in docs:
          # 添加默认值处理
        content_type = doc.metadata.get("content_type", "text")
        source = doc.metadata.get("source", "unknown")
        print('doc.page_content',doc)
        if content_type == "image":
            # 生成Markdown图像引用
            formatted.append(f"![相关图片]({source})")
        else:
            formatted.append(doc.page_content)
    return "\n\n".join(formatted)



# 运行时动态加载
with open("prompts/tech_qa.yaml") as f:
    prompt_template = yaml.safe_load(f)["system_prompt"]
prompt = ChatPromptTemplate.from_template(prompt_template)

# 构建多模态 RAG 链
rag_chain = (
    RunnableParallel({
        "context": retriever | _format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)



# 执行问答（纯文本提问示例）
# question = "怎么查看文件格式，知道格式后用什么工具处理"

# # 带图像的提问示例（需将图片路径加入文档库）
question = "milvus适用场景"

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
inputs = tokenizer(question, return_tensors="pt")
print("Token数量:", inputs.input_ids.shape[1])  

# 检索与问题相关的文档
docs = retriever.get_relevant_documents(question)
for doc in docs:
    if doc.metadata.get("content_type") == "image":
        print(f"关联图片: {doc.metadata['source']}")


response = rag_chain.invoke(question)
print("\n" + "="*50)
print(f"问题：{question}")
print("="*50 + "\n")
print(response)

