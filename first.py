# 导入所需模块
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import OllamaLLM
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredImageLoader
import yaml  # 新增此行
from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnableParallel, RunnableLambda

from typing import List
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from PIL import Image
import os

# 标准化图像加载器
class SimpleImageLoader:
    def __init__(self, path: str):
        self.image_paths = [os.path.join(path, f) for f in os.listdir(path) 
                          if f.lower().endswith('.png')]

    def load(self) -> List[Document]:
        docs = []
        for img_path in self.image_paths:
            with Image.open(img_path) as img:
                docs.append(Document(
                    page_content=f"PNG图像: {os.path.basename(img_path)}",
                    metadata={
                        "source": img_path,
                        "width": img.width,
                        "height": img.height,
                        "content_type": "image"
                    }
                ))
        return docs
    
# 加载本地文本文件（示例路径：data/sample.txt）
text_loader = DirectoryLoader(
    path="doc/",          # 目标目录路径
    glob="**/*.{md,txt}",       # 递归匹配所有子目录的.txt文件
    loader_cls=TextLoader,  # 指定文本加载器
    loader_kwargs={
        "autodetect_encoding": True,
        "metadata_columns": ["source", "content_type"]  # 强制包含content_type
    },
    show_progress=True     # 显示进度条（需安装tqdm库）
)

image_loader = SimpleImageLoader("doc/img/")

documents = text_loader.load() + image_loader.load()
print('documents',documents)
# loader = TextLoader("doc/python.md")
# documents = loader.load()

# 使用递归字符分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；",":"]
)
splits = text_splitter.split_documents(documents)
print(f"原始文档数: {len(documents)}")
print(f"分割后文档数: {len(splits)}")  # 正确分割后数量应 >1


# 使用多模态嵌入模型（如 Llama 3.2 或 Bakllava）
embeddings = OllamaEmbeddings(model="llama3.2-vision")  # 支持图像与文本联合编码
# 生成跨模态向量
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 动态计算k值（至少1，不超过文档总数的1/3）
valid_doc_count = len(documents)
retriever = vectorstore.as_retriever(    search_kwargs={
        "k": max(5, max(1, valid_doc_count // 3))
    })

print('retriever',retriever)

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 查询所有文档（示例）
docs = vector_store.get(include=["documents", "metadatas"])
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
question = "python3环境相关的知识点"



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

