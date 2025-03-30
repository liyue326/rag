# 导入所需模块
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# 加载本地文本文件（示例路径：data/sample.txt）
loader = TextLoader("data/sample.txt", encoding="utf-8")
documents = loader.load()
print(documents)