# 导入所需模块
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import OllamaLLM
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 加载本地文本文件（示例路径：data/sample.txt）
loader = TextLoader("doc/sample.txt", encoding="utf-8")
documents = loader.load()

# # 使用递归字符分割器
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50,
#     separators=["\n\n", "\n", "。", "！", "？", "；",":"]
# )
# splits = text_splitter.split_documents(documents)
# print(f"原始文档数: {len(documents)}")
# print(f"分割后文档数: {len(splits)}")  # 正确分割后数量应 >1


# 使用多模态嵌入模型（如 Llama 3.2 或 Bakllava）
embeddings = OllamaEmbeddings(model="llama3.2-vision")  # 支持图像与文本联合编码
# 生成跨模态向量
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_prod"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

print('retriever',retriever)

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 查询所有文档（示例）
docs = vector_store.get(include=["documents", "metadatas"])
print("文档内容:", docs["documents"])
print("元数据:", docs["metadatas"])

template = """
基于以下多模态上下文回答问题：
{context}

用户输入：{question}
请结合图文信息生成答案，若包含图表请用Markdown格式描述。
"""
prompt = ChatPromptTemplate.from_template(template)




# 配置多模态生成模型
llm = OllamaLLM(
    model="llama3.2-vision",  # 支持图文混合输入
    temperature=0.3,
    num_thread= 4
)

# 辅助函数定义（必须在链调用前）
def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 构建多模态 RAG 链
rag_chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 执行问答（纯文本提问示例）
question = "文档中提到的核心技术有哪些？"
response = rag_chain.invoke(question)
print("\n" + "="*50)
print(f"问题：{question}")
print("="*50 + "\n")
print(response)

# 带图像的提问示例（需将图片路径加入文档库）
# image_question = "分析这张图片中的技术原理"
# response = rag_chain.invoke({"question": image_question, "image": "data/tech_diagram.png"})