什么是LangChain
LangChain起源
LangChain六大主要领域
LangChain的主要价值组件
LangChain组件
数据连接组件data connection
data connection整体流程
data connection——文档加载器
data connection——文档转换
模型IO组件
Chain链组件
链的序列化，保存链到磁盘，从磁盘加载链：
总结
参考文献
什么是LangChain
LangChain: 一个让你的LLM变得更强大的开源框架。LangChain 就是一个 LLM 编程框架，你想开发一个基于 LLM 应用，需要什么组件它都有，直接使用就行；甚至针对常规的应用流程，它利用链(LangChain中Chain的由来)这个概念已经内置标准化方案了。下面我们从新兴的大语言模型（LLM）技术栈的角度来看看为何它的理念这么受欢迎。

LangChain起源
LangChain 的作者是 Harrison Chase，最初是于 2022 年 10 月开源的一个项目，在 GitHub 上获得大量关注之后迅速转变为一家初创公司。2017 年 Harrison Chase 还在哈佛上大学，如今已是硅谷的一家热门初创公司的 CEO，这对他来说是一次重大而迅速的跃迁。Insider 独家报道，人工智能初创公司 LangChain 在种子轮一周后，再次获得红杉领投的 2000 万至 2500 万美元融资，估值达到 2 亿美元。

LangChain六大主要领域
管理和优化prompt。不同的任务使用不同prompt，如何去管理和优化这些prompt是langchain的主要功能之一。
链，初步理解为一个具体任务中不同子任务之间的一个调用。
数据增强的生成，数据增强生成涉及特定类型的链，它首先与外部数据源交互以获取数据用于生成步骤。这方面的例子包括对长篇文字的总结和对特定数据源的提问/回答。
代理，根据不同的指令采取不同的行动，直到整个流程完成为止。
评估，生成式模型是出了名的难以用传统的指标来评估。评估它们的一个新方法是使用语言模型本身来进行评估。LangChain提供了一些提示/链来协助这个工作。
内存：在整个流程中帮我们管理一些中间状态。
总的来说LangChain可以理解为：在一个流程的整个生命周期中，管理和优化prompt，根据prompt使用不同的代理进行不同的动作，在这期间使用内存管理中间的一些状态，然后使用链将不同代理之间进行连接起来，最终形成一个闭环。

LangChain的主要价值组件
组件：用于处理语言模型的抽象概念，以及每个抽象概念的实现集合。无论你是否使用LangChain框架的其他部分，组件都是模块化的，易于使用。
现成的链：用于完成特定高级任务的组件的结构化组合。现成的链使人容易上手。对于更复杂的应用和细微的用例，组件使得定制现有链或建立新链变得容易。
LangChain组件
model I/O：语言模型接口
data connection：与特定任务的数据接口
chains：构建调用序列
agents：给定高级指令，让链选择使用哪些工具
memory：在一个链的运行之间保持应用状态
callbacks：记录并流式传输任何链的中间步骤
indexes：索引指的是结构化文件的方法，以便LLM能够与它们进行最好的交互
数据连接组件data connection
LLM应用需要用户特定的数据，这些数据不属于模型的训练集。LangChain通过以下方式提供了加载、转换、存储和查询数据的构建模块：

文档加载器：从许多不同的来源加载文档
文档转换器：分割文档，删除多余的文档等
文本嵌入模型：采取非结构化文本，并把它变成一个浮点数的列表 矢量存储：存储和搜索嵌入式数据
检索器：查询你的数据
data connection整体流程

data connection——文档加载器
python安装包命令：

pip install langchain
pip install unstructured
pip install jq
CSV基本用法

import os
from pathlib import Path

from langchain.document_loaders import UnstructuredCSVLoader
from langchain.document_loaders.csv_loader import CSVLoader
EXAMPLE_DIRECTORY = file_path = Path(__file__).parent.parent / "examples"


def test_unstructured_csv_loader() -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DIRECTORY, "stanley-cups.csv")
    loader = UnstructuredCSVLoader(str(file_path))
    docs = loader.load()
    print(docs)
    assert len(docs) == 1

def test_csv_loader():
  file_path = os.path.join(EXAMPLE_DIRECTORY, "stanley-cups.csv")
  loader = CSVLoader(file_path)
  docs = loader.load()
  print(docs)

test_unstructured_csv_loader()
test_csv_loader()
文件目录用法

from langchain.document_loaders import DirectoryLoader, TextLoader

text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader('../examples/', 
              glob="**/*.txt",  # 遍历txt文件
              show_progress=True,  # 显示进度
              use_multithreading=True,  # 使用多线程
              loader_cls=TextLoader,  # 使用加载数据的方式
              silent_errors=True,  # 遇到错误继续
              loader_kwargs=text_loader_kwargs)  # 可以使用字典传入参数

docs = loader.load()
print("\n")
print(docs[0])
HTML用法

from langchain.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
loader = UnstructuredHTMLLoader("../examples/example.html")
docs = loader.load()
print(docs[0])

loader = BSHTMLLoader("../examples/example.html")
docs = loader.load()
print(docs[0])
JSON用法

import json
from pathlib import Path
from pprint import pprint


file_path='../examples/facebook_chat.json'
data = json.loads(Path(file_path).read_text())
pprint(data)

"""
{'image': {'creation_timestamp': 1675549016, 'uri': 'image_of_the_chat.jpg'},
 'is_still_participant': True,
 'joinable_mode': {'link': '', 'mode': 1},
 'magic_words': [],
 'messages': [{'content': 'Bye!',
               'sender_name': 'User 2',
               'timestamp_ms': 1675597571851},
              {'content': 'Oh no worries! Bye',
               'sender_name': 'User 1',
               'timestamp_ms': 1675597435669},
              {'content': 'No Im sorry it was my mistake, the blue one is not '
                          'for sale',
               'sender_name': 'User 2',
               'timestamp_ms': 1675596277579},
              {'content': 'I thought you were selling the blue one!',
               'sender_name': 'User 1',
               'timestamp_ms': 1675595140251},
              {'content': 'Im not interested in this bag. Im interested in the '
                          'blue one!',
               'sender_name': 'User 1',
               'timestamp_ms': 1675595109305},
              {'content': 'Here is $129',
               'sender_name': 'User 2',
               'timestamp_ms': 1675595068468},
              {'photos': [{'creation_timestamp': 1675595059,
                           'uri': 'url_of_some_picture.jpg'}],
               'sender_name': 'User 2',
               'timestamp_ms': 1675595060730},
              {'content': 'Online is at least $100',
               'sender_name': 'User 2',
               'timestamp_ms': 1675595045152},
              {'content': 'How much do you want?',
               'sender_name': 'User 1',
               'timestamp_ms': 1675594799696},
              {'content': 'Goodmorning! $50 is too low.',
               'sender_name': 'User 2',
               'timestamp_ms': 1675577876645},
              {'content': 'Hi! Im interested in your bag. Im offering $50. Let '
                          'me know if you are interested. Thanks!',
               'sender_name': 'User 1',
               'timestamp_ms': 1675549022673}],
 'participants': [{'name': 'User 1'}, {'name': 'User 2'}],
 'thread_path': 'inbox/User 1 and User 2 chat',
 'title': 'User 1 and User 2 chat'}
"""
使用langchain加载数据：
```python
from langchain.document_loaders import JSONLoader
loader = JSONLoader(
    file_path='../examples/facebook_chat.json',
    jq_schema='.messages[].content' # 会报错Expected page_content is string, got <class 'NoneType'> instead.
    page_content=False, # 报错后添加这一行)

data = loader.load()
print(data[0])
PDF用法

'''
第一种用法
'''
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("../examples/layout-parser-paper.pdf")
pages = loader.load_and_split()

print(pages[0])

'''
第二种用法
'''
from langchain.document_loaders import MathpixPDFLoader

loader = MathpixPDFLoader("example_data/layout-parser-paper.pdf")

data = loader.load()
print(data[0])

'''
第三种用法
'''
from langchain.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader("../examples/layout-parser-paper.pdf")

data = loader.load()
print(data[0])
data connection——文档转换
加载了文件后，经常会需要转换它们以更好地适应应用。最简单的例子是，你可能想把一个长的文档分割成较小的块状，以适应你的模型的上下文窗口。LangChain有许多内置的文档转换工具，可以很容易地对文档进行分割、组合、过滤和其他操作。

通过字符进行文本分割

state_of_the_union = """
斗之力，三段！”

    望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…

    “萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来…

    中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。

    “三段？嘿嘿，果然不出我所料，这个“天才”这一年又是在原地踏步！”

    “哎，这废物真是把家族的脸都给丢光了。”

    “要不是族长是他的父亲，这种废物，早就被驱赶出家族，任其自生自灭了，哪还有机会待在家族中白吃白喝。”

    “唉，昔年那名闻乌坦城的天才少年，如今怎么落魄成这般模样了啊？”
"""

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 128,  # 分块长度
    chunk_overlap  = 10,  # 重合的文本长度
    length_function = len,
)

texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])

# 这里metadatas用于区分不同的文档
metadatas = [{"document": 1}, {"document": 2}]
documents = text_splitter.create_documents([state_of_the_union, state_of_the_union], metadatas=metadatas)
pprint(documents)

# 获取切割后的文本
print(text_splitter.split_text(state_of_the_union)[0])
对代码进行分割

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

print([e.value for e in Language])  # 支持语言
print(RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON))  # 分割符号

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
python_docs

"""
[Document(page_content='def hello_world():\n    print("Hello, World!")', metadata={}),
 Document(page_content='# Call the function\nhello_world()', metadata={})]
"""
通过markdownheader进行分割

举个例子：md = # Foo\n\n ## Bar\n\nHi this is Jim \nHi this is Joe\n\n ## Baz\n\n Hi this is Molly' .我们定义分割头：[("#", "Header 1"),("##", "Header 2")]文本应该被公共头进行分割，最终得到：`{'content': 'Hi this is Jim \nHi this is Joe', 'metadata': {'Header 1': 'Foo', 'Header 2': 'Bar'}} {'content': 'Hi this is Molly', 'metadata': {'Header 1': 'Foo', 'Header 2': 'Baz'}}

from langchain.text_splitter import MarkdownHeaderTextSplitter
markdown_document = "# Foo\n\n    ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly"

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)
md_header_splits

"""
    [Document(page_content='Hi this is Jim  \nHi this is Joe', metadata={'Header 1': 'Foo', 'Header 2': 'Bar'}),
     Document(page_content='Hi this is Lance', metadata={'Header 1': 'Foo', 'Header 2': 'Bar', 'Header 3': 'Boo'}),
     Document(page_content='Hi this is Molly', metadata={'Header 1': 'Foo', 'Header 2': 'Baz'})]
"""
通过字符递归分割

默认列表为：["\n\n", "\n", " ", ""]

 state_of_the_union = """
斗之力，三段！”

望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…

“萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来…

中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。

“三段？嘿嘿，果然不出我所料，这个“天才”这一年又是在原地踏步！”

“哎，这废物真是把家族的脸都给丢光了。”

“要不是族长是他的父亲，这种废物，早就被驱赶出家族，任其自生自灭了，哪还有机会待在家族中白吃白喝。”

“唉，昔年那名闻乌坦城的天才少年，如今怎么落魄成这般模样了啊？”
"""

from langchain.text_splitter import CharacterTextSplitter
from pprint import pprint
text_splitter = CharacterTextSplitter(
    chunk_size = 128,
    chunk_overlap  = 10,
    length_function = len,
)

texts = text_splitter.create_documents([state_of_the_union])
pprint(texts[0].page_content)

"""
('斗之力，三段！”\n'
 '\n'
 '望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…')
 
['斗之力，三段！”\n'
 '\n'
 '望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…',
 '“萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来…\n'
 '\n'
 '中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。']
"""
通过tokens进行分割

我们知道，语言模型有一个令牌限制。不应该超过令牌的限制。因此，当你把你的文本分成几块时，计算标记的数量是一个好主意。有许多标记器。当你计算文本中的令牌时，你应该使用与语言模型中使用的相同的令牌器。

安装tiktoken安装包

pip install tiktoken
python代码实现如下：

state_of_the_union = """
斗之力，三段！”

望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…

“萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来…

中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。

“三段？嘿嘿，果然不出我所料，这个“天才”这一年又是在原地踏步！”

“哎，这废物真是把家族的脸都给丢光了。”

“要不是族长是他的父亲，这种废物，早就被驱赶出家族，任其自生自灭了，哪还有机会待在家族中白吃白喝。”

“唉，昔年那名闻乌坦城的天才少年，如今怎么落魄成这般模样了啊？”
"""

from langchain.text_splitter import CharacterTextSplitter
from pprint import pprint
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=128, chunk_overlap=10
)
texts = text_splitter.split_text(state_of_the_union)

pprint(texts)
for text in texts:
  print(len(text))
"""
WARNING:langchain.text_splitter:Created a chunk of size 184, which is longer than the specified 128
['斗之力，三段！”',
 '望着测验魔石碑上面闪亮得甚至有些刺眼的五个大字，少年面无表情，唇角有着一抹自嘲，紧握的手掌，因为大力，而导致略微尖锐的指甲深深的刺进了掌心之中，带来一阵阵钻心的疼痛…',
 '“萧炎，斗之力，三段！级别：低级！”测验魔石碑之旁，一位中年男子，看了一眼碑上所显示出来的信息，语气漠然的将之公布了出来…',
 '中年男子话刚刚脱口，便是不出意外的在人头汹涌的广场上带起了一阵嘲讽的骚动。',
 '“三段？嘿嘿，果然不出我所料，这个“天才”这一年又是在原地踏步！”\n\n“哎，这废物真是把家族的脸都给丢光了。”',
 '“要不是族长是他的父亲，这种废物，早就被驱赶出家族，任其自生自灭了，哪还有机会待在家族中白吃白喝。”',
 '“唉，昔年那名闻乌坦城的天才少年，如今怎么落魄成这般模样了啊？”']

"""

'''
直接使用tiktoken
'''
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=0)

texts = text_splitter.split_text(state_of_the_union)
print(texts[0])
模型IO组件
模型包括LLM、聊天模型、文本嵌入模型。大型语言模型（LLM）是我们涵盖的第一种模型类型。这些模型接受一个文本字符串作为输入，并返回一个文本字符串作为输出。聊天模型是我们涵盖的第二种类型的模型。这些模型通常由一个语言模型支持，但它们的API更加结构化。具体来说，这些模型接受一个聊天信息的列表作为输入，并返回一个聊天信息。第三类模型是文本嵌入模型。这些模型将文本作为输入，并返回一个浮点数列表。

LangChain提供了连接任何语言模型的构建模块。

提示： 模板化、动态选择和管理模型输入
对模型进行编程的新方法是通过提示语。一个提示指的是对模型的输入。这种输入通常是由多个组件构成的。LangChain提供了几个类和函数，使构建和处理提示信息变得容易。常用的方法是：提示模板： 对模型输入进行参数化处；与例子选择器： 动态地选择要包含在提示中的例子。

一个提示模板可以包含：对语言模型的指示；一组少量的例子，以帮助语言模型产生一个更好的反应；一个对语言模型的问题。例如:

from langchain import PromptTemplate


template = """/
You are a naming consultant for new companies.
What is a good name for a company that makes {product}?
"""

prompt = PromptTemplate.from_template(template)
prompt.format(product="colorful socks")

"""
    You are a naming consultant for new companies.
    What is a good name for a company that makes colorful socks?
"""
如果需要创建一个与角色相关的消息模板，则需要使用MessagePromptTemplate。

template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
MessagePromptTemplate的类型：LangChain提供不同类型的MessagePromptTemplate。最常用的是AIMessagePromptTemplate、SystemMessagePromptTemplate和HumanMessagePromptTemplate，它们分别创建AI消息、系统消息和人类消息。

通用的提示模板：假设我们想让LLM生成一个函数名称的英文解释。为了实现这一任务，我们将创建一个自定义的提示模板，将函数名称作为输入，并对提示模板进行格式化，以提供该函数的源代码。

我们首先创建一个函数，它将返回给定的函数的源代码。

import inspect


def get_source_code(function_name):
    # Get the source code of the function
    return inspect.getsource(function_name)
接下来，我们将创建一个自定义的提示模板，将函数名称作为输入，并格式化提示模板以提供函数的源代码。


链允许我们将多个组件结合在一起，创建一个单一的、连贯的应用程序。例如，我们可以创建一个链，接受用户输入，用PromptTemplate格式化，然后将格式化的响应传递给LLM。我们可以通过将多个链组合在一起，或将链与其他组件组合在一起，建立更复杂的链。LLMChain是最基本的构建块链。它接受一个提示模板，用用户输入的格式化它，并从LLM返回响应。

我们首先要创建一个提示模板：

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
然后，创建一个非常简单的链，它将接受用户的输入，用它来格式化提示，然后将其发送到LLM。

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("colorful socks"))

    Colorful Toes Co.

'''
如果有多个变量，可以用一个字典一次输入它们。
'''
prompt = PromptTemplate(
    input_variables=["company", "product"],
    template="What is a good name for {company} that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run({
    'company': "ABC Startup",
    'product': "colorful socks"
    }))

    Socktopia Colourful Creations.
也可以在LLMChain中使用一个聊天模型：

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="What is a good name for a company that makes {product}?",
            input_variables=["product"],
        )
    )
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
chat = ChatOpenAI(temperature=0.9)
chain = LLMChain(llm=chat, prompt=chat_prompt_template)
print(chain.run("colorful socks"))

    Rainbow Socks Co.
链的序列化，保存链到磁盘，从磁盘加载链：
Serialization | ️ Langchain

总结
LangChain 为特定用例提供了多种组件，例如个人助理、文档问答、聊天机器人、查询表格数据、与 API 交互、提取、评估和汇总。通过提供模块化和灵活的方法简化了构建高级语言模型应用程序的过程。通过了解组件、链、提示模板、输出解析器、索引、检索器、聊天消息历史记录和代理等核心概念，可以创建适合特定需求的自定义解决方案。LangChain 能够释放语言模型的全部潜力，并在广泛的用例中创建智能的、上下文感知的应用程序。

n




# RAG 技术实践指南

## 核心理念

**RAG 不是 不是替换，而是扩充**：RAG不是用检索到的信息"代替"原始问题，而是将它作为补充材料，和原始问题一同提供给模型。

## 文档加载

### 预处理器选择指南

| 文件类型 | 推荐 loader | 先切先切标题？ | 再切长度 | Overlap | 额外配置 |
|---------|-------------|-----------|----------|---------|----------|
| 中文 PDF | PyPDFium2Loader | 否 | 500 字 | 100 | separators 带句号 |
| Markdown 博客 | MDHeaderSplitter | 是 | 1000 字 | 200 | 顺序不可逆 |
| Python 源码 | LanguageSplitter | 否 | 600 token | 60 | language="python" |
| 扫描合同 | AmazonTextractPDFLoader | 否 | 400 字 | 80 | OCR 后加空格正则 |
| Excel 报表 | pandas 自建 | 否 | 300 字 | 50 | metadata 带 sheet 名 |

### 关键要点

> **⚠️ 重要提醒**
> 
> - **loader 选错 = 后续全白搭白搭**
> - **OCR 钱别省，省了就幻觉**
> - **预处理/清洗**：清理不必要的字符、格式化代码、纠正拼写错误等，以提高文本质量
> - splitter 调参别凭感觉，用 tiktoken 数 token，再跑一遍检索评测脚本，看召回率涨没涨
> - 任何切分改完，重新建向量库，旧库不会自动更新！

## 文档分割

### Chunk Size 配置表

| 任务类型 | 推荐 Size | 推荐 Overlap (η) | 设计原理 |
|---------|-----------|-----------------|----------|
| 客服 FAQ | 400-600 字 | 20% | 答案通常一段，跨块少 |
| 法律合同 | 800-1200 字 | 15-20% | 条款需完整，overlap 保"第 3 条"不被切 |
| 技术 API 文档 | 300-500 字 | 25% | 函数 函数名+示例代码常跨 2 段 |
| 论文 | 600-800 字 | 20% | 兼顾图表标题与正文 |
| 对话日志 | 150-250 字 | 30% | 每句短，高 overlap 防"上下文失忆" |

**计算公式**: `overlap = η × chunk_size`

## 向量化（嵌入）

### 基本概念

**Embedding** 就是把"任意符号序列"→ 高维浮点向量，使得 **语义相似 ≈ 欧式距离近**，后面所有检索、聚类、分类、可视化都靠这一锤子。

### 模型选型决策矩阵

| 决策因素 | 选项 | 影响说明 |
|---------|------|----------|
| **① 语种** | 纯中文 / 中英混合 / 多语言 | 决定是否用双语或多语模型 |
| **② 长度** | 平均 128 token vs 512 token | 短文本选 128 维度低省钱；长文档需 512 以上 |
| **③ 维度** | 384 / 768 / 1024 / 1536 | 维度↑精度↑但向量库内存↑<br>1536 维 100 M 条 ≈ 600 GB 原始 RAM |
| **④ 是否微调** | 通用 vs 领域微调 | 医疗、法律术语必须微调，否则检索 TOP10 一半飘 |

### 处理流程

1. **Embedding 模型** → 把文本/图像变成向量
2. **向量库** → 接收 (向量, 原文,原文, 元数据) 三元组，建索引  
3. **查询时** → 向量库把用户问题的向量做近似最近邻搜索，返回 Top-K 原文

### "一" vs "多"关系

- **"一"个索引**：指知识库中的一个文档、一个章节或一个完整的答案，通常对应唯一ID（如 `doc_id_123`）
- **"多"个向量条目**：为表示这个"索引"而生成的多个向量，通过对长文本切分的每个"块"进行嵌入得到

## 检索策略

### 两阶段检索对比

| 特性 | 初步检索 | 重排 |
|------|----------|------|
| **目标** | 高召回率：找出所有可能相关的文档 | 高精度：确保最相关的文档排在最前 |
| **范围** | 整个文档库（百万/千万级） | 初步检索出的候选集（百/千级） |
| **模型** | 双编码器、BM25（快速） | 交叉编码器、列表式损失模型（慢速但精准） |
| **角色** | 撒网 | 精选 |







## 常见问题及解决方案

### 如何在 RAG 系统中缓解"复读机"问题？

#### 🎯 调整生成参数

- **适当提高温度**  
  例如从 0.1 提高到 0.3 或 0.5，引入一些创造性。

- **启用并加强重复惩罚**  
  确保 `repetition_penalty` 或类似参数被开启并设置为一个有效的值。

- **调整 Top-p**  
  尝试使用更大的 `top_p` 值以扩大候选词池。

#### 🔍 优化检索质量

- **确保检索质量**  
  保证检索到的文档块是**相关、高质量且信息充足**的。

- **采用先进检索技术**  
  如果一个问题需要多角度信息，尝试使用 **Multi-Query** 或 **HyDE** 技术来获得更全面的上下文。

#### 📝 改进 Prompt 工程

- **加入明确指令**  
  在 Prompt 中加入明确的指令，如：  
  > "请用流畅、多样化的语言回答，避免重复相同的词汇和句式。"

- **设定生成长度限制**  
  在代码层面强制限制单次回复的最大长度，防止循环无限进行下去。

#### 💎 总结

**"复读机"现象**是 LLM 在自回归生成过程中陷入的一种"局部最优"陷阱，是模型算法、参数设置和输入上下文共同作用下的失败表现。理解和解决它需要对 LLM 的工作原理有深入的了解。



### RAG 索引更新触发条件

#### 🚨 硬性触发条件

以下是必须立即启动索引更新的情况：

- **📄 文档数量阈值**  
  新增官方文档超过 **50页** 或删除重要文档

- **🔄 业务流程变更**  
  关键业务流程发生重大变更，影响现有文档的有效性

- **👥 用户反馈指向性**  
  收到多次用户反馈回答不准，且明确涉及特定文档内容

- **✏️ 文档修改规模**  
  文档修改量超过知识库总量的 **10%**

- **🎯 预设业务事件**  
  预先计划的业务活动，如：
  - 产品发布会
  - 法规生效日
  - 系统升级节点

#### ⚠️ 软性触发条件

以下情况建议尽快安排索引更新：

- **📊 检索效果下滑**  
  检索效果评估分数下降 **15%** 以上

- **🕳️ 知识缺口暴露**  
  发现明显的知识缺口或文档中存在错误信息

#### 💰 成本与风险评估

| 更新频率 | 优势 | 风险 | 适用场景 |
|---------|------|------|----------|
| **实时** | 信息零延迟 | • 计算成本极高<br>• 系统复杂度高 | 金融交易、紧急告警 |
| **小时级** | 较好的时效性 | • 中等资源消耗<br>• 需要监控队列 | 新闻聚合、市场分析 |
| **天级** | 平衡成本与收益 | • 可能存在一天的数据延迟 | 大多数企业内部知识库 |
| **周级** | 成本效益最佳 | • 最长一周延迟<br>• 批量处理压力大 | 标准企业知识管理 |
| **月级** | 维护成本最低 | • 信息严重滞后<br>• 用户体验差 | 归档文档、历史资料 |

#### 🔧 实操 Checklist

- ** 更新前的检查清单

- [ ] **备份现有索引** - 防止更新失败无法回滚
- [ ] **验证数据质量** - 避免垃圾数据入库
- [ ] **预估资源需求** - 防止更新过程中资源耗尽
| [ ] **安排维护窗口** - 告知用户可能的服务影响

- ** 更新后的验证清单

- [ ] **抽样测试检索** - 确认新旧文档都能被正确查到
- [ ] **比对回答质量** - 确保更新后答案未退化
- [ ] **更新版本标签** - 记录此次更新的内容和时间



### SelfQuery 在 RAG 系统中的定位与实践

#### ❓ 核心问题：只用 SelfQuery 够吗？

**简短回答**：不够。SelfQuery 只是一个**强有力的补充工具**，而非万能解决方案。

---

#### 🔄 SelfQuery 的工作边界

- SelfQuery 擅长处理的查询类型
| 查询类型 | 示例 | 转换结果 |
|---------|------|----------|
| **明确的时间范围** | “去年的销售报告” | `filter={"year": 2023}` |
| **特定的属性条件** | “技术部的项目文档” | `filter={"department": "技术部"}` |
| **组合过滤条件** | “2023年技术部的项目文档” | `filter={"year": 2023, "department": "技术部"}` |

- SelfQuery 无法处理的查询类型
| 查询类型 | 示例 | 原因分析 |
|---------|------|----------|
| **纯语义查询** | “介绍一下机器学习的基本机器学习的基本概念” | 无明确元数据条件 |
| **复杂推理需求** | “比较一下我们公司今年和去年的营收状况” | 需要整合多个文档信息并进行推理 |
| **隐含知识挖掘** | “我们的产品在哪些方面比竞争对手有优势？” | 需要从多个文档中提炼和总结 |

---

## 🎯 现实世界的查询分布

### 典型企业场景查询比例估算
```mermaid
pie title RAG 查询类型分布
    "纯语义查询" : 40
    "混合查询" : 35
    "纯过滤查询" : 25

# 关键技术要点解析

## 图检索

### 本质
图检索的本质是关系推理。

### 关键洞察
当你需要回答的不只是"这是什么"，而是"这有什么关系"时，图检索就是你的首选武器。

### 未来趋势
向量检索 + 图检索 的混合模式正在成为处理复杂知识的金标准。

---

## 重排窗口限制

### 🎯 核心定义
**重排窗口限制** 指的是在执行重排步骤时，最多只对初步检索结果中的前 M 个文档进行精细排序，而不是对所有检索到的 N 个文档进行重排。

### 推荐配置表格

| 系统规模 | 检索窗口 N | 重排窗口 M | 最终输出 K | 理由 |
|----------|------------|-------------|-------------|------|
| 实验原型 | 50 | 10 | 3 | 快速迭代验证 |
| 中小型生产 | 100 | 20 | 5 | 兼顾效果与响应时间 |
| 大型企业级 | 200 | 50 | 8 | 追求最高精度，资源充足 |
| 超大规模 | 500 | 100 | 10 | 处理极端复杂的查询 |

### 重要原则
永远不要把初步检索得到的全部 N 个文档都送去重排。

### 主要原因分析

#### 1. 性能考量（最主要原因）

| 模型类型 | 处理速度 | 原因 |
|----------|----------|------|
| 初步检索模型（Bi-Encoder） | 极快 | 文档向量已预先计算好，只需一次相似度计算 |
| 重排模型（Cross-Encoder） | 极慢 | 需要在推理时为每对（查询，文档）计算复杂的交叉注意力 |

#### 2. 成本效益平衡
- **经济成本**：重排模型的 API 调用费用通常远高于初步检索。
- **时间成本**：用户无法忍受数十秒的等待时间。

---

## 上下文拼接顺序

### 概述
上下文拼接顺序：优化 RAG 性能的关键一步

### 动态排序策略

#### 基于查询类型的智能排序

| 查询类型 | 推荐顺序 | 理由 |
|----------|-----------|------|
| 事实性查询"什么是X？" | 按相关性降序 | 让模型优先看到最强证据 |
| 时序性查询"Y产品的演变历程" | 按时间正序 | 符合事物发展的逻辑 |
| 因果关系"为什么会发生Z？" | 先背景/原因，后结果 | 帮助模型建立逻辑链条 |

---

## Stop 列表必要性分析

### 1. 在传统关键词检索中的"必须性"
在这些系统中，停止列表通常是必须的。原因如下：

- **降低噪声，突出核心**：移除"的"、"是"、"在"等高频虚词，能让评分算法更专注于有实质含义的名词、动词等。

### 2. 在现代向量检索/RAG中的"必要性"演变
随着技术转向语义搜索，**停止列表从一个"必需品"变成了一个"优化项"。

### 停止列表中通常包含的词类

| 类别 | 中文示例 | 英文示例 |
|-------|----------|----------|
| 冠词 | "一个"、"这个" | "a", "an", "the" |
| 介词 | "在…里"、"关于…" | "in", "on", "about" |
| 代词 | "我"、"你"、"它" | "I", "you", "it" |
| 连词 | "和"、"但是" | "and", "but" |
| 助动词 | "正在"、"已经" | "is", "was", "have" |

### 👍 使用停止列表的好处
1. **减少噪音**：提高搜索结果的质量。
2. **节省存储空间和内存**。
3. **提高处理速度**。
4. **增强语义焦点**。