# pip install --upgrade --quiet gigachain==0.2.6 gigachain_community==0.2.6 gigachain-cli==0.0.25 duckduckgo-search==6.2.4 pyfiglet==1.0.2 langchain-anthropic llama_index==0.9.40 pypdf==4.0.1 sentence_transformers==2.3.1

import getpass
from typing import List, Union

from langchain.chat_models.gigachat import GigaChat
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex

# # 1. GigaChat
# Define GigaChat throw langchain.chat_models


def get_giga(giga_key: str) -> GigaChat:
    giga = GigaChat(
        credentials=giga_key, model="GigaChat", timeout=30, verify_ssl_certs=False
    )
    giga.verbose = False
    return giga


def test_giga():
    giga_key = getpass.getpass("Enter your GigaChat credentials: ")
    giga = get_giga(giga_key)
    assert giga is not None


# # 2. Prompting
# ### 2.1 Define classic prompt


# Implement a function to build a classic prompt (with System and User parts)
def get_prompt(user_content: str) -> list[Union[SystemMessage, HumanMessage]]:
    messages = [
        SystemMessage(
            content="You are an intelligent assistant. Your goal is to answer questions as \
                accurately as possible based on the instructions and context provided."
        ),
        HumanMessage(content=user_content),
    ]
    return messages


# Let's check how it works
def test_prompt():
    giga_key = getpass.getpass("Enter your GigaChat credentials: ")
    giga = get_giga(giga_key)
    user_content = "Hello!"
    prompt = get_prompt(user_content)
    res = giga.invoke(prompt)
    print(res.content)


# ### 3. Define few-shot prompting


# Implement a function to build a few-shot prompt to count even digits in the given number.
# The answer should be in the format 'Answer: The number {number} consist of {text} even digits.',
# for example 'Answer: The number 11223344 consist of four even digits.'
def get_prompt_few_shot(number: str) -> List[HumanMessage]:
    examples = [
        (
            "123456",
            "Answer: The number 123456 consist of three even digits. (Digits: 2, 4, 6)",
        ),
        (
            "24680",
            "Answer: The number 24680 consist of five even digits. (Digits: 2, 4, 6, 8, 0)",
        ),
        ("13579", "Answer: The number 13579 consist of zero even digits."),
        (
            "11223344",
            "Answer: The number 11223344 consist of four even digits. (Digits: 2, 2, 4, 4)",
        ),
    ]

    prompt = [
        HumanMessage(
            content="Your task is to count the number of even digits in the given number. "
            "Even digits are: 0, 2, 4, 6, 8. To find the correct count, look at each digit "
            "one by one. If the digit is even (0, 2, 4, 6, 8), count it. "
            "Finally, replace '???' in the answer with the correct number of even digits, spelled out in words."
        )
    ]

    prompt.extend(
        [HumanMessage(content=f"Number: {num}\n{answer}") for num, answer in examples]
    )

    prompt.append(
        HumanMessage(
            content="Now, please count the number of even digits in the following number. "
            "Remember to count each even digit (0, 2, 4, 6, 8) and replace '???' with the correct number, "
            "written as words."
        )
    )

    prompt.append(
        HumanMessage(
            content=f"Answer: The number {number} consist of '???' even digits."
        )
    )

    return prompt


# Let's check how it works
def test_few_shot():
    giga_key = getpass.getpass("Enter your GigaChat credentials: ")
    giga = get_giga(giga_key)
    number = "62388712774"
    prompt = get_prompt_few_shot(number)
    res = giga.invoke(prompt)
    print(res.content)


# # 4. Llama_index
# Implement your own class to use llama_index. You need to implement some code to build
# llama_index across your own documents. For this task you should use GigaChat Pro.
class LlamaIndex:
    def __init__(self, path_to_data: str, llm: GigaChat):
        self.system_prompt = """
        You are a Q&A assistant. Your goal is to answer questions as
        accurately as possible based on the instructions and context provided.
        """
        self.embed_model = "sentence-transformers/all-mpnet-base-v2"

        print(f"Initializing LlamaIndex with data path: {path_to_data}.")
        try:
            documents = SimpleDirectoryReader(path_to_data).load_data()
        except Exception as e:
            raise ValueError(
                f"Error loading documents from path: {path_to_data}. Exception: {e}"
            )

        print(f"Initializing embedding model: {self.embed_model}.")
        try:
            embed_model = HuggingFaceEmbeddings(model_name=self.embed_model)
            service_context = ServiceContext.from_defaults(
                chunk_size=1024, llm=llm, embed_model=embed_model
            )
        except Exception as e:
            raise ValueError(f"Error initializing service context: {e}")

        print("Creating vector store index from documents.")
        try:
            self.index = VectorStoreIndex.from_documents(
                documents, service_context=service_context
            )
            self.query_engine = self.index.as_query_engine()
        except Exception as e:
            raise ValueError(f"Error creating index from documents: {e}")

    def query(self, user_prompt: str) -> str:
        user_input = self.system_prompt + "\n" + user_prompt

        try:
            response = self.query_engine.query(user_input)
            return response.response
        except Exception as e:
            raise ValueError(f"Error during query: {e}")


# Let's check
def test_llama_index():
    giga_key = getpass.getpass("Enter your GigaChat credentials: ")
    giga_pro = GigaChat(
        credentials=giga_key, model="GigaChat-Pro", timeout=30, verify_ssl_certs=False
    )

    llama_index = LlamaIndex("data/", giga_pro)
    res = llama_index.query("Who are the authors of Attention is all you need?")
    print(res)
