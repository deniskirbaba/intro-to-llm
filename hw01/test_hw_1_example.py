import warnings

warnings.filterwarnings("ignore")

import os

from dotenv import load_dotenv
from hw_1 import LlamaIndex, get_giga, get_prompt, get_prompt_few_shot
from langchain.chat_models.gigachat import GigaChat

load_dotenv(".env")
sb_auth_data = os.getenv("SB_AUTH_DATA")


def test_giga():
    giga = get_giga(sb_auth_data)
    assert giga is not None


def test_prompt():
    prompt = get_prompt("Hello!")
    assert len(prompt) == 2


def test_few_shot():
    giga = get_giga(sb_auth_data)

    number = "11223344"
    prompt = get_prompt_few_shot(number)
    res = giga.invoke(prompt)
    answer = res.content[res.content.rfind("Answer:") :]
    assert answer == "Answer: The number 11223344 consist of four even digits."


def test_llama_index():
    giga_pro = GigaChat(
        credentials=sb_auth_data,
        model="GigaChat-Pro",
        timeout=30,
        verify_ssl_certs=False,
        temperature=0.01,
    )
    llama_index = LlamaIndex("data/", giga_pro)
    res = llama_index.query("Who are the authors of Attention is all you need?")
    assert res != ""
