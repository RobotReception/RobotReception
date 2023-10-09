from fastapi import FastAPI

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from utils import *

app = FastAPI()

class ArabicChatBot:

    def __init__(self):
        self.api_key = "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"
        self.model_name = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613"]

        buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

        self.llm = ChatOpenAI(model_name=self.model_name[1], openai_api_key=self.api_key)

        system_msg_template = SystemMessagePromptTemplate.from_template(
            template="""أجب على السؤال بأكبر قدر ممكن من الصدق 
            باستخدام السياق المتوفر ، وأريد الإجابة كلها باللغة العربية وإذا لم تكن الإجابة موجودة في النص أدناه ،  " اجب بطريقة مخصرة من عندك" واذا تم سؤال من انت قول انا روبوت دردشة معتمد على بيانات سابقة تم تدريبي عليها  "'"""
        )

        human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

        prompt_template = ChatPromptTemplate.from_messages(
            [
                system_msg_template,
                MessagesPlaceholder(variable_name="history"),
                human_msg_template
            ]
        )

        self.conversation = ConversationChain(memory=buffer_memory, prompt=prompt_template, llm=self.llm)

    def get_response(self, query: str):
        context = find_match(query, self.model_name[0])
        response = self.conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")

        return response

    def run(self, query):
        response = self.get_response(query)
        return response

chatbot = ArabicChatBot()

@app.post("/get_response")
async def get_response(query: str):
    response = chatbot.run(query)
    print(response)
    return {"response": response}

