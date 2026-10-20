from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

TEMPLATE = "{input}"

sentiment_template = PromptTemplate(
    input_variables=["input"],
    template=TEMPLATE,
)

class DecoderResponse:
    def __init__(self, model_name):
        self.model_name = model_name
        self.llm = OllamaLLM(model=self.model_name, temperature=0.0)

    '''def set_trainer(self):
        texts = self.df_train['text'].tolist()
        documents = [Document(page_content=t, metadata={"index": i}) for i, t in enumerate(texts)]
        self.vectorstore = FAISS.from_documents(documents, self.embedding_model)'''

    def train(self, text):
        # Cria o pipeline HuggingFace
        sentiment_chain = sentiment_template | self.llm

        result_str = sentiment_chain.invoke({"input": text})
        if "deepseek" in self.model_name:
            think_close_tag_position = result_str.find("</think>")
            if think_close_tag_position != -1:
                result_str = result_str[think_close_tag_position+len("</think>"):]

        #print(f"Result: {result_str}")

        return result_str
       