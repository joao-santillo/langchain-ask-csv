#from langchain_experimental.agents import create_csv_agent
#from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import csv
import streamlit as st
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_openai import ChatOpenAI

class PreProcessing:
    
    @staticmethod
    def load_openAIKey():
        if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
            print("OPENAI_API_KEY is not set")
            exit(1)
        else:
            print("OPENAI_API_KEY is set")
    
    @staticmethod
    def load_langSmithKey():
        if os.getenv("LANGCHAIN_API_KEY") is None or os.getenv("LANGCHAIN_API_KEY") == "":
            print("LANGCHAIN_API_KEY is not set")
            exit(1)
        else:
            print("LANGCHAIN_API_KEY is set")

    def set_streamlit(self,page_title,header):
        st.set_page_config(page_title)
        st.header(header)

    def start_llm(self):
        llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.8)
        return llm

    def load_context(_self, _url):
        loader = UnstructuredExcelLoader(_url)
        docs = loader.load()
        return docs

    @st.cache_resource
    def split_context(_self, _docs, _chunk_size):

        if _chunk_size is not None:
            splitter = RecursiveCharacterTextSplitter(separators= ["\n\n\n", "\n\n\n\n"],
                                        chunk_size=_chunk_size, 
                                        chunk_overlap=0)
        
            documents = splitter.split_documents(_docs)

        st.write("Foram importados " + str(len(documents)) + " eventos para o contexto da aplicação")

        return documents

    @st.cache_resource
    def embed_context(_self, _documents):
        # Embedding prior to VectorStore:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(_documents, embeddings, persist_directory="./chroma_db")
        return vectorstore


def main():
    load_dotenv()
    pp = PreProcessing()

    pp.load_langSmithKey()
    pp.load_openAIKey()
    pp.set_streamlit("COLAB IA 💰","COLAB AI - De x Para de Eventos 💰")
    llm = pp.start_llm()

    chunk_size = st.number_input("Informe o tamanho do chunk: ", min_value=120, max_value=200, 
                                    help = '''O chunk é um parâmetro de importação que carrega 
                                    cada linha do DexPara como um documento para o BD Vetorial. 
                                    Informar a quantidade média de caracteres de uma linha do DexPara 
                                    é suficiente para esse parâmetro''')
    
    docs = pp.load_context("planilha-de-x-para.xlsx")
    documents = pp.split_context(docs,chunk_size)
    vectorstore = pp.embed_context(documents)

    # Creating document chain:
    template = '''
    Objetivo da Conversa:
        Identificar o evento de folha de pagamento PARA apropriado com base no código do evento fornecido pelo usuário.

    Contexto Inicial:
        Você forneceu uma planilha Excel com as seguintes colunas:
        ["CODIGO_EVENTO_DE","DESCRICAO_EVENTO_DE", "TIPO_EVENTO_DE", "CODIGO_FOLHA_DE", "INSS_DE", "FGTS_DE","IRRF_DE", "MEDIAVEL_DE", "CODIGO_EVENTO_PARA","DESCRICAO_EVENTO_PARA","TIPO_EVENTO_PARA", "CODIGO_FOLHA_PARA","DETERMINACAO_PARA", "CODIGO_EVENTO_REFERENCIA","DESCRICAO_EVENTO_REFERENCIA", "INSS_PARA", "FGTS_PARA","IRRF_PARA", "MEDIAVEL_PARA"]
        {context}

    Pergunta a ser respondida:
        Baseado no código do evento fornecido pelo usuário, identifique o evento de folha de pagamento PARA apropriado para a migração de dados.
        {input}

    Instruções para o ChatGPT:
        Use a planilha fornecida como contexto para identificar o evento de folha de pagamento PARA apropriado com base no código do evento fornecido pelo usuário.
        Responda como uma tabela.

    Informações Importantes:
        O evento de folha de pagamento PARA deve ser identificado com base nas informações fornecidas nas colunas da planilha.
        Forneça uma resposta que seja relevante e apropriada para o código do evento fornecido pelo usuário. 
        Forneça apenas a tabela, não gere texto adicional.
    '''

    #prompt = ChatPromptTemplate.from_template(template)

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Considerando a conversa acima, gere uma busca para procurar informações relevantes para a conversa")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
   
    # Creating retrieval chain:
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    #chain = prompt | llm | output_parser
    #output_parser = CommaSeparatedListOutputParser()

    #representa o arquivo "DE" sem estar preenchido com o Para
    #csv_file = st.file_uploader("Importe um De", type="csv")
    csv_file = True

    if csv_file is not None:
        
        #agent = create_csv_agent(
            #OpenAI(temperature=0), csv_file, verbose=True)
        
        user_question = st.text_input("Informe qual evento DE: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="Aguarde... Logo retorno"):
                #st.write(agent.run(user_question))
                response = retrieval_chain.invoke({
                    "input":user_question
                })
                st.write(response['answer'])


if __name__ == "__main__":
    main()
