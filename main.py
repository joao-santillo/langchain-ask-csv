#from langchain_experimental.agents import create_csv_agent
#from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import csv
import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI

def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable:
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # Streamlit page configs:
    st.set_page_config(page_title="COLAB IA 💰")
    st.header("COLAB AI - De x Para de Eventos 💰")

    # Initializing llm:
    llm = ChatOpenAI(model='gpt-4-turbo-preview')
    
    

    # Define the columns we want to embed vs which ones we want in metadata:
    columns_to_embed = ["CODIGO_FOLHA_DE","DESCRICAO_EVENTO_DE","TIPO_EVENTO_DE", "DESCRICAO_EVENTO_PARA", "TIPO_EVENTO_PARA", "DETERMINACAO_PARA", "DESCRICAO_EVENTO_REFERENCIA"]
    columns_to_metadata = ["CODIGO_EVENTO_DE","DESCRICAO_EVENTO_DE", "TIPO_EVENTO_DE", "CODIGO_FOLHA_DE", "INSS_DE", "FGTS_DE","IRRF_DE", "MEDIAVEL_DE", "CODIGO_EVENTO_PARA","DESCRICAO_EVENTO_PARA","TIPO_EVENTO_PARA", "CODIGO_FOLHA_PARA","DETERMINACAO_PARA", "CODIGO_EVENTO_REFERENCIA","DESCRICAO_EVENTO_REFERENCIA", "INSS_PARA", "FGTS_PARA","IRRF_PARA", "MEDIAVEL_PARA"]
    
    # Process the CSV into the embedable content vs the metadata and put it into Document format so that we can chunk it into pieces.
    """docs = []
    with open('Planilha_DexPara_Bravo.csv', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for i, row in enumerate(csv_reader):
            to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
            values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
            to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
            newDoc = Document(page_content=to_embed, metadata=to_metadata)
            docs.append(newDoc)"""
    
    # Loading context from csv file:
    loader = CSVLoader(file_path='C:\\Users\\joao.santillo\\OneDrive - LG INFORMATICA S.A\\COLAB IA\\Langchain\\langchain-ask-csv\\Planilha_DexPara_Bravo.csv')
    docs = loader.load()

    st.write(docs)

    # Splitting the document using Chracter splitting:
    
    splitter = CharacterTextSplitter(separator="\n",
                                chunk_size=500, 
                                chunk_overlap=0,
                                length_function=len)
    
    documents = splitter.split_documents(docs)

    # Embedding prior to VectorStore:
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Creating document chain:
    query = ""

    template = '''
    Você é um consultor de folha de pagamento, está em um projeto de migração de dados e ficou responsável pela migração da 
    ficha financeira histórica do fornecedor atual para o sistema da empresa de software que você trabalha. 
    A sua tarefa é para um certo evento DE de folha de pagamento fornecido, 
    definir um evento de folha de pagamento PARA com base no contexto fornecido. 
    Se não souber a resposta, informe que não encontrou um evento correspondente
    Responda a seguinte pergunta baseado somente no contexto fornecido: 
    
    <context>
    {context}
    </context>

    Question: {input}
    '''

    prompt = ChatPromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(llm, prompt)
   
    # Creating retrieval chain:
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    #chain = prompt | llm | output_parser
    #output_parser = CommaSeparatedListOutputParser()

    #representa o arquivo "DE" sem estar preenchido com o Para
    csv_file = st.file_uploader("Importe um De", type="csv")

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
