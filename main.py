from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st


def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="COLAB IA")
    st.header("DexPara de Eventos")

    csv_file = st.file_uploader("Faça o upload do DexPara", type="csv")
    if csv_file is not None:

        agent = create_csv_agent(
            OpenAI(temperature=0), csv_file, verbose=True)

        user_question = st.text_input("Faça uma pergunta sobre o DexPara importado: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="Aguarde... Logo retorno"):
                st.write(agent.run(user_question))


if __name__ == "__main__":
    main()
