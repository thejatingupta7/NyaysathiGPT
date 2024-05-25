# Importing Libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import PromptTemplate, HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


# Faiss Index Path
FAISS_INDEX = "embed_db/"


# Custom prompt template
custom_prompt_template = """[INST] <<SYS>>
You are a trained bot to guide people about Indian Law. You will answer user's query with your knowledge and the context provided. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Do not say thank you and tell you are an AI Assistant and be open about everything.
<</SYS>>
Use the following pieces of context to answer the users question.
Context : {context}
Question : {question}
Answer : [/INST]
"""

# Set custom prompt template
def set_custom_prompt_template():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


# Load the LLM
def load_llm():
    repo_id = 'meta-llama/Meta-Llama-3-8B'
    model = AutoModelForCausalLM.from_pretrained(repo_id, device_map='auto', load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


# Create Retrieval QA chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


# Create QA pipeline
def qa_pipeline():
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local("vectorstore/", embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt_template()
    chain = retrieval_qa_chain(llm, qa_prompt, db)
    return chain
