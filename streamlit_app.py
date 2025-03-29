import os
import json
import requests
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# -----------------------------
# CONFIGURATION
# -----------------------------
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

urls = {
    "How to Apply": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/how-to-apply/",
    "Tuition, Fees, & Aid": "https://datascience.uchicago.edu/education/tuition-fees-aid/",
    "Course Progressions": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/course-progressions/",
    "In-Person Program": "https://datascience.uchicago.edu/education/masters-programs/in-person-program/",
    "Online Program": "https://datascience.uchicago.edu/education/masters-programs/online-program/",
    "Our Students": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/our-students/",
    "Events & Deadlines": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/events-deadlines/",
    "FAQs": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faqs/",
    "Career Outcomes": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/career-outcomes/",
    "Master’s Programs Overview": "https://datascience.uchicago.edu/education/masters-programs/",
    "Faculty, Instructors, Staff": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/instructors-staff/"
}

# -----------------------------
# SCRAPING FUNCTIONS
# -----------------------------
def scrape_page_combined(url, title):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    main_content = soup.find(class_='main-content')
    for accordion in main_content.find_all(class_='list-accordion'):
        accordion.decompose()
    all_text = ": ".join([tag.get_text(strip=True) for tag in main_content.find_all(['h1', 'h2', 'h3', 'p', 'ul'])])
    return {"title": title, "url": url, "content": all_text}

def scrape_faqs(url):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    accordion_items = soup.find_all('li', class_='accordion__item')
    faqs = []
    for item in accordion_items:
        q_tag = item.find('a', class_='accordion-title')
        a_tag = item.find('div', class_='accordion-content') or item.find('p')
        question = q_tag.get_text(strip=True) if q_tag else 'No question found'
        answer = a_tag.get_text(strip=True) if a_tag else 'No answer found'
        faqs.append(f"Q: {question}\nA: {answer}")
    return {"title": "FAQs", "url": url, "content": "\n\n".join(faqs)}

def scrape_booth_page():
    url = "https://www.chicagobooth.edu/mba/joint-degree/mba-ms-applied-data-science"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    body_elements = soup.find_all(class_='body-copy')
    content = " ".join([tag.get_text(strip=True) for el in body_elements for tag in el.find_all(['h1', 'h2', 'h3', 'p'])])
    return {"title": "Booth Program", "url": url, "content": content}

def scrape_online_courses():
    url = urls["Online Program"]
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    course_data = []
    sections = soup.find_all('li', class_='accordion-item')
    for section in sections:
        section_title = section.find('a', class_='accordion-title').get_text(strip=True)
        course_items = section.find_all('li', class_='accordion__item')
        course_list = []
        for item in course_items:
            name = item.find('a', class_='accordion-title').get_text(strip=True)
            desc = item.find('div', class_='textblock').get_text(strip=True)
            course_list.append(f"{name}: {desc}")
        course_data.append({"title": section_title, "url": url, "content": " : ".join(course_list)})
    return course_data

# -----------------------------
# DATA GATHERING
# -----------------------------
all_data = [scrape_page_combined(url, title) for title, url in urls.items() if title != "FAQs"]
all_data.append(scrape_booth_page())
all_data.append(scrape_faqs(urls["FAQs"]))
all_data.extend(scrape_online_courses())

# Convert scraped data to LangChain documents
documents = [Document(page_content=entry["content"], metadata={"title": entry["title"], "source": entry["url"]}) for entry in all_data]

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("University of Chicago MSADS Chatbot")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if openai_api_key:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = [Document(page_content=chunk, metadata=doc.metadata) for doc in documents for chunk in splitter.split_text(doc.page_content)]
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt_template = PromptTemplate(
        input_variables=["query"],
        template="""
        You are a knowledgeable assistant with expertise in the University of Chicago MSADS Program.
        Answer the following question in a clear and informative manner:
        
        Question: {query}
        """
    )

    def query_with_selection(query):
        retrieved = retriever.get_relevant_documents(query)
        scores = [doc.metadata.get('similarity_score', 0) for doc in retrieved]
        threshold = np.percentile(scores, 75)
        selected = [doc for doc, score in zip(retrieved, scores) if score >= threshold]
        formatted_query = prompt_template.format(query=query)
        result = qa_chain({"query": formatted_query, "retrieved_documents": selected})
        return result["result"], result["source_documents"]

    gpt4 = ChatOpenAI(model="gpt-4-turbo", openai_api_key=openai_api_key, max_tokens=1000, temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(
        llm=gpt4,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    with st.form("chat_form"):
        query = st.text_area("Ask a question about the University of Chicago MSADS Program")
        submitted = st.form_submit_button("Submit")
        if submitted:
            answer, sources = query_with_selection(query)
            st.write("**Answer:**", answer)
            st.write("**Sources:**")
            for src in set(doc.metadata["source"] for doc in sources):
                st.write(src)
else:
    st.warning("Please enter your OpenAI API Key to proceed.")
