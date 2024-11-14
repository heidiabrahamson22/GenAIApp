import requests
from bs4 import BeautifulSoup
import json
from langchain.schema import Document
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


# URLs to scrape
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

# Function to scrape each page and combine all text
def scrape_page_combined(url,title):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    main_content = soup.find(class_='main-content')
    
    for accordion in main_content.find_all(class_='list-accordion'):
        accordion.decompose()  # Removes the element from the soup object

    all_text = ": ".join([tag.get_text(strip=True) for tag in main_content.find_all(['h1', 'h2', 'h3', 'p','ul'])])

    # Create a single dictionary entry with all text combined
    page_data = {
        "title": title,
        "url": url,
        "content": all_text
    }

        
    return page_data

# Collect all pages' data in a single list
all_data = []
for title, url in urls.items():
    print(f"Scraping: {title}")
    page_content = scrape_page_combined(url,title)
    all_data.append(page_content)

url = "https://www.chicagobooth.edu/mba/joint-degree/mba-ms-applied-data-science"
response = requests.get(url)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')

# Find all elements with class 'body-copy'
body_copy_elements = soup.find_all(class_='body-copy')

# Combine all text content from each of the 'body-copy' elements
all_text = " ".join(
    [tag.get_text(strip=True) for element in body_copy_elements for tag in element.find_all(['h1', 'h2', 'h3', 'p'])]
)
booth_data = {
    "title": "Booth Program",
    "url": url,
    "content": all_text
}

all_data.append(booth_data)

response = requests.get('https://datascience.uchicago.edu/education/masters-programs/online-program/')
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')

accordion_sections = soup.find_all('li', class_='accordion-item')

# Initialize dictionaries to hold the course data
course_data = []

# Loop through each accordion section
for section in accordion_sections:
    # Get the section title (Noncredit, Core, or Elective)
    section_title = section.find('a', class_='accordion-title').get_text(strip=True)

    # Get all the courses under this section
    course_items = section.find_all('li', class_='accordion__item')

    # Initialize a list to hold course data for this section
    course_list = []

    for course_item in course_items:
        # Get the course name and description
        course_name = course_item.find('a', class_='accordion-title').get_text(strip=True)
        course_description = course_item.find('div', class_='textblock').get_text(strip=True)

        # Combine the course name and description into one string
        content = course_name + ": " + course_description
        
        # Append the course content to the list
        course_list.append(content)
    
    # For each section, add the data to the list in the required format
    course_data.append({
        'title': section_title,  # The title of the section (e.g., Noncredit, Core, Elective)
        'url': 'https://datascience.uchicago.edu/education/masters-programs/online-program/',  # The URL
        'content': " : ".join(course_list)  # Combine all courses under this section with ":" separator
    })

# Print the results
for data in course_data:
    all_data.append(data)

# Convert JSON entries to LangChain Document objects
documents = []
for entry in all_data:
    doc = Document(
        page_content=entry["content"],
        metadata={"title": entry["title"], "source": entry["url"]}
    )
    documents.append(doc)

st.title("University of Chicago MSADS Chatbot")

# User input for OpenAI API Key
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if openai_api_key:
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = {}

    def add_documents_to_store(documents, embedding_model):
        for doc in documents:
            vector = embedding_model.embed_query(doc.page_content)
            vector_store[doc.metadata["title"]] = {"vector": vector, "content": doc.page_content}

    # Add documents to the vector store
    add_documents_to_store(documents, embedding_model)

    # Simple retrieval function to find the top k most similar documents
    def retrieve_similar_documents(query, k=3):
        query_vector = embedding_model.embed_query(query)
        similarities = []
        for title, data in vector_store.items():
            similarity = np.dot(query_vector, data["vector"])
            similarities.append((title, data["content"], similarity))
        sorted_docs = sorted(similarities, key=lambda x: x[2], reverse=True)[:k]
        return [doc[1] for doc in sorted_docs]

    gpt4 = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key, max_tokens=500)

    def generate_response(query):
        context = "\n\n".join(retrieve_similar_documents(query, k=3))
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        answer = gpt4(prompt)
        return answer

    # Streamlit UI for user input
    with st.form("my_form"):
        text = st.text_area("Ask a question about the University of Chicago MSADS Program!")
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            answer = generate_response(text)
            st.write("**Answer:**", answer)
else:
    st.warning("Please enter your OpenAI API Key to proceed.")
