import requests
from bs4 import BeautifulSoup
import json
from langchain.schema import Document
import os
from langchain.vectorstores import FAISS
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

def scrape_faqs(url):
    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all accordion items
    accordion_items = soup.find_all('li', class_='accordion__item')

    # Initialize a list to hold the FAQs
    faqs = []

    # Iterate over each accordion item
    for item in accordion_items:
        # Extract the question
        question_tag = item.find('a', class_='accordion-title')
        question = question_tag.get_text(strip=True) if question_tag else 'No question found'

        # Extract the answer from <div> or <p> tags
        answer_tag = item.find('div', class_='accordion-content') or item.find('p')
        answer = answer_tag.get_text(strip=True) if answer_tag else 'No answer found'

        # Append the question and answer to the FAQs list
        faqs.append(f"Q: {question}\nA: {answer}")

    # Combine all FAQs into a single string
    all_faqs = "\n\n".join(faqs)

    # Create a dictionary with the scraped data
    faq_data = {
        "title": "FAQs",
        "url": url,
        "content": all_faqs
    }

    return faq_data

# URL of the FAQs page
faq_url = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faqs/"

# Scrape the FAQs
faq_content = scrape_faqs(faq_url)

all_data.append(faq_content)

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

# Check if the API key is provided before proceeding
if openai_api_key:
    # Initialize embedding model with the provided API key
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Use embedding model with FAISS
    vector_store = FAISS.from_documents(documents, embedding_model)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    # Split each document's content into chunks
    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_doc = Document(
                page_content=chunk,
                metadata=doc.metadata
            )
            chunked_documents.append(chunked_doc)

    # Set the retriever to return fewer chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    # Step 7: Function to query with similarity score filtering
    def query_with_improved_selection(query):
        # Retrieve documents based on query
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # Assume similarity scores are available; if not, FAISS can provide distance metrics.
        # Filter chunks by the top 25% based on similarity score or distance metric.
        scores = [doc.metadata.get('similarity_score', 0) for doc in retrieved_docs]
        threshold = np.percentile(scores, 75)  # Keep top 25% by score
        selected_docs = [doc for doc, score in zip(retrieved_docs, scores) if score >= threshold]
        
        # Run the QA chain with filtered documents
        result = qa_chain({"query": query, "retrieved_documents": selected_docs})
        return result["result"], result["source_documents"]
    
    # Initialize the ChatOpenAI model with the provided API key
    gpt4 = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key, max_tokens=500)
    qa_chain = RetrievalQA.from_chain_type(
        llm=gpt4,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Define the response generation function
    def generate_response(input_text):
        # Call query_with_improved_selection instead of the default QA retrieval
        answer, source_documents = query_with_improved_selection(input_text)
        return answer, source_documents
    
    # Form for user input in Streamlit
    with st.form("my_form"):
        text = st.text_area("Ask a question about the University of Chicago MSADS Program!")
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            # Generate response and sources using the improved selection function
            answer, source_documents = generate_response(text)
            
            # Display the answer
            st.write("**Answer:**", answer)
            
            # Display unique sources
            st.write("**Sources:**")
            unique_sources = set(doc.metadata["source"] for doc in source_documents)
            for source in unique_sources:
                st.write(source)
else:
    st.warning("Please enter your OpenAI API Key to proceed.")
