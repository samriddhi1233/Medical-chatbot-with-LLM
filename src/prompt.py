from langchain_core.prompts import ChatPromptTemplate

# System prompt text for Gemini RAG
system_prompt_text = (
    "You are a medical assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use a maximum of three sentences and keep the answer concise."
)

# ChatPromptTemplate for dynamic context + user input
system_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text + "\n\nCONTEXT:\n{context}"),
    ("human", "{input}")
])
