import streamlit as st
from dotenv import load_dotenv
import os
import logging
import time
from rag_llm import RAGPipelineSetup

# Đoạn mã để thêm Google Tag Manager vào phần `<head>`
gtm_head_code = """
<!-- Google Tag Manager -->
<script>
  (function(w,d,s,l,i){
    w[l]=w[l]||[];
    w[l].push({'gtm.start':
    new Date().getTime(),event:'gtm.js'});
    var f=d.getElementsByTagName(s)[0],
    j=d.createElement(s),
    dl=l!='dataLayer'?'&l='+l:'';
    j.async=true;
    j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl;
    f.parentNode.insertBefore(j,f);
  })(window,document,'script','dataLayer','GTM-WXJF39MM');
</script>
<!-- End Google Tag Manager -->
"""
st.markdown(gtm_head_code, unsafe_allow_html=True)

# Đoạn mã để thêm Google Tag Manager vào ngay sau thẻ mở `<body>`
gtm_body_code = """
<!-- Google Tag Manager (noscript) -->
<noscript>
  <iframe src="https://www.googletagmanager.com/ns.html?id=GTM-WXJF39MM"
  height="0" width="0" style="display:none;visibility:hidden"></iframe>
</noscript>
<!-- End Google Tag Manager (noscript) -->
"""
st.markdown(gtm_body_code, unsafe_allow_html=True)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSS để ẩn các phần không mong muốn
hide_elements_css = """
<style>
/* Ẩn biểu tượng GitHub và các lớp liên quan */
.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK {
  display: none !important;
}

/* Ẩn menu chính (MainMenu) */
#MainMenu {
  visibility: hidden !important;
}

/* Ẩn footer */
footer {
  visibility: hidden !important;
}

/* Ẩn header */
header {
  visibility: hidden !important;
}
</style>
"""
st.markdown(hide_elements_css, unsafe_allow_html=True)

# Thay đổi các thông tin theo cấu hình của bạn
DATABASE_TO_COLLECTION = {
    "Trường Đại học Khoa học Tự nhiên": "US_vectorDB",
    "Trường Đại học Công nghệ Thông tin": "UIT_vectorDB",
    "Trường Đại Học Khoa Học Xã Hội - Nhân Văn": "USSH_vectorDB",
    "Trường Đại Học Bách Khoa": "UT_vectorDB",
    "Trường Đại Học Quốc Tế": "IU_vectorDB",
    "Trường Đại Học Kinh tế - Luật": "UEL_vectorDB"
}

DATABASES = list(DATABASE_TO_COLLECTION.keys())

selected_database = st.sidebar.selectbox(
    "Chọn Trường cần hỏi đáp",
    DATABASES
)
st.sidebar.markdown("""
### **Chatbot tư vấn tuyển sinh**

Ứng dụng này được phát triển nhằm cung cấp thông tin và tư vấn tuyển sinh cho các trường Đại học.

### Nhóm tác giả:
- **Nguyễn Lê Lâm Phúc**
- **Trương Minh Hoàng**
- **Đặng Minh Phúc**
- **Nguyễn Thị Xuân Hương**
- **Lý Vĩnh Thuận**

Chúng tôi mong muốn mang đến sự hỗ trợ tốt nhất cho các bạn học sinh trong việc lựa chọn trường và ngành học phù hợp.
""")

selected_collection = DATABASE_TO_COLLECTION.get(selected_database, "US_vectorDB")

# Thông tin cấu hình từ rag_llm
EMBEDDINGS_MODEL_NAME =st.secrets["EMBEDDINGS_MODEL_NAME"]
QDRANT_URL = st.secrets["QDRANT_URL"]
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
RERANKER_MODEL_NAME = st.secrets["RERANKER_MODEL_NAME"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_KEY2= st.secrets["GROQ_API_KEY2"]
GROQ_API_KEY3= st.secrets["GROQ_API_KEY3"]

# Initialize session state if not already present
if "rag_pipeline" not in st.session_state or st.session_state.get("selected_database") != selected_database:
    st.session_state.selected_database = selected_database
    # Create instance of RAGPipelineSetup with selected database
    rag_setup = RAGPipelineSetup(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        qdrant_collection_name=selected_collection,
        huggingface_api_key=HUGGINGFACE_API_KEY,
        embeddings_model_name=EMBEDDINGS_MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
        groq_api_key2=GROQ_API_KEY2,
        groq_api_key3=GROQ_API_KEY3,
        reranker_model_name=RERANKER_MODEL_NAME
    )
    # Create or refresh pipeline with selected database
    st.session_state.rag_pipeline = rag_setup.rag(source=selected_collection)

# Streamed response generator with context (chat history)
def response_generator(prompt):
    try:
        # Retrieve the current conversation history (if any)
        # Only keep the most recent 5 user messages
        max_history = 5
        chat_history = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
        context = "\n".join(chat_history[-max_history:])

        start_time = time.time()  # Start timing
        response = st.session_state.rag_pipeline({"question": prompt, "chat_history": context})
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        formatted_response = f"Thời gian phản hồi: {elapsed_time:.2f} giây\n\n{response['answer']}"
        logger.info(f"Response generated: {formatted_response}")
        return formatted_response
    except Exception as e:
        logger.error(f"Error in response generation: {str(e)}")
        return f"Error: {str(e)}"

st.title("WSE-Bot")

# Display messages from history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input user question
if prompt := st.chat_input("Nhập câu hỏi của bạn:"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant's response with loading indicator
    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý câu hỏi của bạn..."):
            # Get formatted response
            response = response_generator(prompt)
            # Display response using st.markdown for proper formatting
            st.markdown(response)
        # Add assistant's response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
