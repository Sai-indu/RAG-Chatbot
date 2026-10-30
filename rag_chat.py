from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel
import json
import requests
import os
import io
import tempfile
from typing import Optional, Dict, List, Tuple
import base64
import asyncio
from pathlib import Path

# ==================== LANGCHAIN + RAG IMPORTS ====================
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks import AsyncCallbackHandler

# RAG Components
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Whisper for voice transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    print("✓ Whisper model loaded")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper not installed. Voice messages disabled.")
    whisper_model = None

# Check if Ollama LLaVA is available
def check_ollama_llava():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            llava_models = [m['name'] for m in models if 'llava' in m.get('name', '').lower()]
            if llava_models:
                return True, llava_models[0]
            else:
                print("⚠️ LLaVA not found in Ollama")
                print("  Install: ollama pull llava")
                return False, None
    except Exception as e:
        print(f"⚠️ Cannot connect to Ollama: {e}")
        return False, None

LLAVA_AVAILABLE, LLAVA_MODEL = check_ollama_llava()
if LLAVA_AVAILABLE:
    print(f"✓ LLaVA found in Ollama: {LLAVA_MODEL}")

# OCR dependencies
OCR_AVAILABLE = False
try:
    from PIL import Image as PILImage
    import pytesseract
    import fitz  # PyMuPDF
    # Update these paths if Tesseract is installed elsewhere
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Nstore.Indumathi\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    os.environ['TESSDATA_PREFIX'] = r'C:\Users\Nstore.Indumathi\AppData\Local\Programs\Tesseract-OCR\tessdata'
    OCR_AVAILABLE = True
    print("✓ OCR dependencies loaded (PyMuPDF + Tesseract)")
except ImportError as e:
    print(f"⚠️ OCR not available: {e}")

# ==================== CONFIGURATION ====================
RAG_DOCS_PATH = "rag_documents"
FAISS_INDEX_PATH = "faiss_index"
COMMON_ISSUES_FILE = "atm_common_issues.json"

HIGH_CONFIDENCE_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.50
LOW_CONFIDENCE_THRESHOLD = 0.30

SYSTEM_INSTRUCTION = """You are an ATM Support Assistant for the ATM Operations department. Your role is to help ATM technicians and operators troubleshoot ATM issues quickly and effectively.
Guidelines:
- Provide clear, step-by-step solutions
- Reference error codes when relevant
- Prioritize safety and security
- Be concise and action-oriented
- When you receive context from documentation, cite the source
- Always maintain a professional and helpful tone
Department: ATM Operations
Focus: Hardware, Software, Cash Management, Network, Security issues
"""

# ==================== HELPER FUNCTIONS ====================
def load_common_issues():
    try:
        with open(COMMON_ISSUES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ {COMMON_ISSUES_FILE} not found!")
        return {"common_issues": [], "smes": []}
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return {"common_issues": [], "smes": []}

def create_rag_documents_folder():
    Path(RAG_DOCS_PATH).mkdir(exist_ok=True)
    print(f"✓ RAG documents folder ready: {RAG_DOCS_PATH}")

def load_and_split_documents(docs_path: str) -> List[Document]:
    documents = []

    if not os.path.exists(docs_path):
        print(f"⚠️ RAG documents path not found: {docs_path}")
        return documents

    # PDFs
    try:
        pdf_loader = DirectoryLoader(
            docs_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        print(f"✓ Loaded {len(pdf_docs)} PDF documents")
    except Exception as e:
        print(f"⚠️ Error loading PDFs: {e}")

    # TXT
    try:
        txt_loader = DirectoryLoader(
            docs_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True
        )
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        print(f"✓ Loaded {len(txt_docs)} TXT documents")
    except Exception as e:
        print(f"⚠️ Error loading TXT files: {e}")

    # DOCX
    try:
        doc_loader = DirectoryLoader(
            docs_path,
            glob="**/*.docx",
            loader_cls=UnstructuredWordDocumentLoader,
            show_progress=True
        )
        doc_docs = doc_loader.load()
        documents.extend(doc_docs)
        print(f"✓ Loaded {len(doc_docs)} DOCX documents")
    except Exception as e:
        print(f"⚠️ Error loading DOCX files: {e}")

    if not documents:
        print("⚠️ No documents found")
        return []

    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"→ Total content: {total_chars:,} characters")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"✓ Split into {len(split_docs)} chunks")
    return split_docs

def create_or_load_vectorstore(docs_path: str, index_path: str) -> Optional[FAISS]:
    print("→ Initializing HuggingFace embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✓ Embeddings model loaded")

    if os.path.exists(index_path):
        try:
            print(f"→ Loading FAISS index from {index_path}")
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True,
                distance_strategy="COSINE"
            )
            print(f"✓ FAISS index loaded with COSINE similarity ({vectorstore.index.ntotal} vectors)")
            return vectorstore
        except Exception as e:
            print(f"⚠️ Error loading index: {e}")
            print("→ Will create new index...")

    print("→ Creating new FAISS index...")
    documents = load_and_split_documents(docs_path)
    if not documents:
        print("⚠️ No documents to index")
        return None

    try:
        vectorstore = FAISS.from_documents(documents, embeddings, distance_strategy="COSINE")
        vectorstore.save_local(index_path)
        print(f"✓ FAISS index created and saved ({vectorstore.index.ntotal} vectors)")
        return vectorstore
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return None

# ==================== IMAGE & OCR FUNCTIONS ====================
def resize_image_if_needed(image_path: str, max_size_kb: int = 200) -> str:
    try:
        file_size_kb = os.path.getsize(image_path) / 1024
        if file_size_kb <= max_size_kb:
            return image_path

        img = PILImage.open(image_path)
        max_dimension = 400
        aspect_ratio = img.width / img.height
        if img.width > img.height:
            new_width = min(max_dimension, img.width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(max_dimension, img.height)
            new_width = int(new_height * aspect_ratio)

        img_resized = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

        if img_resized.mode in ('RGBA', 'LA', 'P'):
            background = PILImage.new('RGB', img_resized.size, (255, 255, 255))
            if img_resized.mode == 'P':
                img_resized = img_resized.convert('RGBA')
            background.paste(img_resized, mask=img_resized.split()[-1] if img_resized.mode in ('RGBA', 'LA') else None)
            img_resized = background

        resized_path = image_path.replace(os.path.splitext(image_path)[1], "_resized.jpg")
        img_resized.save(resized_path, "JPEG", quality=80, optimize=True)
        return resized_path
    except Exception as e:
        print(f"⚠️ Resize failed: {e}")
        return image_path

def analyze_atm_image_with_llava(image_path: str) -> str:
    llava_available, llava_model = check_ollama_llava()
    if not llava_available:
        return "Image analysis not available. Install: ollama pull llava"

    try:
        optimized_path = resize_image_if_needed(image_path, max_size_kb=200)
        with open(optimized_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        prompt = "List error codes and messages visible on this ATM screen. Be brief."
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": llava_model,
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 300, "num_ctx": 2048}
        }, timeout=180)

        if optimized_path != image_path and os.path.exists(optimized_path):
            os.remove(optimized_path)

        if response.status_code == 200:
            return f"=== ATM IMAGE ANALYSIS ===\n\n{response.json().get('response', 'No response')}\n\n=== END ==="
        else:
            return f"Error: LLaVA returned status {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def convert_pdf_to_images(pdf_path: str, dpi: int = 200):
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_data = pix.tobytes("png")
            img = PILImage.open(io.BytesIO(img_data))
            images.append(img)
        pdf_document.close()
        return images
    except Exception as e:
        raise Exception(f"PDF conversion failed: {str(e)}")

def extract_text_from_images(images, lang: str = 'eng'):
    extracted_text = []
    for idx, image in enumerate(images):
        try:
            text = pytesseract.image_to_string(image, lang=lang)
            extracted_text.append(f"--- Page {idx + 1} ---\n{text}\n")
        except Exception as e:
            extracted_text.append(f"--- Page {idx + 1} ---\n[OCR Error: {e}]\n")
    return "\n".join(extracted_text)

# ==================== RAG CHATBOT ====================
class RAGATMChatBot:
    def __init__(self):
        self.common_issues = load_common_issues()

        try:
            print("→ Initializing Ollama LLM...")
            self.llm = Ollama(model="llama3.2:3b", base_url="http://localhost:11434")
            test = self.llm.invoke("test")
            print(f"✓ LLM initialized and tested")
        except Exception as e:
            print(f"❌ LLM initialization failed: {e}")
            self.llm = None

        create_rag_documents_folder()
        self.rebuild_rag_index()
        self.rag_enabled = self.vectorstore is not None

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_INSTRUCTION),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        if self.llm:
            self.chain = prompt | self.llm
            self.chain_with_history = RunnableWithMessageHistory(
                self.chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
        else:
            self.chain = None
            self.chain_with_history = None

        self.store: Dict[str, InMemoryChatMessageHistory] = {}

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def check_common_issue(self, user_query: str) -> Optional[dict]:
        query_lower = user_query.lower()
        for issue in self.common_issues.get('common_issues', []):
            if issue['issue'].lower() in query_lower:
                return issue
            for symptom in issue.get('symptoms', []):
                if symptom.lower() in query_lower:
                    return issue
        return None

    def get_sme_contact(self, expertise_needed: Optional[str] = None, category: Optional[str] = None, source: str = "general") -> str:
        smes = self.common_issues.get('smes', [])
        if not smes:
            return "Contact your supervisor."

        category_to_sme = {
            'Card Issues': 0, 'Display': 0, 'Keypad Issues': 0, 'Printer Issues': 0, 'Sensor Issues': 0,
            'UPS/Battery': 0, 'Power': 0, 'Environmental': 0,
            'Cash Dispensing': 1, 'Cash Management': 1, 'Deposit Issues': 1,
            'Security': 2, 'Vault/Security': 2, 'Anti-Skimming': 2, 'Alarm Systems': 2,
            'Network': 3, 'Software': 3, 'Communication': 3, 'Remote Monitoring': 3, 'Accessibility': 3,
            'Check Deposit': 4,
        }

        if category and category in category_to_sme:
            idx = category_to_sme[category]
            if idx < len(smes):
                sme = smes[idx]
                return f"**👤 Expert Contact:**\n{sme['name']} - {sme['role']}\n📞 {sme['phone']}\n✉️ {sme['email']}"

        if expertise_needed:
            expertise_lower = expertise_needed.lower()
            expertise_keywords = {
                'hardware': ['hardware', 'card reader', 'dispenser', 'physical', 'mechanical', 'sensor', 'printer', 'display'],
                'cash': ['cash', 'dispenser', 'denomination', 'vault', 'cassette', 'currency', 'deposit'],
                'security': ['security', 'fraud', 'skimming', 'tamper', 'alarm', 'vault'],
                'software': ['software', 'system', 'update', 'patch', 'application', 'firmware'],
                'network': ['network', 'connection', 'connectivity', 'communication', 'remote', 'offline']
            }
            for sme in smes:
                sme_expertise = ' '.join([e.lower() for e in sme.get('expertise', [])])
                for keywords in expertise_keywords.values():
                    if any(kw in expertise_lower for kw in keywords) and any(kw in sme_expertise for kw in keywords):
                        return f"**👤 Expert Contact:**\n{sme['name']} - {sme['role']}\n📞 {sme['phone']}\n✉️ {sme['email']}"

        if source == "json" and len(smes) > 1:
            sme = smes[1]
            return f"**👤 SME Contact:**\n{sme['name']} - {sme['role']}\n📞 {sme['phone']}\n✉️ {sme['email']}"
        elif source == "rag":
            sme = smes[0]
            return f"**👤 Technical Support:**\n{sme['name']} - {sme['role']}\n📞 {sme['phone']}\n✉️ {sme['email']}"

        sme = smes[0]
        return f"**👤 Support Contact:**\n{sme['name']} - {sme['role']}\n📞 {sme['phone']}\n✉️ {sme['email']}"

    def format_common_issue_solution(self, issue: dict) -> str:
        """Format common issue solution WITHOUT duplicate numbering"""
        response = f"**🔍 Issue:** {issue['issue']}\n\n"
        response += f"**Category:** {issue['category']}\n"
        response += f"**Priority:** {issue['priority'].upper()}\n\n"

        solutions = issue.get('solutions', [issue.get('solution', '')])
        if isinstance(solutions, str):
            solutions = [s.strip() for s in solutions.split('\n') if s.strip()]

        response += f"**💡 Solutions:**\n"

        import re
        for idx, sol in enumerate(solutions[:7], 1):
            sol = sol.strip()
            # Remove existing numbering if present (1., 1), 1-, etc.)
            sol = re.sub(r'^\d+[\.\)\-]\s*', '', sol)
            # Remove bullet points
            sol = re.sub(r'^[-•]\s*', '', sol)
            # Add clean numbering
            response += f"{idx}. {sol}\n"

        if issue.get('error_codes'):
            response += f"\n**📟 Error Codes:** {', '.join(issue['error_codes'])}"

        if issue.get('priority') == 'critical':
            response += "\n⚠️ **CRITICAL - Act immediately**"

        response += "\n_____________________________________________________\n\n**If unresolved, contact:**\n" + self.get_sme_contact(
            category=issue.get('category'),
            source="json"
        )
        return response


    def search_rag_documents(self, query: str, k: int = 3) -> Tuple[List[Document], List[float]]:
        """Search RAG documents - should ONLY be called for technical queries"""

        if not self.rag_enabled or not self.vectorstore:
            print("⚠️ RAG not enabled or vectorstore missing")
            return [], []

        try:
            print(f"🔎 RAG SEARCH INITIATED for: '{query[:50]}...'")
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            docs = [doc for doc, _ in results]
            scores = [score for _, score in results]

            if scores:
                print(f"📊 RAG Search Results:")
                for idx, (doc, score) in enumerate(zip(docs, scores), 1):
                    source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    print(f"   {idx}. {source} - Score: {score:.2f}")

            return docs, scores
        except Exception as e:
            print(f"❌ RAG search error: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    def format_rag_response(self, query: str, docs: List[Document], scores: List[float]) -> str:
        """Format RAG response with proper step extraction (limited to 8 steps) and clean spacing for HTML rendering"""
        if not docs:
            return None
        
        top_doc = docs[0]
        top_score = scores[0]
        source = top_doc.metadata.get('source', 'Unknown')
        source_name = os.path.basename(source)
        content = top_doc.page_content.strip()
        
        import re
        
        solutions = []
        MAX_STEPS = 8  # Limit to 8 steps
        
        # Strategy 1: Match numbered items where number comes BEFORE the text
        numbered_pattern = r'(\d+)[\.\)\-\:]\s*([^0-9]+?)(?=\d+[\.\)\-\:]|$)'
        matches = re.findall(numbered_pattern, content, re.DOTALL)
        
        if matches and len(matches) >= 2:
            print(f"✓ Found {len(matches)} explicit numbered steps (limiting to {MAX_STEPS})")
            for num, text in matches[:MAX_STEPS]:
                # Clean the text: remove trailing numbers/punctuation and extra whitespace
                clean_text = re.sub(r'\d+[\.\)\-\:]*\s*$', '', text)
                clean_text = re.sub(r'\s+', ' ', clean_text.strip())
                clean_text = clean_text.rstrip('.,;:')
                
                if clean_text and len(clean_text) > 3:
                    solutions.append(f"{num}. {clean_text}")
        
        # Strategy 2: Handle cases where steps run together
        if not solutions:
            print("→ Fallback: splitting run-on numbered text")
            parts = re.split(r'(\d+)\.\s*', content)
            
            for i in range(1, len(parts)-1, 2):
                num = parts[i]
                text = parts[i+1].strip() if i+1 < len(parts) else ""
                
                if text:
                    clean_text = re.sub(r'\d+[\.\)\-\:]*\s*$', '', text)
                    clean_text = re.sub(r'\s+', ' ', clean_text.strip())
                    clean_text = clean_text.rstrip('.,;:')
                    
                    if len(clean_text) > 3:
                        solutions.append(f"{num}. {clean_text}")
                        
                    if len(solutions) >= MAX_STEPS:
                        break
        
        # Strategy 3: Line-by-line processing
        if not solutions:
            print("→ Fallback: line-by-line processing")
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                match = re.match(r'^(\d+)[\.\)\-\:]\s*(.+)$', line)
                if match:
                    num = match.group(1)
                    text = match.group(2).strip()
                    if len(text) > 3:
                        solutions.append(f"{num}. {text}")
                        if len(solutions) >= MAX_STEPS:
                            break
        
        # Final fallback
        if not solutions:
            print("→ Ultimate fallback: raw content")
            cleaned = re.sub(r'\s+', ' ', content.strip())
            formatted_content = cleaned[:500] + "..." if len(cleaned) > 500 else cleaned
        else:
            # Use <br> tags for HTML
            formatted_content = '<br>'.join(solutions)
        
        print(f"→ Final formatted RAG content:\n{formatted_content}\n{'─'*70}")
        
        # Category detection
        detected_category = None
        content_lower = content.lower()
        category_patterns = {
            'Cash Dispensing': ['cash', 'dispenser', 'dispense', 'denomination', 'bills', 'notes'],
            'Card Issues': ['card', 'reader', 'chip', 'magnetic', 'stripe', 'eject'],
            'Network': ['network', 'connection', 'connectivity', 'offline', 'communication'],
            'Security': ['security', 'fraud', 'skimming', 'tamper', 'alarm', 'vault'],
            'Software': ['software', 'system', 'update', 'patch', 'firmware', 'boot'],
            'Printer Issues': ['printer', 'receipt', 'paper', 'print', 'journal'],
            'Display': ['screen', 'display', 'touch', 'monitor', 'brightness']
        }
        
        for category, patterns in category_patterns.items():
            if any(p in content_lower for p in patterns):
                detected_category = category
                break
        
        # Build response
        confidence_label = (
            "good match" if top_score >= 0.7 else
            "moderate match" if top_score >= 0.5 else
            "low confidence"
        )
        
        header = f"<strong>💡 Suggested Resolution :</strong>"
        
        response = f"""{header}<br><br>
    {formatted_content}<br><br>
    <strong>Source:</strong> {source_name} (Confidence: {top_score:.2f})<br><br>
    <hr>
    <strong>If unresolved, contact:</strong><br>
    {self.get_sme_contact(category=detected_category, source="rag")}"""
        
        return response
    
    def chat(self, user_query: str, session_id: str) -> str:
        try:
            print(f"\n{'='*60}")
            print(f"📝 Query: {user_query[:50]}...")
            print(f"{'='*60}")

            # STEP 1: Detect greetings FIRST (before any RAG search)
            greeting_patterns = [
                'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon',
                'good evening', 'how are you', 'whats up', "what's up", 'sup',
                'who are you', 'what can you do', 'help me', 'introduce yourself',
                'what is your name', 'thank you', 'thanks', 'bye', 'goodbye'
            ]

            query_lower = user_query.lower().strip()

            # Check if it's a greeting or introduction (short query < 30 chars or matches patterns)
            is_greeting = (
                len(query_lower) < 30 and any(pattern in query_lower for pattern in greeting_patterns)
            ) or any(query_lower.startswith(pattern) for pattern in greeting_patterns)

            # IMPORTANT: If greeting detected, SKIP all RAG searches
            if is_greeting:
                print("✅ GREETING DETECTED - Skipping RAG, using LLM directly")
                print("🚫 RAG search bypassed")

                if not self.llm or not self.chain_with_history:
                    return "Hello! I'm the ATM Support Assistant. How can I help you today?"

                try:
                    print("→ Invoking LLM for greeting...")
                    response = self.chain_with_history.invoke(
                        {"input": user_query},
                        config={"configurable": {"session_id": session_id}}
                    )
                    response_text = response.content if hasattr(response, "content") else str(response)
                    print("✅ LLM response generated")
                    return response_text
                except Exception as e:
                    print(f"❌ LLM error: {e}")
                    return "Hello! I'm the ATM Support Assistant. I help with ATM troubleshooting. What issue are you facing?"

            print("→ Not a greeting, proceeding with technical flow...")

            # STEP 2: Check common issues JSON
            common_issue = self.check_common_issue(user_query)
            if common_issue:
                print("✅ Found in common issues JSON - Skipping RAG")
                return self.format_common_issue_solution(common_issue)

            # STEP 3: Check if query has technical keywords
            technical_keywords = [
                'error', 'code', 'issue', 'problem', 'fault', 'fail', 'not working',
                'crash', 'stuck', 'jam', 'dispense', 'card', 'cash', 'printer',
                'screen', 'display', 'network', 'offline', 'sensor', 'reader',
                'receipt', 'deposit', 'withdrawal', 'malfunction', 'alarm',
                'boot', 'restart', 'reboot', 'service', 'maintenance'
            ]

            has_technical_content = any(keyword in query_lower for keyword in technical_keywords)
            print(f"🔍 Technical content detected: {has_technical_content}")

            # STEP 4: Only search RAG if technical content exists
            if has_technical_content and self.rag_enabled and self.vectorstore:
                print("→ Searching RAG documents (technical query)...")
                try:
                    docs, scores = self.search_rag_documents(user_query, k=3)
                    if docs and scores:
                        max_score = max(scores)
                        print(f"📊 RAG confidence: {max_score:.2f}")

                        if max_score >= LOW_CONFIDENCE_THRESHOLD:
                            rag_response = self.format_rag_response(user_query, docs, scores)
                            if rag_response:
                                print("✅ Using RAG response")
                                return rag_response
                        else:
                            print(f"⚠️ RAG confidence too low ({max_score:.2f})")
                    else:
                        print("⚠️ No RAG results found")
                except Exception as e:
                    print(f"❌ RAG search error: {e}")
            else:
                print("🚫 RAG search skipped (no technical keywords)")

            # STEP 5: Fall back to LLM
            print("→ Using LLM for response...")

            if not self.llm or not self.chain_with_history:
                return f"⚠️ AI service not available.\n\n{self.get_sme_contact(source='general')}"

            try:
                response = self.chain_with_history.invoke(
                    {"input": user_query},
                    config={"configurable": {"session_id": session_id}}
                )
                response_text = response.content if hasattr(response, "content") else str(response)
                print("✅ LLM response generated")

                # Add SME contact if uncertain
                uncertain_phrases = ['unsure', 'not sure', "don't know", 'uncertain', 'cannot help']
                if any(phrase in response_text.lower() for phrase in uncertain_phrases):
                    response_text += "\n\n" + self.get_sme_contact(source="general")

                return response_text

            except Exception as e:
                print(f"❌ LLM error: {e}")
                return f"⚠️ Unable to process request.\n\n{self.get_sme_contact(source='general')}"

        except Exception as e:
            print(f"❌ Critical error: {e}")
            import traceback
            traceback.print_exc()
            return f"System error. Contact support:\n\n{self.get_sme_contact(source='general')}"

    async def chat_stream(self, user_query: str, websocket: WebSocket, session_id: str):
        try:
            print(f"\n{'='*60}")
            print(f"📝 WebSocket Query: {user_query[:50]}...")
            print(f"{'='*60}")

            # STEP 1: Detect greetings FIRST (before any RAG search)
            greeting_patterns = [
                'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon',
                'good evening', 'how are you', 'whats up', "what's up", 'sup',
                'who are you', 'what can you do', 'help me', 'introduce yourself',
                'what is your name', 'thank you', 'thanks', 'bye', 'goodbye',
                'this is', 'my name is', 'i am', "i'm"
            ]

            query_lower = user_query.lower().strip()

            # Check if it's a greeting or introduction
            is_greeting = (
                len(query_lower) < 30 and any(pattern in query_lower for pattern in greeting_patterns)
            ) or any(query_lower.startswith(pattern) for pattern in greeting_patterns[:13])  # First 13 are strict greetings

            # IMPORTANT: If greeting detected, SKIP all RAG searches
            if is_greeting:
                print("✅ GREETING DETECTED - Skipping RAG, using LLM directly")
                print("🚫 RAG search bypassed for greeting")

                class StreamCallback(AsyncCallbackHandler):
                    async def on_llm_new_token(self, token: str, **kwargs):
                        await websocket.send_json({"type": "stream", "content": token})

                async for chunk in self.chain_with_history.astream(
                    {"input": user_query},
                    config={"configurable": {"session_id": session_id}, "callbacks": [StreamCallback()]}
                ):
                    pass  # Streaming handled by callback

                await websocket.send_json({"type": "done", "content": ""})
                print("✅ LLM greeting response sent")
                return

            print("→ Not a greeting, proceeding with technical flow...")

            # STEP 2: Check common issues
            common_issue = self.check_common_issue(user_query)
            if common_issue:
                print("✅ Found in common issues JSON - Skipping RAG")
                solution = self.format_common_issue_solution(common_issue)
                lines = solution.split('\n')
                for line in lines:
                    await websocket.send_json({"type": "stream", "content": line + '\n'})
                    await asyncio.sleep(0.05)
                await websocket.send_json({"type": "done", "content": ""})
                return

            # STEP 3: Check if query has technical keywords
            technical_keywords = [
                'error', 'code', 'issue', 'problem', 'fault', 'fail', 'not working',
                'crash', 'stuck', 'jam', 'dispense', 'card', 'cash', 'printer',
                'screen', 'display', 'network', 'offline', 'sensor', 'reader',
                'receipt', 'deposit', 'withdrawal', 'malfunction', 'alarm',
                'boot', 'restart', 'reboot', 'service', 'maintenance'
            ]

            has_technical_content = any(keyword in query_lower for keyword in technical_keywords)
            print(f"🔍 Technical content detected: {has_technical_content}")

            # STEP 4: Check RAG only if technical content detected
            if has_technical_content and self.rag_enabled and self.vectorstore:
                print("→ Searching RAG documents (technical query)...")
                docs, scores = self.search_rag_documents(user_query, k=3)
                if docs and scores and max(scores) >= LOW_CONFIDENCE_THRESHOLD:
                    max_score = max(scores)
                    print(f"📊 RAG confidence: {max_score:.2f}")
                    rag_response = self.format_rag_response(user_query, docs, scores)

                    # Stream line by line
                    lines = rag_response.split('\n')
                    for line in lines:
                        await websocket.send_json({"type": "stream", "content": line + '\n'})
                        await asyncio.sleep(0.05)

                    await websocket.send_json({"type": "done", "content": ""})
                    print("✅ RAG response sent")
                    return
                else:
                    print("⚠️ RAG confidence too low or no results")
            else:
                print("🚫 RAG search skipped (no technical keywords)")

            # STEP 5: LLM streaming with callback
            print("→ Using LLM for response...")

            class StreamCallback(AsyncCallbackHandler):
                async def on_llm_new_token(self, token: str, **kwargs):
                    await websocket.send_json({"type": "stream", "content": token})

            full_response = ""
            async for chunk in self.chain_with_history.astream(
                {"input": user_query},
                config={"configurable": {"session_id": session_id}, "callbacks": [StreamCallback()]}
            ):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += token

            # Add SME contact if needed
            if any(w in full_response.lower() for w in ['unsure', 'contact', "don't know"]):
                sme_contact = "\n\n" + self.get_sme_contact(source="general")
                lines = sme_contact.split('\n')
                for line in lines:
                    await websocket.send_json({"type": "stream", "content": line + '\n'})
                    await asyncio.sleep(0.05)

            await websocket.send_json({"type": "done", "content": ""})
            print("✅ LLM response sent")

        except Exception as e:
            print(f"❌ WebSocket error: {e}")
            import traceback
            traceback.print_exc()
            await websocket.send_json({"type": "error", "content": f"Error: {str(e)}"})

    def reset_conversation(self, session_id: str):
        if session_id in self.store:
            del self.store[session_id]
        print(f"✓ Reset: {session_id}")

    def rebuild_rag_index(self):
        print("→ Rebuilding RAG index...")
        if os.path.exists(FAISS_INDEX_PATH):
            import shutil
            shutil.rmtree(FAISS_INDEX_PATH)
        self.vectorstore = create_or_load_vectorstore(RAG_DOCS_PATH, FAISS_INDEX_PATH)
        self.rag_enabled = self.vectorstore is not None
        return self.rag_enabled

# ==================== FASTAPI ====================
app = FastAPI(title="RAG ATM ChatBot", version="4.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

chatbot_sessions: Dict[str, RAGATMChatBot] = {}

class ChatRequest(BaseModel):
    department_id: str = "ATM"
    user_id: str = "user123"
    message: str

class ResetRequest(BaseModel):
    department_id: str = "ATM"
    user_id: str = "user123"

@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    await websocket.accept()
    session_id = f"ATM_{user_id}"
    if session_id not in chatbot_sessions:
        chatbot_sessions[session_id] = RAGATMChatBot()
    bot = chatbot_sessions[session_id]
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "").strip()
            if not message:
                await websocket.send_json({"type": "error", "content": "Empty message"})
                continue
            await websocket.send_json({"type": "start", "content": ""})
            await bot.chat_stream(message, websocket, session_id)
    except WebSocketDisconnect:
        print(f"Disconnected: {user_id}")

@app.get("/")
async def home():
    if os.path.exists('index.html'):
        return FileResponse('index.html')
    return HTMLResponse("""<html><body style="font-family:Arial;padding:50px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;">
    <h1>🏧 RAG ATM ChatBot v4.1</h1><p>✓ JSON + RAG + LLM + Voice + Vision + OCR + Smart SME Routing</p><p><a href="/docs" style="color:white;">📚 Docs</a></p></body></html>""")

@app.get("/health")
async def health():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_ok = r.status_code == 200
    except:
        ollama_ok = False
    return JSONResponse({
        'status': 'healthy',
        'ollama': 'connected' if ollama_ok else 'disconnected',
        'whisper': WHISPER_AVAILABLE,
        'llava': LLAVA_AVAILABLE,
        'ocr': OCR_AVAILABLE,
        'sessions': len(chatbot_sessions)
    })
@app.get("/api/common-issues")
async def get_common_issues():
    """Return all common ATM issues for frontend display"""
    try:
        issues_data = load_common_issues()
        # Return only the issues array, frontend doesn't need SMEs for quick cards
        return JSONResponse({
            "success": True,
            "issues": issues_data.get("common_issues", []),
            "count": len(issues_data.get("common_issues", []))
        })
    except Exception as e:
        raise HTTPException(500, f"Failed to load issues: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(400, "Empty message")
    session_id = f"{request.department_id}_{request.user_id}"
    if session_id not in chatbot_sessions:
        chatbot_sessions[session_id] = RAGATMChatBot()
    response_text = chatbot_sessions[session_id].chat(request.message, session_id)
    return JSONResponse({'response': response_text, 'session_id': session_id})

@app.post("/voice")
async def voice_endpoint(audio: UploadFile = File(...), department_id: str = Form("ATM"), user_id: str = Form("user123")):
    if not WHISPER_AVAILABLE:
        raise HTTPException(501, "Whisper not installed")
    temp_path = None
    try:
        temp_path = f"temp_audio_{user_id}.wav"
        with open(temp_path, "wb") as f:
            f.write(await audio.read())
        result = whisper_model.transcribe(temp_path, language='en')
        text = result['text'].strip()
        os.remove(temp_path)

        session_id = f"{department_id}_{user_id}"
        if session_id not in chatbot_sessions:
            chatbot_sessions[session_id] = RAGATMChatBot()
        response_text = chatbot_sessions[session_id].chat(text, session_id)
        return JSONResponse({'transcribed_text': text, 'response': response_text, 'session_id': session_id})
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(500, f'Voice error: {e}')

@app.post("/image-upload")
async def image_upload(file: UploadFile = File(...), department_id: str = Form("ATM"), user_id: str = Form("user123")):
    if not LLAVA_AVAILABLE:
        raise HTTPException(501, "LLaVA not available. Run: ollama pull llava")
    temp_path = None
    try:
        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}:
            raise HTTPException(400, f'Unsupported: {ext}')
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            temp_path = f.name
            f.write(await file.read())
        if os.path.getsize(temp_path) > 10*1024*1024:
            raise HTTPException(400, 'Max 10MB')

        # Return immediate acknowledgment with processing status
        # The frontend should show loading indicator when it receives this

        analysis = analyze_atm_image_with_llava(temp_path)
        os.remove(temp_path)

        session_id = f"{department_id}_{user_id}"
        if session_id not in chatbot_sessions:
            chatbot_sessions[session_id] = RAGATMChatBot()

        query = f"""Based on this ATM screen analysis, provide troubleshooting steps:
{analysis}
Please provide:
1. Issue description
2. Immediate actions
3. Step-by-step resolution
4. When to escalate"""
        response_text = chatbot_sessions[session_id].chat(query, session_id)
        return JSONResponse({
            'success': True,
            'image_analysis': analysis,
            'response': response_text,
            'session_id': session_id,
            'filename': file.filename,
            'processing_time': 'completed'  # Signal that processing is done
        })
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(500, f'Image error: {e}')

# Add this WebSocket endpoint to your backend (app.py)
# Place it AFTER the main chat WebSocket and BEFORE the image upload WebSocket

@app.websocket("/ws/voice/{user_id}")
async def websocket_voice(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for voice processing with real-time progress updates"""
    await websocket.accept()
    session_id = f"ATM_{user_id}"

    if not WHISPER_AVAILABLE:
        await websocket.send_json({
            "type": "error",
            "message": "Whisper not installed. Voice feature unavailable."
        })
        await websocket.close()
        return

    try:
        while True:
            # Receive audio data
            data = await websocket.receive_json()

            if data.get("type") == "audio_upload":
                # STEP 1: Receiving audio
                await websocket.send_json({
                    "type": "status",
                    "message": "🎤 Receiving audio...",
                    "stage": "upload"
                })

                await asyncio.sleep(0.3)

                # STEP 2: Transcribing with Whisper
                await websocket.send_json({
                    "type": "status",
                    "message": "🔄 Transcribing speech to text...",
                    "stage": "transcription"
                })

                try:
                    # Decode base64 audio
                    audio_data = data.get("audio_data")
                    if not audio_data:
                        raise ValueError("No audio data received")

                    # Handle base64 with data URL prefix
                    if ',' in audio_data:
                        audio_bytes = base64.b64decode(audio_data.split(',')[1])
                    else:
                        audio_bytes = base64.b64decode(audio_data)

                    # Save to temporary file
                    temp_path = f"temp_audio_{user_id}.wav"
                    with open(temp_path, "wb") as f:
                        f.write(audio_bytes)

                    # STEP 3: Transcribe with Whisper
                    print(f"→ Starting Whisper transcription for {user_id}...")
                    result = whisper_model.transcribe(temp_path, language='en')
                    text = result['text'].strip()
                    print(f"✓ Transcribed: {text[:100]}...")

                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    if not text:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No speech detected in audio. Please try again."
                        })
                        continue

                    # STEP 4: Send transcription result
                    await websocket.send_json({
                        "type": "response_start",
                        "transcribed_text": text
                    })

                    await asyncio.sleep(0.3)

                    # STEP 5: Generating response status
                    await websocket.send_json({
                        "type": "status",
                        "message": "✨ Generating response...",
                        "stage": "processing"
                    })

                    await asyncio.sleep(0.3)

                    # Get chatbot instance
                    if session_id not in chatbot_sessions:
                        chatbot_sessions[session_id] = RAGATMChatBot()

                    # STEP 6: Remove status and start streaming
                    await websocket.send_json({
                        "type": "remove_status"
                    })

                    await asyncio.sleep(0.2)

                    # Signal start of response
                    await websocket.send_json({
                        "type": "start"
                    })

                    # Stream the chatbot response
                    await chatbot_sessions[session_id].chat_stream(text, websocket, session_id)

                except ValueError as ve:
                    print(f"❌ Audio processing error: {ve}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Audio processing error: {str(ve)}"
                    })
                except Exception as e:
                    print(f"❌ Voice processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error processing voice: {str(e)}"
                    })

    except WebSocketDisconnect:
        print(f"Voice WebSocket disconnected: {user_id}")
    except Exception as e:
        print(f"Voice WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"WebSocket error: {str(e)}"
            })
        except:
            pass

# Add this UPDATED WebSocket endpoint to your backend (app.py)
# Replace the existing @app.websocket("/ws/image-upload/{user_id}") with this:

@app.websocket("/ws/image-upload/{user_id}")
async def websocket_image_upload(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for image upload with real-time progress updates"""
    await websocket.accept()
    session_id = f"ATM_{user_id}"

    try:
        while True:
            # Receive image data
            data = await websocket.receive_json()

            if data.get("type") == "image_upload":
                # STEP 1: Upload acknowledgment
                await websocket.send_json({
                    "type": "status",
                    "message": "📸 Image received, preparing analysis...",
                    "stage": "upload"
                })

                await asyncio.sleep(0.5)

                # STEP 2: Starting analysis
                await websocket.send_json({
                    "type": "status",
                    "message": "🔄 Analyzing ATM image with LLaVA...",
                    "stage": "analysis"
                })

                # Process image
                image_data = data.get("image_data")
                filename = data.get("filename", "image.jpg")

                try:
                    # Decode base64 image
                    import base64
                    image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)

                    temp_path = f"temp_image_{user_id}.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(image_bytes)

                    # STEP 3: Processing with LLaVA (THIS IS THE LONG PROCESS)
                    await websocket.send_json({
                        "type": "status",
                        "message": "🤖 Processing with LLaVA... This may take 1-2 minutes ⏳",
                        "stage": "processing"
                    })

                    # Send periodic "still processing" updates during LLaVA analysis
                    async def send_processing_updates():
                        """Send updates every 10 seconds during long processing"""
                        messages = [
                            "🔍 Still analyzing image details...",
                            "🧠 Extracting error codes and messages...",
                            "📊 Identifying ATM screen elements...",
                            "⚙️ Processing visual information..."
                        ]
                        for i, msg in enumerate(messages):
                            await asyncio.sleep(10)
                            await websocket.send_json({
                                "type": "status",
                                "message": msg,
                                "stage": "processing"
                            })

                    # Start background task for updates
                    update_task = asyncio.create_task(send_processing_updates())

                    # Analyze image with LLaVA (THIS IS THE 2-MINUTE PROCESS)
                    print(f"→ Starting LLaVA analysis for {filename}...")
                    analysis = analyze_atm_image_with_llava(temp_path)
                    print(f"✓ LLaVA analysis completed: {len(analysis)} chars")

                    # Cancel update task once processing is done
                    update_task.cancel()

                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    # STEP 4: Analysis complete - send result
                    await websocket.send_json({
                        "type": "status",
                        "message": "✅ Image analysis complete!",
                        "stage": "extraction"
                    })

                    await asyncio.sleep(0.3)

                    # Send the LLaVA analysis result
                    await websocket.send_json({
                        "type": "analysis_result",
                        "content": analysis
                    })

                    await asyncio.sleep(0.5)

                    # STEP 5: Generating troubleshooting steps
                    await websocket.send_json({
                        "type": "status",
                        "message": "✨ Generating troubleshooting steps...",
                        "stage": "generation"
                    })

                    await asyncio.sleep(0.5)

                    # Get chatbot response based on the analysis
                    if session_id not in chatbot_sessions:
                        chatbot_sessions[session_id] = RAGATMChatBot()

                    query = f"""Based on this ATM screen analysis, provide troubleshooting steps:
{analysis}
Please provide:
1. Issue description
2. Immediate actions
3. Step-by-step resolution
4. When to escalate"""

                    # Remove the processing status
                    await websocket.send_json({
                        "type": "remove_status"
                    })

                    await asyncio.sleep(0.2)

                    # Signal start of streaming response (this triggers blue dots)
                    await websocket.send_json({
                        "type": "start"
                    })

                    # Stream the response
                    await chatbot_sessions[session_id].chat_stream(query, websocket, session_id)

                except asyncio.CancelledError:
                    # Handle task cancellation gracefully
                    pass
                except Exception as e:
                    print(f"❌ Image processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error processing image: {str(e)}"
                    })

    except WebSocketDisconnect:
        print(f"Image upload WebSocket disconnected: {user_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()


# Add WebSocket endpoint for audio/voice processing
async def websocket_voice(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for voice processing with simplified progress updates"""
    await websocket.accept()
    session_id = f"ATM_{user_id}"

    if not WHISPER_AVAILABLE:
        await websocket.send_json({
            "type": "error",
            "message": "Whisper not installed"
        })
        return

    try:
        while True:
            # Receive audio data
            data = await websocket.receive_json()

            if data.get("type") == "audio_upload":
                # Step 1: Receiving audio
                await websocket.send_json({
                    "type": "status",
                    "message": "🎤 Receiving audio...",
                    "stage": "upload"
                })

                await asyncio.sleep(0.3)

                # Step 2: Transcribing
                await websocket.send_json({
                    "type": "status",
                    "message": "🔄 Transcribing speech to text...",
                    "stage": "transcription"
                })

                try:
                    # Decode base64 audio
                    audio_data = data.get("audio_data")
                    audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)

                    temp_path = f"temp_audio_{user_id}.wav"
                    with open(temp_path, "wb") as f:
                        f.write(audio_bytes)

                    # Transcribe with Whisper
                    result = whisper_model.transcribe(temp_path, language='en')
                    text = result['text'].strip()
                    os.remove(temp_path)

                    # Step 3: Generating response (with sparkles)
                    await websocket.send_json({
                        "type": "status",
                        "message": "✨ ⏳ Processing...",
                        "stage": "generating"
                    })

                    await asyncio.sleep(0.3)

                    # Get chatbot response
                    if session_id not in chatbot_sessions:
                        chatbot_sessions[session_id] = RAGATMChatBot()

                    # Start streaming the response
                    await websocket.send_json({
                        "type": "response_start",
                        "transcribed_text": text
                    })

                    await chatbot_sessions[session_id].chat_stream(text, websocket, session_id)

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Error processing audio: {str(e)}"
                    })

    except WebSocketDisconnect:
        print(f"Voice WebSocket disconnected: {user_id}")

@app.post("/ocr-upload")
async def ocr_upload(file: UploadFile = File(...), department_id: str = Form("ATM"), user_id: str = Form("user123")):
    if not OCR_AVAILABLE:
        raise HTTPException(501, "OCR not available")
    temp_path = None
    try:
        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}:
            raise HTTPException(400, f'Unsupported: {ext}')
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            temp_path = f.name
            f.write(await file.read())
        if os.path.getsize(temp_path) > 50*1024*1024:
            raise HTTPException(400, 'Max 50MB')

        if ext == '.pdf':
            images = convert_pdf_to_images(temp_path)
            text = extract_text_from_images(images)
        else:
            img = PILImage.open(temp_path)
            text = pytesseract.image_to_string(img, lang='eng')

        os.remove(temp_path)
        text = text.strip()

        if len(text) < 10:
            return JSONResponse({'success': False, 'warning': 'Little text extracted', 'extracted_text': text})

        session_id = f"{department_id}_{user_id}"
        if session_id not in chatbot_sessions:
            chatbot_sessions[session_id] = RAGATMChatBot()
        response_text = chatbot_sessions[session_id].chat(text, session_id)
        return JSONResponse({
            'success': True,
            'extracted_text': text,
            'response': response_text,
            'session_id': session_id,
            'char_count': len(text)
        })
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(500, f'OCR error: {e}')

@app.post("/reset")
async def reset_chat(request: ResetRequest):
    session_id = f"{request.department_id}_{request.user_id}"
    if session_id in chatbot_sessions:
        chatbot_sessions[session_id].reset_conversation(session_id)
    return JSONResponse({'message': 'Chat cleared', 'session_id': session_id})

@app.post("/rebuild-rag")
async def rebuild_rag():
    temp_bot = RAGATMChatBot()
    success = temp_bot.rebuild_rag_index()
    if success:
        for bot in chatbot_sessions.values():
            bot.vectorstore = temp_bot.vectorstore
            bot.rag_enabled = True
        return JSONResponse({'success': True, 'message': 'RAG rebuilt', 'sessions_updated': len(chatbot_sessions)})
    return JSONResponse({'success': False, 'message': 'Rebuild failed'}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    print("\n" + "="*70)
    print("🏧 RAG ATM CHATBOT v4.1 - Smart SME Routing Enabled")
    print("="*70)
    print("✓ http://localhost:5000")
    print(f"✓ Voice: {'ON' if WHISPER_AVAILABLE else 'OFF'}")
    print(f"✓ Vision: {'ON' if LLAVA_AVAILABLE else 'OFF'}")
    print(f"✓ OCR: {'ON' if OCR_AVAILABLE else 'OFF'}")
    print("\n" + "="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=5000)