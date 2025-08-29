"""
PDF Ingestion Pipeline for RAG System

Handles loading PDFs from Google Drive, text extraction, and chunking.
"""

import os
import io
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import PyPDF2
import pdfplumber
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextSplitter:
    """Handles text chunking with overlap for better context preservation"""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """Initialize text splitter"""
        self.chunk_size = max(chunk_size, 50)  # Minimum chunk size
        self.chunk_overlap = max(chunk_overlap, 10)  # Minimum overlap
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)"""
        return len(text) // 4
    
    def split_text_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences while preserving context"""
        import re
        
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create overlapping chunks from text"""
        try:
            # Split into sentences
            sentences = self.split_text_by_sentences(text)
            
            if not sentences:
                return []
            
            chunks = []
            current_chunk = ""
            current_tokens = 0
            chunk_index = 0
            
            for i, sentence in enumerate(sentences):
                sentence_tokens = self.estimate_tokens(sentence)
                
                # Check if adding this sentence exceeds chunk size
                if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_data = {
                        "text": current_chunk.strip(),
                        "chunk_index": chunk_index,
                        "token_count": current_tokens,
                        "sentence_start": max(0, len(chunks) * self.chunk_size - self.chunk_overlap),
                        "sentence_end": i
                    }
                    chunks.append(chunk_data)
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0 and len(chunks) > 0:
                        # Find sentences for overlap
                        overlap_text = ""
                        overlap_tokens = 0
                        
                        for j in range(i - 1, -1, -1):
                            sentence_tokens_j = self.estimate_tokens(sentences[j])
                            if overlap_tokens + sentence_tokens_j <= self.chunk_overlap:
                                overlap_text = sentences[j] + " " + overlap_text
                                overlap_tokens += sentence_tokens_j
                            else:
                                break
                        
                        current_chunk = overlap_text + sentence
                        current_tokens = overlap_tokens + sentence_tokens
                    else:
                        current_chunk = sentence
                        current_tokens = sentence_tokens
                else:
                    # Add sentence to current chunk
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_tokens += sentence_tokens
            
            # Add final chunk if not empty
            if current_chunk.strip():
                chunk_data = {
                    "text": current_chunk.strip(),
                    "chunk_index": chunk_index,
                    "token_count": current_tokens,
                    "sentence_start": len(sentences) - 1,
                    "sentence_end": len(sentences)
                }
                chunks.append(chunk_data)
            
            logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
            return chunks
            
        except Exception as e:
            logger.error(f"Text chunking failed: {e}")
            return [{
                "text": text[:1000] + "..." if len(text) > 1000 else text,
                "chunk_index": 0,
                "token_count": self.estimate_tokens(text),
                "sentence_start": 0,
                "sentence_end": 1
            }]


class GoogleDriveLoader:
    """Handles loading PDFs from Google Drive"""
    
    def __init__(self):
        """Initialize Google Drive API client"""
        self.credentials_file = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
        self.folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        
        if not self.folder_id:
            logger.warning("GOOGLE_DRIVE_FOLDER_ID not set in environment")
        
        self.service = None
        self._setup_drive_client()
    
    def _setup_drive_client(self):
        """Setup Google Drive API client"""
        try:
            if not os.path.exists(self.credentials_file):
                logger.error(f"Credentials file not found: {self.credentials_file}")
                return
            
            # Load credentials
            creds = Credentials.from_service_account_file(
                self.credentials_file,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            
            # Build service
            self.service = build('drive', 'v3', credentials=creds)
            logger.info("Google Drive API client initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup Google Drive client: {e}")
    
    def list_pdf_files(self, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all PDF files in Google Drive folder"""
        if not self.service:
            logger.error("Google Drive service not available")
            return []
        
        try:
            folder_id = folder_id or self.folder_id
            if not folder_id:
                logger.error("No folder ID provided")
                return []
            
            # Query for PDF files in folder
            query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name, size, modifiedTime, webViewLink)"
            ).execute()
            
            files = results.get('files', [])
            logger.info(f"Found {len(files)} PDF files in Google Drive")
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list PDF files: {e}")
            return []
    
    def download_file(self, file_id: str, file_name: str) -> Optional[bytes]:
        """Download file content from Google Drive"""
        if not self.service:
            logger.error("Google Drive service not available")
            return None
        
        try:
            # Download file
            request = self.service.files().get_media(fileId=file_id)
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    logger.debug(f"Download progress: {int(status.progress() * 100)}%")
            
            file_content = file_io.getvalue()
            logger.info(f"Downloaded file: {file_name} ({len(file_content)} bytes)")
            
            return file_content
            
        except Exception as e:
            logger.error(f"Failed to download file {file_name}: {e}")
            return None


class PDFProcessor:
    """Handles PDF text extraction and processing"""
    
    def __init__(self):
        """Initialize PDF processor"""
        self.text_splitter = TextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 300)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50))
        )
    
    def extract_text_pypdf2(self, pdf_content: bytes) -> str:
        """Extract text using PyPDF2"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num + 1}: {e}")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_content: bytes) -> str:
        """Extract text using pdfplumber (better for complex layouts)"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                text = ""
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                
                return text.strip()
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return ""
    
    def extract_text(self, pdf_content: bytes) -> str:
        """Extract text using multiple methods (fallback approach)"""
        # Try pdfplumber first (better quality)
        text = self.extract_text_pdfplumber(pdf_content)
        
        # Fallback to PyPDF2 if pdfplumber fails
        if not text or len(text) < 50:
            logger.info("Falling back to PyPDF2 for text extraction")
            text = self.extract_text_pypdf2(pdf_content)
        
        # Final check
        if not text or len(text) < 10:
            logger.warning("Very little text extracted from PDF")
            return "No readable text found in PDF"
        
        return text
    
    def process_pdf(self, pdf_content: bytes, filename: str, drive_url: str) -> List[Dict[str, Any]]:
        """Process PDF: extract text and create chunks"""
        try:
            # Extract text
            text = self.extract_text(pdf_content)
            
            if not text:
                logger.error(f"No text extracted from {filename}")
                return []
            
            # Create chunks
            chunks = self.text_splitter.create_chunks(text)
            
            # Prepare document chunks with metadata
            document_chunks = []
            file_hash = hashlib.md5(pdf_content).hexdigest()[:8]
            
            for chunk_data in chunks:
                chunk_id = f"{filename}_{file_hash}_chunk_{chunk_data['chunk_index']}"
                
                # Create text snippet for citations (first 150 chars)
                text_snippet = chunk_data['text'][:150] + "..." if len(chunk_data['text']) > 150 else chunk_data['text']
                
                doc_chunk = {
                    "text": chunk_data['text'],
                    "text_snippet": text_snippet,
                    "filename": filename,
                    "drive_url": drive_url,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_data['chunk_index'],
                    "file_type": "pdf",
                    "file_hash": file_hash,
                    "token_count": chunk_data['token_count'],
                    "created_at": datetime.utcnow().isoformat(),
                    "extraction_method": "pdfplumber+pypdf2"
                }
                
                document_chunks.append(doc_chunk)
            
            logger.info(f"Processed PDF {filename}: {len(document_chunks)} chunks created")
            return document_chunks
            
        except Exception as e:
            logger.error(f"PDF processing failed for {filename}: {e}")
            return []


class PDFIngestionPipeline:
    """Complete PDF ingestion pipeline: Google Drive → Text Extraction → Chunking"""
    
    def __init__(self):
        """Initialize ingestion pipeline"""
        self.drive_loader = GoogleDriveLoader()
        self.pdf_processor = PDFProcessor()
    
    def ingest_folder(self, folder_id: Optional[str] = None, max_files: Optional[int] = None) -> List[Dict[str, Any]]:
        """Ingest all PDFs from Google Drive folder"""
        try:
            # List PDF files
            pdf_files = self.drive_loader.list_pdf_files(folder_id)
            
            if not pdf_files:
                logger.warning("No PDF files found in folder")
                return []
            
            # Limit files if specified
            if max_files:
                pdf_files = pdf_files[:max_files]
                logger.info(f"Processing {len(pdf_files)} PDFs (limited by max_files)")
            
            all_chunks = []
            
            for i, file_info in enumerate(pdf_files, 1):
                logger.info(f"Processing file {i}/{len(pdf_files)}: {file_info['name']}")
                
                # Download PDF
                pdf_content = self.drive_loader.download_file(
                    file_info['id'], 
                    file_info['name']
                )
                
                if not pdf_content:
                    logger.warning(f"Skipping {file_info['name']} - download failed")
                    continue
                
                # Process PDF
                chunks = self.pdf_processor.process_pdf(
                    pdf_content,
                    file_info['name'],
                    file_info.get('webViewLink', f"https://drive.google.com/file/d/{file_info['id']}")
                )
                
                all_chunks.extend(chunks)
                logger.info(f"{file_info['name']}: {len(chunks)} chunks added")
            
            logger.info(f"Ingestion complete: {len(all_chunks)} total chunks from {len(pdf_files)} PDFs")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Folder ingestion failed: {e}")
            return []
    
    def ingest_single_file(self, file_id: str) -> List[Dict[str, Any]]:
        """Ingest a single PDF file by ID"""
        try:
            if not self.drive_loader.service:
                logger.error("Google Drive service not available")
                return []
            
            # Get file metadata
            file_info = self.drive_loader.service.files().get(
                fileId=file_id,
                fields="id, name, size, modifiedTime, webViewLink"
            ).execute()
            
            # Download and process
            pdf_content = self.drive_loader.download_file(file_id, file_info['name'])
            
            if not pdf_content:
                return []
            
            chunks = self.pdf_processor.process_pdf(
                pdf_content,
                file_info['name'],
                file_info.get('webViewLink', f"https://drive.google.com/file/d/{file_id}")
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"Single file ingestion failed: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for ingestion pipeline"""
        try:
            drive_ok = self.drive_loader.service is not None
            
            # Test folder access if folder_id is set
            folder_access = False
            if self.drive_loader.folder_id and drive_ok:
                try:
                    files = self.drive_loader.list_pdf_files()
                    folder_access = True
                except:
                    pass
            
            return {
                "status": "healthy" if drive_ok else "partial",
                "google_drive": {
                    "status": "ok" if drive_ok else "error",
                    "service_initialized": drive_ok,
                    "folder_access": folder_access,
                    "folder_id": self.drive_loader.folder_id
                },
                "pdf_processor": {
                    "status": "ok",
                    "chunk_size": self.pdf_processor.text_splitter.chunk_size,
                    "chunk_overlap": self.pdf_processor.text_splitter.chunk_overlap
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


def get_pdf_ingestion_pipeline() -> PDFIngestionPipeline:
    """Get configured PDF ingestion pipeline"""
    return PDFIngestionPipeline()


if __name__ == "__main__":
    print("Testing PDF Ingestion Pipeline...")
    
    pipeline = get_pdf_ingestion_pipeline()
    
    # Health check
    health = pipeline.health_check()
    print(f"Health Status: {health}")
    
    # Test text splitter
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    test_text = "This is a test document. It contains multiple sentences. Each sentence should be processed correctly. The chunking should work well with overlaps. This ensures good context preservation."
    
    chunks = splitter.create_chunks(test_text)
    print(f"Text Splitter Test: {len(chunks)} chunks created")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk['text'])} chars, ~{chunk['token_count']} tokens")
    
    print("PDF Ingestion Pipeline test completed!")