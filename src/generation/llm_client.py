"""
LLM Generation Component for RAG System

Handles answer generation using Groq API with Qwen model.
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTemplates:
    """Prompt templates for different RAG scenarios"""
    
    RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided documents. 

IMPORTANT GUIDELINES:
1. ONLY use information from the provided context documents
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question"
3. Be accurate and factual - do not make up information
4. Provide specific, detailed answers when possible
5. Reference the document numbers [Document X] when citing information
6. If asked about something not in the context, clearly state this limitation

Your role is to be helpful while staying grounded in the provided evidence."""

    RAG_USER_PROMPT = """Context Documents:
{context}

Question: {question}

Please provide a comprehensive answer based on the context documents. If the information isn't available in the context, please state that clearly."""

    SAFETY_SYSTEM_PROMPT = """You are a helpful AI assistant. You should:
1. Decline to answer harmful, illegal, or inappropriate questions
2. Be respectful and professional
3. Focus on being helpful while maintaining safety
4. If a question seems inappropriate, politely explain why you cannot answer it"""

    @classmethod
    def create_rag_prompt(cls, question: str, context: str) -> List[Dict[str, str]]:
        """Create RAG prompt with system and user messages"""
        return [
            {
                "role": "system",
                "content": cls.RAG_SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": cls.RAG_USER_PROMPT.format(context=context, question=question)
            }
        ]
    
    @classmethod
    def create_safety_check_prompt(cls, question: str) -> List[Dict[str, str]]:
        """Create prompt for safety checking"""
        return [
            {
                "role": "system",
                "content": cls.SAFETY_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Is this question appropriate and safe to answer? Question: '{question}'\n\nRespond with 'SAFE' if appropriate, or 'UNSAFE: [reason]' if not."
            }
        ]


class GuardrailsChecker:
    """Implements guardrails for safe and grounded responses"""
    
    def __init__(self):
        """Initialize guardrails checker"""
        self.harmful_keywords = [
            'hack', 'illegal', 'bomb', 'weapon', 'drug', 'suicide', 'violence',
            'fraud', 'scam', 'malware', 'virus', 'exploit', 'breach'
        ]
        
        self.uncertainty_patterns = [
            r'i think', r'i believe', r'probably', r'maybe', r'might be',
            r'could be', r'seems like', r'appears to', r'likely'
        ]
    
    def is_question_safe(self, question: str) -> Tuple[bool, str]:
        """Check if question is safe to answer"""
        question_lower = question.lower()
        
        # Check for harmful keywords
        for keyword in self.harmful_keywords:
            if keyword in question_lower:
                return False, f"Question contains potentially harmful content: {keyword}"
        
        # Check question length
        if len(question) > 1000:
            return False, "Question is too long"
        
        # Check for prompt injection attempts
        injection_patterns = [
            'ignore previous', 'forget instructions', 'new instructions',
            'system prompt', 'override', 'jailbreak', 'act as'
        ]
        
        for pattern in injection_patterns:
            if pattern in question_lower:
                return False, f"Potential prompt injection detected: {pattern}"
        
        return True, "Question is safe"
    
    def is_response_grounded(self, response: str, context: str) -> Tuple[bool, float]:
        """Check if response is grounded in the provided context"""
        if not response or not context:
            return False, 0.0
        
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Check for "I don't know" patterns (these are good!)
        dont_know_patterns = [
            "i don't have", "i don't know", "not enough information",
            "cannot find", "not available in", "not mentioned",
            "insufficient information", "unable to determine"
        ]
        
        for pattern in dont_know_patterns:
            if pattern in response_lower:
                return True, 1.0  # Admitting uncertainty is perfectly grounded
        
        # Check for uncertainty indicators
        uncertainty_count = 0
        for pattern in self.uncertainty_patterns:
            uncertainty_count += len(re.findall(pattern, response_lower))
        
        # Check for document references
        doc_references = len(re.findall(r'\[document \d+\]', response_lower))
        
        # Simple keyword overlap check
        response_words = set(response_lower.split())
        context_words = set(context_lower.split())
        overlap = len(response_words.intersection(context_words))
        total_response_words = len(response_words)
        
        if total_response_words == 0:
            return False, 0.0
        
        # Calculate grounding score
        overlap_score = min(overlap / total_response_words, 1.0)
        reference_score = min(doc_references * 0.2, 0.4)  # Bonus for references
        uncertainty_penalty = min(uncertainty_count * 0.1, 0.3)  # Small penalty for uncertainty
        
        confidence_score = overlap_score + reference_score - uncertainty_penalty
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Consider grounded if confidence > 0.3
        is_grounded = confidence_score > 0.3
        
        return is_grounded, confidence_score


class GroqLLMClient:
    """Groq API client for LLM generation"""
    
    def __init__(self):
        """Initialize Groq client"""
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "qwen/qwen-2.5-72b-instruct")
        
        if not self.api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            raise ValueError("GROQ_API_KEY is required")
        
        # Initialize client
        self.client = Groq(api_key=self.api_key)
        
        # Initialize components
        self.guardrails = GuardrailsChecker()
        
        logger.info(f"Groq client initialized with model: {self.model}")
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         temperature: float = 0.1,
                         max_tokens: int = 1000,
                         timeout: int = 30) -> Optional[str]:
        """Generate response using Groq API"""
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                stream=False
            )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                logger.info(f"Generated response: {len(content)} characters")
                return content
            else:
                logger.error("No response choices returned from Groq")
                return None
                
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return None
    
    def generate_rag_response(self, 
                             question: str,
                             context: str,
                             apply_guardrails: bool = True,
                             temperature: float = 0.1) -> Dict[str, Any]:
        """Generate RAG response with guardrails"""
        try:
            # Safety check
            if apply_guardrails:
                is_safe, safety_reason = self.guardrails.is_question_safe(question)
                if not is_safe:
                    return {
                        "answer": "I cannot answer this question due to safety concerns.",
                        "safe": False,
                        "safety_reason": safety_reason,
                        "grounded": True,  # Refusal is always grounded
                        "confidence": 1.0
                    }
            
            # Check if context is empty
            if not context or context.strip() == "No relevant documents found.":
                return {
                    "answer": "I don't have enough information to answer this question. No relevant documents were found.",
                    "safe": True,
                    "grounded": True,
                    "confidence": 1.0,
                    "reason": "no_context"
                }
            
            # Generate response
            messages = PromptTemplates.create_rag_prompt(question, context)
            
            response = self.generate_response(
                messages=messages,
                temperature=temperature,
                max_tokens=1200
            )
            
            if not response:
                return {
                    "answer": "I apologize, but I'm having trouble generating a response right now. Please try again.",
                    "safe": True,
                    "grounded": False,
                    "confidence": 0.0,
                    "error": "generation_failed"
                }
            
            # Grounding check
            grounded = True
            confidence = 0.8  # Default confidence
            
            if apply_guardrails:
                grounded, confidence = self.guardrails.is_response_grounded(response, context)
            
            return {
                "answer": response,
                "safe": True,
                "grounded": grounded,
                "confidence": confidence,
                "model": self.model,
                "temperature": temperature
            }
            
        except Exception as e:
            logger.error(f"RAG response generation failed: {e}")
            return {
                "answer": "I apologize, but I encountered an error while generating a response. Please try again.",
                "safe": True,
                "grounded": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Groq API connection"""
        try:
            test_messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Respond with exactly: 'Connection test successful'"
                },
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
            
            response = self.generate_response(test_messages, temperature=0.0, max_tokens=50)
            
            success = response is not None and "successful" in response.lower()
            
            return {
                "status": "success" if success else "partial",
                "model": self.model,
                "response_received": response is not None,
                "test_response": response[:100] if response else None
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Test connection
            connection_test = self.test_connection()
            
            # Test guardrails
            safe_question = "What is machine learning?"
            unsafe_question = "How to hack a computer?"
            
            safe_check = self.guardrails.is_question_safe(safe_question)
            unsafe_check = self.guardrails.is_question_safe(unsafe_question)
            
            guardrails_ok = safe_check[0] and not unsafe_check[0]
            
            return {
                "status": "healthy" if connection_test["status"] == "success" and guardrails_ok else "partial",
                "groq_api": connection_test,
                "guardrails": {
                    "status": "ok" if guardrails_ok else "error",
                    "safety_check": guardrails_ok
                },
                "configuration": {
                    "model": self.model,
                    "api_key_configured": bool(self.api_key)
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


def get_llm_client() -> GroqLLMClient:
    """Get configured Groq LLM client instance"""
    return GroqLLMClient()


if __name__ == "__main__":
    print("Testing Groq LLM Client...")
    
    try:
        client = get_llm_client()
        
        # Health check
        health = client.health_check()
        print(f"Health Status: {health}")
        
        # Test guardrails
        print("\nTesting Guardrails...")
        guardrails = GuardrailsChecker()
        
        test_questions = [
            "What is machine learning?",  # Safe
            "How to build a bomb?",       # Unsafe
            "Explain quantum computing"   # Safe
        ]
        
        for question in test_questions:
            is_safe, reason = guardrails.is_question_safe(question)
            status = "SAFE" if is_safe else "UNSAFE"
            print(f"  '{question[:30]}...': {status} - {reason}")
        
        # Test RAG response
        print("\nTesting RAG Response...")
        test_context = """[Document 1]
Source: ai_guide.pdf
Content: Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.

[Document 2]
Source: tech_overview.pdf  
Content: Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information."""
        
        test_question = "What is machine learning?"
        
        rag_response = client.generate_rag_response(test_question, test_context)
        print(f"RAG Response: {rag_response}")
        print(f"Answer preview: {rag_response.get('answer', '')[:200]}...")
        
        print("\nGroq LLM Client test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()