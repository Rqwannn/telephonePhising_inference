from flask import request, jsonify
from flask_restful import Resource, reqparse
import os
import json
import asyncio
import re
from typing import Dict, List, Any

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

from dotenv import load_dotenv

load_dotenv()
set_llm_cache(InMemoryCache())  # Add caching for repeated queries

class InsightOpenAI(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('text', type=str, location='form', required=True)
        self.parser.add_argument('language', type=str, location='form', default='id')  # Support multiple languages

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "chatbot"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API")

        self.openai_api_key = os.getenv("OPENAI_API")
        
        # Using GPT-4o for optimal performance and speed
        self.chat_model = ChatOpenAI(
            api_key=self.openai_api_key,
            model="gpt-4o",
            temperature=0.2,  # Lower temperature for more consistent results and better JSON formatting
            max_tokens=4096,   # Increased token limit for comprehensive analysis
            request_timeout=30, # Increased timeout for complex analyses
        )

        # Setup enhanced tools
        self._setup_tools()
        
        # Memory for conversation context
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def _setup_tools(self):
        # Tool 1: Improved speaker role identification with better JSON handling
        def identify_speaker_roles(text):
            prompt = f"""
            Berikut adalah transkrip percakapan telepon yang diduga voice phishing:

            "{text}"

            🔍 Analisis mendalam tentang peran speaker:
            1. Identifikasi dengan pasti siapa yang merupakan penipu dan korban
            2. Jelaskan bukti kuat yang menunjukkan peran tersebut
            3. Berikan detail karakteristik bahasa yang digunakan oleh penipu
            4. Identifikasi taktik manipulasi atau teknik social engineering yang digunakan
            
            PENTING: Berikan hasil HANYA dalam format JSON yang valid dengan struktur berikut:
            {{
              "penipu": ["SPEAKER_XX", ...],
              "korban": ["SPEAKER_XX", ...],
              "bukti": "penjelasan bukti yang kuat",
              "karakteristik_bahasa_penipu": ["ciri1", "ciri2", ...],
              "taktik_manipulasi": ["taktik1", "taktik2", ...]
            }}
            
            JANGAN sertakan penjelasan tambahan di luar objek JSON. Pastikan format JSON valid dan dapat di-parse.
            """

            response = self.chat_model.invoke([HumanMessage(content=prompt)])
            return response.content.strip()

        self.role_tool = Tool(
            name="IdentifySpeakerRoles",
            func=identify_speaker_roles,
            description="Menentukan siapa yang merupakan penipu dan korban dalam percakapan dengan analisis mendalam."
        )

        # Tool 2: Enhanced voice phishing analysis
        def analyze_voice_phishing(text):
            prompt = f"""
            Berikut adalah transkrip percakapan telepon yang diduga sebagai voice phishing:

            "{text}"

            🔍 Berikan analisis komprehensif dengan struktur berikut:
            
            1. RINGKASAN EKSEKUTIF:
               - Jelaskan ringkasan singkat tentang modus penipuan yang digunakan
            
            2. MODUS OPERANDI:
               - Identifikasi jenis teknik voice phishing spesifik (misalnya: spoofing, vishing, social engineering)
               - Jelaskan tahapan-tahapan yang dilakukan penipu untuk meyakinkan korban
               - Identifikasi target informasi/data yang ingin diperoleh oleh penipu
            
            3. TANDA-TANDA PERINGATAN:
               - Minimal 5 red flags atau tanda peringatan dalam percakapan ini
               - Kutip bagian percakapan yang menunjukkan tanda peringatan tersebut
            
            4. DAMPAK POTENSIAL:
               - Jelaskan konsekuensi yang mungkin terjadi jika korban mengikuti arahan penipu
               - Perkirakan tingkat kerugian finansial/data yang mungkin terjadi
            
            5. METODE PENCEGAHAN SPESIFIK:
               - Minimal 5 langkah konkrit yang dapat diambil untuk mengenali dan mencegah kasus serupa
               - Sertakan cara verifikasi identitas yang benar
            
            6. PROTOKOL PENANGANAN:
               - Langkah-langkah yang harus diambil jika seseorang sudah menjadi korban
               - Pihak yang harus dihubungi dan informasi yang perlu disiapkan
               - Timeline penanganan yang disarankan
            
            Berikan analisis yang sangat informatif, spesifik, dan praktis.
            """

            response = self.chat_model.invoke([HumanMessage(content=prompt)])
            return response.content.strip()

        self.insight_tool = Tool(
            name="VoicePhishingAnalyzer",
            func=analyze_voice_phishing,
            description="Menganalisis percakapan voice phishing dan memberikan insight komprehensif."
        )

        # Tool 3: Pattern recognition for similar scams
        def recognize_scam_pattern(text):
            prompt = f"""
            Berikut adalah transkrip percakapan telepon yang diduga voice phishing:

            "{text}"

            🔍 Lakukan analisis pola penipuan:
            1. Identifikasi pola spesifik dalam percakapan ini yang cocok dengan jenis penipuan terkenal
            2. Bandingkan dengan kasus-kasus serupa yang umum terjadi
            3. Jelaskan varian spesifik dari teknik penipuan yang digunakan
            4. Berikan cara untuk mengenali pola serupa di masa depan
            
            Format dengan judul jelas dan poin-poin spesifik.
            """

            response = self.chat_model.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        self.pattern_tool = Tool(
            name="ScamPatternRecognizer",
            func=recognize_scam_pattern,
            description="Mengidentifikasi pola penipuan dan membandingkan dengan kasus serupa."
        )
        
        # Tool 4: Security recommendation generator
        def generate_security_recommendations(text):
            prompt = f"""
            Berdasarkan transkrip percakapan voice phishing berikut:

            "{text}"

            🔍 Berikan rekomendasi keamanan yang sangat spesifik dan praktis:
            1. Teknologi pengamanan yang dapat mencegah kasus serupa
            2. Pengaturan privasi dan keamanan yang disarankan
            3. Alat verifikasi identitas yang dapat digunakan
            4. Protokol komunikasi yang aman untuk transaksi sensitif
            5. Pendidikan dan pelatihan yang diperlukan untuk mencegah kasus serupa
            
            Pastikan rekomendasi bersifat konkrit, dapat ditindaklanjuti, dan sesuai dengan konteks percakapan.
            """

            response = self.chat_model.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        self.security_tool = Tool(
            name="SecurityRecommendationGenerator",
            func=generate_security_recommendations,
            description="Menghasilkan rekomendasi keamanan spesifik berdasarkan kasus penipuan."
        )

    def _extract_json(self, text):
        """Helper method to extract JSON from text that might contain other content"""
        # Find JSON pattern using regex
        json_match = re.search(r'({[\s\S]*})', text)
        if json_match:
            try:
                # Try to parse the extracted JSON
                return json.loads(json_match.group(1))
            except:
                pass
        
        # If we can't extract JSON with regex, try a more aggressive approach
        try:
            # Look for first { and last }
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
        except:
            pass
            
        # Return a default structure if all parsing fails
        return {
            "penipu": [],
            "korban": [],
            "bukti": "Tidak dapat mengekstrak bukti",
            "karakteristik_bahasa_penipu": [],
            "taktik_manipulasi": []
        }

    async def _process_analysis(self, text):
        """Process analysis asynchronously for faster response"""
        # Create tasks to run in parallel
        role_task = asyncio.create_task(self._get_role_analysis(text))
        insight_task = asyncio.create_task(self._get_insight_analysis(text))
        pattern_task = asyncio.create_task(self._get_pattern_analysis(text))
        security_task = asyncio.create_task(self._get_security_recommendations(text))
        
        # Gather all results
        role_result, insight_result, pattern_result, security_result = await asyncio.gather(
            role_task, insight_task, pattern_task, security_task
        )
        
        # Process and combine results with better JSON handling
        role_json = self._extract_json(role_result)
        
        return {
            "roles": role_json,
            "insight": insight_result,
            "pattern_analysis": pattern_result,
            "security_recommendations": security_result
        }
    
    async def _get_role_analysis(self, text):
        return await asyncio.to_thread(self.role_tool.func, text)
    
    async def _get_insight_analysis(self, text):
        return await asyncio.to_thread(self.insight_tool.func, text)
    
    async def _get_pattern_analysis(self, text):
        return await asyncio.to_thread(self.pattern_tool.func, text)
    
    async def _get_security_recommendations(self, text):
        return await asyncio.to_thread(self.security_tool.func, text)

    def post(self):
        try:
            args = self.parser.parse_args()
            text = args['text']
            
            # Use async processing for speed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(self._process_analysis(text))
            loop.close()
            
            # Add original text and message
            results["original"] = text
            results["message"] = "successful"
            
            return jsonify(results)
            
        except Exception as e:
            # Enhanced error handling with more information
            error_details = {
                "message": f"Terjadi masalah saat memproses permintaan: {str(e)}",
                "type": str(type(e).__name__),
                "original_text_length": len(args['text']) if 'args' in locals() and args and 'text' in args else "unknown"
            }
            
            return jsonify(error_details), 500