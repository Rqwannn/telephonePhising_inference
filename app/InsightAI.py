from flask import request, jsonify
from flask_restful import Resource, reqparse
import os

from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv

load_dotenv()

class InsightAI(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('text', type=str, location='form', required=True)  # Bisa JSON

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "chatbot"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API")

        self.anthropic_api_key = os.getenv("ANTHROPIC_API")
        
        self.chat_model = ChatAnthropic(
            api_key=self.anthropic_api_key,
            model="claude-3-5-sonnet-20241022",
            temperature=0.4,
            max_tokens=2448
        )

        # 🔹 Tool 1: Identifikasi peran masing-masing speaker
        def identify_speaker_roles(text):
            prompt = f"""
            Berikut adalah transkrip percakapan telepon yang diduga voice phishing:

            "{text}"

            🔍 Identifikasi peran setiap speaker dalam percakapan ini:
            - **SPEAKER_XX**: Penipu atau Korban?
            - Berikan hasil dalam format JSON dengan format:
              {{"penipu": ["SPEAKER_XX", ...], "korban": ["SPEAKER_XX", ...]}}
            """

            response = self.chat_model.invoke([HumanMessage(content=prompt)])
            return response.content.strip()

        self.role_tool = Tool(
            name="IdentifySpeakerRoles",
            func=identify_speaker_roles,
            description="Menentukan siapa yang merupakan penipu dan korban dalam percakapan."
        )

        # 🔹 Tool 2: Analisis dan insight pencegahan
        def analyze_voice_phishing(text):
            prompt = f"""
            Berikut adalah transkrip percakapan telepon yang diduga sebagai voice phishing:

            "{text}"

            🔍 Analisis:
            - Apa modus penipuan yang digunakan?
            - Apa tanda-tanda voice phishing dalam percakapan ini?
            - Bagaimana cara mencegah kejadian serupa?
            - Apa yang harus dilakukan user jika mengalami hal serupa?
            """

            response = self.chat_model.invoke([HumanMessage(content=prompt)])
            return response.content.strip()

        self.insight_tool = Tool(
            name="VoicePhishingAnalyzer",
            func=analyze_voice_phishing,
            description="Menganalisis percakapan voice phishing dan memberikan insight pencegahan."
        )

        # 🔹 Memory agar agent dapat menyimpan konteks percakapan
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # 🔹 Inisialisasi Agent
        self.agent = initialize_agent(
            tools=[self.role_tool, self.insight_tool],
            llm=self.chat_model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

    def post(self):
        try:
            args = self.parser.parse_args()
            text = args['text']

            role_response = self.agent.run(f"Identifikasi peran speaker dalam percakapan ini:\n{text}")

            insight_response = self.agent.run(f"Analisis percakapan ini:\n{text}")

            return jsonify(
                {
                    "roles": role_response,
                    "insight": insight_response,
                    "original": text,
                    "message": "successful",
                }
            )
        except Exception as e:
            return jsonify({
                "message": f"Terjadi masalah saat memproses permintaan: {str(e)}"
            })
