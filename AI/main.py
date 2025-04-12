
import base64
import io
import logging
import os
import tempfile
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.cloud import speech
from google.cloud import texttospeech

# Configure environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths to your JSON credential files (update these paths)
STT_CREDENTIALS = "D:/GDGC/AI/stt.json"
TTS_CREDENTIALS = "D:/GDGC/AI/tts.json"
GEMINI_API_KEY = "AIzaSyA6b1GrsEZYA0m1CppGsnhJ72F8R7TE7BE"

# Initialize FastAPI app
app = FastAPI(title="AI Voice Assistant API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AITherapyAssistant:
    def __init__(self):
        self.setup_complete = False
        self.conversation_history = []
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.setup_complete = True

    def speech_to_text(self, audio_path):
        try:
            client = speech.SpeechClient.from_service_account_json(STT_CREDENTIALS)
            with io.open(audio_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=44100,
                language_code="en-US",
            )

            response = client.recognize(config=config, audio=audio)
            for result in response.results:
                transcription = result.alternatives[0].transcript
                logger.info(f"Google STT transcribed: '{transcription}'")
                return transcription
            return "I couldn't understand the audio. Could you please speak again?"
        except Exception as e:
            logger.error(f"Error in Google Cloud STT: {str(e)}")
            return "I couldn't understand the audio. Could you please speak again?"

    def generate_response(self, user_text):
        try:
            self.conversation_history.append({"role": "user", "content": user_text})
            prompt = "You are a supportive therapy assistant. Respond to this: " + user_text

            response = self.client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )
            ai_response = response.text.strip()

            self.conversation_history.append({"role": "assistant", "content": ai_response})
            return ai_response
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            return "I'm having trouble processing your request. Could you try again?"

    def text_to_speech(self, text):
        try:
            client = texttospeech.TextToSpeechClient.from_service_account_json(TTS_CREDENTIALS)
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Wavenet-D",
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=0.9
            )

            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            output_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            with open(output_path, "wb") as out:
                out.write(response.audio_content)

            logger.info(f"Google TTS generated audio at: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error in Google Cloud TTS: {str(e)}")
            return None

    def process_user_input(self, audio_path):
        if audio_path is None:
            return "No audio detected.", "Please provide an audio file.", None
        user_text = self.speech_to_text(audio_path)
        ai_response_text = self.generate_response(user_text)
        response_audio_path = self.text_to_speech(ai_response_text)
        return user_text, ai_response_text, response_audio_path

# Instantiate the assistant
assistant = AITherapyAssistant()

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with open(temp_audio.name, "wb") as buffer:
            buffer.write(await file.read())

        user_text, ai_response_text, response_audio_path = assistant.process_user_input(temp_audio.name)
        os.remove(temp_audio.name)

        if response_audio_path and os.path.exists(response_audio_path):
            with open(response_audio_path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
            os.remove(response_audio_path)
        else:
            audio_base64 = None

        return JSONResponse({
            "user_text": user_text,
            "ai_response": ai_response_text,
            "audio_response_base64": audio_base64
        })
    except Exception as e:
        logger.error(f"Error in /process_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
class TextRequest(BaseModel):
    message: str
@app.post("/process_text")
async def process_text(request: TextRequest):
    try:
        user_text = request.message
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        ai_response_text = assistant.generate_response(user_text)
        response_audio_path = assistant.text_to_speech(ai_response_text)

        if response_audio_path and os.path.exists(response_audio_path):
            with open(response_audio_path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
            os.remove(response_audio_path)
        else:
            audio_base64 = None

        return JSONResponse({
            "user_text": user_text,
            "reply": ai_response_text,
            "audio_response_base64": audio_base64
        })
    except Exception as e:
        logger.error(f"Error in /process_text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
