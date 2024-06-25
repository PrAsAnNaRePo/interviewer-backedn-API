import tempfile
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import time
from prompt_template import SYSTEM_PROMPT
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import base64
from pydantic import BaseModel, Field
import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(level=logging.INFO)
file_handler = RotatingFileHandler("app.log", maxBytes=10000, backupCount=3)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs('static', exist_ok=True)

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

origins = [
    "http://localhost:5174",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BotConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    personality: str = Field(..., min_length=1, max_length=500)
    major_questions: List[Dict[str, str]] = Field(..., min_items=1)
    response_type: str = Field(..., min_length=1, max_length=100)
    temperature: float = Field(..., ge=0.0, le=1.0)
    audio_base64: str
    message_history: List[Dict[str, str]] = []


@app.get("/")
async def root():
    return {"message": "It's working"}


@app.post("/inter-val")
async def interview_validate(request: Request, bot_config: BotConfig):
    try:
        start_time = time.time()
        user_message = transcribe_audio(bot_config.audio_base64)
        chat_response = get_chat_response(user_message, bot_config)
        tts_status = text_to_speech(chat_response['response'])

        audio_file_path = "static/response.mp3"
        with open("audio_file_path", "rb") as audio_file:
            audio_content = audio_file.read()

        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        download_url = str(request.base_url) + 'static/response.mp3'
        end_time = time.time()

        return JSONResponse(
            content={
                "created_at": int(start_time),
                "time_took": end_time - start_time,
                "bot_config": bot_config.model_dump(),
                "chat_response": {
                    "user_message": user_message,
                    "content": chat_response
                },
                "tts": {
                    "audio_url": download_url,
                    "audio_base64": audio_base64,
                    "tts_status": tts_status
                }
            }
        )
    except Exception as e:
        logger.error(f"Error in post_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/init-conv")
async def init_conv():
    start_time = time.time()
    try:
        _message = [
            {
                'role': 'system',
                'content': """You are a Interviewer. Who interviews user and have a great natural friendly conversation.
## Note:
- Start the conversation with Greeting and ask if they ready for the interview.
- Make your response more sounds like natural, human-like and creative."""
            },
        ]
        message_content = [
            {
                "role": 'user',
                'content': 'Hey!!'
            }
        ]

        prompt_msg = _message + message_content
        response = client.chat.completions.create(
            messages=prompt_msg,
            model='gpt-3.5-turbo',
            temperature=1.0
        )

        return JSONResponse(
            content={
                "created_at": int(start_time),
                "response": response.model_dump()
            }
        )

    except Exception as e:
        logger.error(f"Error in init_conv: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def transcribe_audio(audio_base64):
    try:
        audio_data = base64.b64decode(audio_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_audio_file.write(audio_data)
            temp_audio_file_path = temp_audio_file.name

        with open(temp_audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model='whisper-1'
            )
        return transcript.text
    
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        raise
    finally:
        if 'temp_audio_file_path' in locals():
            os.unlink(temp_audio_file_path)

def get_chat_response(user_message, bot_config: BotConfig):
    try:
        majr_ques_str = '\n'.join([f"{idx+1}. {ques['question']} - Ask follow-ups in {ques['difficulity']} mode for this question." for idx, ques in enumerate(bot_config.major_questions)])

        messages = [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT.format(
                    name=bot_config.name,
                    personality=bot_config.personality,
                    major_questions=majr_ques_str,
                    response_type=bot_config.response_type
                )
            }
        ] + bot_config.message_history
        
        verbs_analysis = get_verb_analysis(prompt=user_message)
        messages.append({
            "role": "user", 
            "content": user_message + (f"\n<verb>Found a verb {verbs_analysis['verb']}</verb>" if verbs_analysis['is_verb_present'] == 'yes' else "")
        })
        
        gpt_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=bot_config.temperature
        )

        parsed_gpt_response = gpt_response.choices[0].message.content
        token_usage = gpt_response.usage

        return {
            "response": parsed_gpt_response,
            "messages": bot_config.message_history + [
                {
                    'role': 'user',
                    'content': user_message
                },
                {
                    'role': 'assistant',
                    'content': parsed_gpt_response
                }
            ],
            "token_usage": token_usage.model_dump(),
            "verbal_analysis": verbs_analysis
        }
    except Exception as e:
        logger.error(f"Error in get_chat_response: {str(e)}")
        raise

def get_verb_analysis(prompt):
    try:
        with open('bow.txt', 'r') as f:
            bow = f.read()
        
        response = client.chat.completions.create(
            temperature=0.8,
            response_format={ "type": "json_object" },
            model="gpt-3.5-turbo",
            messages=[
                {
                    'role': 'user',
                    'content': f"Given a text, you have to find if there is any verbs present like {bow} etc,.\ncontent text:{prompt}." + 
                    """\nUse json format to respond like following:
                    {
                        "is_verb_present": <yes/no>,
                        "verb": <The exact verb present in the sentence or none>
                    }"""
                }
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error in get_verb_analysis: {str(e)}")
        raise

def text_to_speech(text):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.write_to_file('response.mp3')
        return {
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)