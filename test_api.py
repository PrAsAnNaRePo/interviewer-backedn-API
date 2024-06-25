import requests
import base64

def test_talk():
    url = "http://0.0.0.0:8000/inter-val"

    with open('alloy.wav', 'rb') as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

    bot_config = {
        "name": "Alice",
        "personality": "Kind and Brisk",
        "major_questions": [
            {
                "question": "Tell me about an instance when you made yourself available to a colleague.",
                "difficulity": "medium"
            },
            {
                "question": "Describe a time when you demonstrated your ability to actively listen.",
                "difficulity": "medium"
            },
            {
                "question": "Share an example of when you showcased your analytical skills.",
                "difficulity": "hard"
            }
        ],
        "response_type": "concise",
        "temperature": 1.0,
        "message_history": [
            {
                "role": "user",
                "content": "Hey!!"
            },
            {
                "role": "assistant",
                "content": "Hello, Can we just start the interview?"
            }
        ],
        "audio_base64": audio_base64
    }

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, json=bot_config, headers=headers)
    print(response.status_code)
    print(response.json())


# -----------------------------------------

def test_init_conv():
    url = "http://0.0.0.0:8000/init-conv"
    response = requests.post(url)
    print(response.text)

test_talk()
test_init_conv()
