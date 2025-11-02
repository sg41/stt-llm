import speech_recognition as sr
import numpy as np
from transformers import pipeline
import requests
import sys, os

# === Настройки ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"  # или другая модель

DEVICE_INDEX = 0  # Индекс микрофона (у вас — 0)

# === Инициализация ===
print("Загрузка Whisper...")
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=-1,  # CPU
    return_timestamps=False,
    generate_kwargs={"language": "russian", "task": "transcribe"}
)

r = sr.Recognizer()
r.energy_threshold = 400
r.dynamic_energy_threshold = True

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://your-app.local",
    "X-Title": "Voice-to-LLM"
}

def audio_to_numpy(audio_data, target_rate=16000):
    """Преобразует AudioData в numpy массив для Whisper"""
    raw = audio_data.get_wav_data(convert_rate=target_rate, convert_width=2)
    audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np

def transcribe_with_whisper(audio_data):
    try:
        audio_np = audio_to_numpy(audio_data)
        result = transcriber(audio_np)
        return result['text'].strip()
    except Exception as e:
        print(f"❌ Ошибка Whisper: {e}")
        return ""

def query_openrouter(prompt):
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            print(f"❌ OpenRouter ошибка {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Сетевая ошибка: {e}")
        return None

# === Основной цикл ===
def main():
    print("\n🎙️  Голосовой ассистент готов. Говорите — запись остановится автоматически.")
    print("   Нажмите Ctrl+C для выхода.\n")

    first_run = True
    with sr.Microphone(device_index=DEVICE_INDEX) as source:
        while True:
            try:
                print("👂 Слушаю...")
                audio = r.listen(source, timeout=10)
                print("✅ Запись завершена. Распознаю...")

                user_text = transcribe_with_whisper(audio)
                if not user_text:
                    print("⚠️  Не удалось распознать речь.\n")
                    continue

                print(f"💬 Вы сказали: {user_text}")

                if first_run:
                    print("🤖 Skip fist run")
                    first_run = False
                    continue

                print("🧠 Отправляю в LLM...")
                llm_response = query_openrouter(user_text)
                if llm_response:
                    print(f"🤖 Ответ:\n{llm_response}\n")
                else:
                    print("⚠️  Не удалось получить ответ от модели.\n")

            except sr.WaitTimeoutError:
                print("⏳ Таймаут: никто не говорит.\n")
            except KeyboardInterrupt:
                print("\n👋 Выход.")
                sys.exit(0)
            except Exception as e:
                print(f"💥 Неожиданная ошибка: {e}\n")

if __name__ == "__main__":
    main()