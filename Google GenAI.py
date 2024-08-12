import os
import json
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, GenerateContentResponse
from typing import List, Union, Iterator, Optional, Dict

# Detaylı günlük kaydını etkinleştirmek için DEBUG'u True olarak ayarlayın
DEBUG = False


class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(default="")
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)
        DEFAULT_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=1.0)
        DEFAULT_TOP_P: float = Field(default=0.9, ge=0.0, le=1.0)
        DEFAULT_TOP_K: int = Field(default=40, ge=1)
        DEFAULT_MAX_TOKENS: int = Field(default=8192, ge=1)
        DEFAULT_SAFETY_SETTINGS: Optional[Dict[str, str]] = Field(default=None)
        DEFAULT_SYSTEM_MESSAGE: Optional[str] = Field(default=None)

    def __init__(self):
        self.id = "google_genai"
        self.type = "manifold"
        self.name = "Google: "
        self.valves = self.Valves(
            **{
                "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
                "USE_PERMISSIVE_SAFETY": False,
                "DEFAULT_TEMPERATURE": 0.7,
                "DEFAULT_TOP_P": 0.9,
                "DEFAULT_TOP_K": 40,
                "DEFAULT_MAX_TOKENS": 8192,
                "DEFAULT_SAFETY_SETTINGS": None,
                "DEFAULT_SYSTEM_MESSAGE": "You are a helpful AI assistant.",
            }
        )

    def get_google_models(self):
        if not self.valves.GOOGLE_API_KEY:
            return [
                {
                    "id": "error",
                    "name": "GOOGLE_API_KEY ayarlanmamış. Lütfen valvlerdeki API Anahtarını güncelleyin.",
                }
            ]
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            models = genai.list_models()
            return [
                {
                    "id": model.name[7:],  # "models/" kısmını kaldır
                    "name": model.display_name,
                }
                for model in models
                if "generateContent" in model.supported_generation_methods
                if model.name.startswith("models/")
            ]
        except Exception as e:
            if DEBUG:
                print(f"Google modellerini getirirken hata oluştu: {e}")
            return [{"id": "error", "name": f"Google'dan modeller alınamadı: {str(e)}"}]

    def pipes(self) -> List[dict]:
        return self.get_google_models()

    def get_safety_settings(self, body: dict) -> Dict[str, str]:
        if self.valves.USE_PERMISSIVE_SAFETY:
            return {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
        return body.get("safety_settings", self.valves.DEFAULT_SAFETY_SETTINGS or {})

    def prepare_contents(
        self, messages: List[dict], system_message: Optional[str]
    ) -> List[dict]:
        contents = []
        for message in messages:
            if message["role"] != "system":
                if isinstance(message.get("content"), list):
                    parts = []
                    for content in message["content"]:
                        if content["type"] == "text":
                            parts.append({"text": content["text"]})
                        elif content["type"] == "image_url":
                            image_url = content["image_url"]["url"]
                            if image_url.startswith("data:image"):
                                image_data = image_url.split(",")[1]
                                parts.append(
                                    {
                                        "inline_data": {
                                            "mime_type": "image/jpeg",
                                            "data": image_data,
                                        }
                                    }
                                )
                            else:
                                parts.append({"image_url": image_url})
                    contents.append({"role": message["role"], "parts": parts})
                else:
                    contents.append(
                        {
                            "role": "user" if message["role"] == "user" else "model",
                            "parts": [{"text": message["content"]}],
                        }
                    )

        if system_message:
            contents.insert(
                0,
                {"role": "user", "parts": [{"text": f"System: {system_message}"}]},
            )

        return contents

    def pipe(self, body: dict) -> Union[str, Iterator[str]]:
        if not self.valves.GOOGLE_API_KEY:
            return "Hata: GOOGLE_API_KEY ayarlanmamış"
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)
            model_id = body["model"]

            if model_id.startswith("google_genai."):
                model_id = model_id[12:]

            model_id = model_id.lstrip(".")

            if not model_id.startswith("gemini-"):
                return f"Hata: Geçersiz model adı formatı: {model_id}"

            messages = body["messages"]
            stream = body.get("stream", False)

            if DEBUG:
                print("Gelen body:", str(body))

            system_message = next(
                (msg["content"] for msg in messages if msg["role"] == "system"),
                self.valves.DEFAULT_SYSTEM_MESSAGE,
            )

            contents = self.prepare_contents(messages, system_message)

            model = genai.GenerativeModel(
                model_name=model_id,
                generation_config=GenerationConfig(
                    temperature=body.get(
                        "temperature", self.valves.DEFAULT_TEMPERATURE
                    ),
                    top_p=body.get("top_p", self.valves.DEFAULT_TOP_P),
                    top_k=body.get("top_k", self.valves.DEFAULT_TOP_K),
                    max_output_tokens=body.get(
                        "max_tokens", self.valves.DEFAULT_MAX_TOKENS
                    ),
                    stop_sequences=body.get("stop", []),
                ),
            )

            safety_settings = self.get_safety_settings(body)

            if DEBUG:
                print("Google API isteği:")
                print("  Model:", model_id)
                print("  İçerikler:", str(contents))
                print("  Üretim Yapılandırması:", model.generation_config)
                print("  Güvenlik Ayarları:", safety_settings)
                print("  Akış:", stream)

            if stream:

                def stream_generator():
                    response = model.generate_content(
                        contents,
                        safety_settings=safety_settings,
                        stream=True,
                    )
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text

                return stream_generator()
            else:
                response = model.generate_content(
                    contents,
                    safety_settings=safety_settings,
                    stream=False,
                )
                return response.text
        except Exception as e:
            if DEBUG:
                print(f"pipe metodunda hata: {e}")
            return f"Hata: {e}"
