from deep_translator import GoogleTranslator
from langdetect import detect


SUPPORTED_LANGUAGES = {
    "en": "english",
    "hi": "hindi",
    "mr": "marathi",
}


class TranslatorService:
    """
    Handles language detection and translation.
    """

    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect language of user query.
        """

        try:
            language = detect(text)

            if language in SUPPORTED_LANGUAGES:
                return language

            return "en"

        except Exception:
            return "en"

    @staticmethod
    def translate_to_english(
        text: str,
        source_language: str,
    ) -> str:

        if source_language == "en":
            return text

        return GoogleTranslator(
            source=source_language,
            target="en",
        ).translate(text)

    @staticmethod
    def translate_from_english(
        text: str,
        target_language: str,
    ) -> str:

        if target_language == "en":
            return text

        return GoogleTranslator(
            source="en",
            target=target_language,
        ).translate(text)