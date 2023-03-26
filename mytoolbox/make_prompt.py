"""Generate the prompt according to the usage."""
from pathlib import Path

TRANSLATOR_PROMPT_PATH = Path(__file__).parent / "prompts" / "translator.txt"

def translate(text:str, lang_code:str="zh-Hant", text_engine:str="gpt-3.5-turbo"):
    """Translate the text to the given language code.
    
    Args:
        text (str): The text to translate.
        lang_code (str): The language code to translate to.
             we use ISO 639-1 language code: https://www.w3schools.com/tags/ref_language_codes.asp
    Returns:
    """
    # Load the prompt
    with open(TRANSLATOR_PROMPT_PATH, "r") as f:
        prompt = f.read()
    # Replace the language code and the text
    prompt = prompt.replace("<lang_code>", lang_code)
    prompt = prompt.replace("<input_text>", text)

    if "text-" in text_engine:
        return  {
            "prompt": prompt,
        }
    elif "gpt-" in text_engine:
        return {
                "messages":[
                {"role": "user", "content": prompt}
                ]
            }   
    else:
        raise ValueError(f"Unsupported text engine: {text_engine}")
