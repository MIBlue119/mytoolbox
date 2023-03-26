"""Script to translate pdf to desired language
Ref: https://github.com/ywchiu/largitdata/blob/master/code/Course_222.ipynb
"""
import os
import argparse

from nltk.tokenize import sent_tokenize
from pypdf import PdfFileReader, PdfFileWriter
from loguru import logger
import openai

from mytoolbox import make_prompt 
from mytoolbox.utils import (get_model_selection,
                            parse_text_response,
                            generate_openai_completion)

def generate_chunks(sentences, chunk_size:int=1000):
    """Generate chunks of sentences.
    
    Args:
        sentences (list): A list of sentences.
        chunk_size (int): The maximum number of characters in a chunk.
    Returns:
        chunks (list): A list of chunks.
    """
    chunks = []
    input_sentences = ''
    for sentence in sentences:
        input_sentences += sentence
        if len(input_sentences) > chunk_size:
            chunks.append(input_sentences)
            input_sentences = ''
    chunks.append(input_sentences)
    return chunks

def translate_chunk(chunk_text, lang_code:str="zh-Hant", text_engine: str = "gpt-3.5-turbo"):
    """Translate a chunk of text."""
    translation_prompt = make_prompt.translate(chunk_text, lang_code, text_engine)
    api_settings = {
                **get_model_selection(text_engine),
                **translation_prompt,
                "n": 1, 
                "max_tokens" : 2048,               
                "temperature": 0.2,
                "presence_penalty" : 2
    }
    response = generate_openai_completion(text_engine=text_engine, api_settings=api_settings)
    translated_text =  parse_text_response(response, text_engine=text_engine)
    return translated_text


def app(args):
    """Translate a pdf file to a desired language."""

    file_path = args.file_path
    lang_code = args.lang_code
    text_engine = args.text_engine
    openai_api_key = os.environ.get("OPENAI_API_KEY",  args.openai_api_key)
    if openai_api_key is None:
        raise ValueError("Please provide the OpenAI API key.")
    openai.api_key = openai_api_key

    reader = PdfFileReader(file_path)
    number_of_pages = len(reader.pages)
    logger.info(f"Number of pages: {number_of_pages}")

    # Get the text from each page
    text_dict = {}
    for i in range(number_of_pages):
        logger.info(f"Page {i}")
        page = reader.pages[i]
        text = page.extractText()
        sentences = sent_tokenize(text)
        chunks = generate_chunks(sentences)
        text_dict[i] = chunks
    
    # Translate the text
    translated_text_dict = {}
    for page, chunks in text_dict.items():
        translated_text_dict[page] = []
        for chunk in chunks:
            translated_text = translate_chunk(chunk, lang_code, text_engine)
            translated_text_dict[page].append(translated_text)
    
    # Write the translated text to a pdf file
    writer = PdfFileWriter()
    for page, chunks in translated_text_dict.items():
        for chunk in chunks:
            writer.addPage(chunk)
    with open("translated.pdf", "wb") as f:
        writer.write(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", dest="file_path", type=str, help="Path to the pdf file to be translated.")
    parser.add_argument("--lang_code", dest="lang_code", type=str, default="zh-Hant", help="The language code to translate to.")
    parser.add_argument("--text_engine", dest="text_engine", type=str, default="gpt-3.5-turbo", help="The text engine to use.")
    parser.add_argument("--openai_api_key", dest="openai_api_key", type=str, default=None, help="The OpenAI API key.")
    args = parser.parse_args()
    app(args)