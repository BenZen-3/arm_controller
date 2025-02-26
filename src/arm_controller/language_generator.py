from dotenv import load_dotenv
import os
from google import genai
from functools import cache
import json

@cache
def load_API_key():
    """
    yeah its the name
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    return api_key

def load_prompt(name = "prompt.txt"):
    """
    Load the prompt from a file named 'prompt.txt' in the same directory as the script.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, name)

    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt = file.read().strip()
        return prompt
    except FileNotFoundError:
        print(f"Error: '{name}' not found.")
        return None
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return None

def get_response(prompt):
    """
    duh
    """

    assert isinstance(prompt, str) # let's not waste precious responses
    key = load_API_key()

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    return response.text

def is_json(content: str):
    """yup its in the name"""
    
    try: 
        json.loads(content)
    except Exception as e:
        return False
    
    return True

def extract_json_from_llm_response(llm_response: str):
    """gimme that json"""

    if llm_response.startswith("```json") and llm_response.endswith("```"):
        json_content = llm_response[7:-3].strip()  # Remove ```json and ``` in an incredibly lame and stupid way!

        if is_json(json_content):
            return json_content

    raise ValueError(f"Invalid LLM response format. Exact response: {llm_response}")
    
def save_json_result(json):
    """
    saves my boi as json
    """

    print(json)

    raise NotImplementedError

if __name__ == "__main__":

    prompt = load_prompt()
    response = get_response(prompt)
    print(response)
    
    json_only = extract_json_from_llm_response(response)

    print(json_only)



    # if is_json:
    #     save_json_result(response)
    # else:
    #     raise Exception("It was not json :(")
