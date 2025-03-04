from dotenv import load_dotenv
import os
from google import genai
from functools import cache
import json
from os import listdir
from pathlib import Path

from .utils import get_llm_data_folder

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
    
def save_json_result(json_data):
    """
    saves my boi as json
    """

    name = "llm_data"
    save_folder = get_llm_data_folder()
    id = len(listdir(save_folder))
    save_path = save_folder.joinpath(f"{id}_{name}")

    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def adjust_coordinates(data, delta_x=0, delta_y=0):
    """
    Adjust x and y coordinates in the given data structure by adding or subtracting constants.
    
    Parameters:
    - data: List of dictionaries or JSON string containing coordinate_list with x and y values
    - delta_x: Value to add to all x coordinates (negative for subtraction)
    - delta_y: Value to add to all y coordinates (negative for subtraction)
    
    Returns:
    - JSON string with adjusted coordinates, ready to be saved to a file
    """
    # Handle JSON string input
    if isinstance(data, str):
        data = json.loads(data)
    
    adjusted_data = []
    max_rad = 2
    
    for item in data:
        # Create a new item dictionary instead of using copy()
        adjusted_item = {
            "text_prompt": item["text_prompt"],
            "coordinate_list": []
        }
        
        # Add any other fields that might exist in the original item
        for key, value in item.items():
            if key not in ["text_prompt", "coordinate_list"]:
                adjusted_item[key] = value
        
        # Process each coordinate
        for coord in item["coordinate_list"]:
            # Create a new coordinate dictionary with adjusted x and y values

            original_x = coord["x"]
            original_y = coord["y"]

            # lets not run into singularities... 
            if original_x == 0:
                original_x += .01
            if original_y == 0:
                original_y += .01

            # if the LLM is stupid, we scale values. up to 20% scaled down.
            depth = 0
            while depth < 10:
                if original_x**2 + original_y**2 > max_rad**2:
                    original_x *= .98
                    original_y *= .98
                else:
                    break
                
                depth += 1

            adjusted_coord = {
                "x": original_x + delta_x,
                "y": original_y + delta_y,
                "speed": coord["speed"]
            }
            
            # Add any other fields that might exist in the original coordinate
            for key, value in coord.items():
                if key not in ["x", "y", "speed"]:
                    adjusted_coord[key] = value
                    
            adjusted_item["coordinate_list"].append(adjusted_coord)
        
        adjusted_data.append(adjusted_item)
    
    # Return the result as a JSON string
    return json.dumps(adjusted_data, indent=2)

def load_all_llm_data():
    """
    Load JSON data from multiple files into a list of individual prompts
    
    Returns:
        list: A list of all movement prompts from all files
    """
    data_path = get_llm_data_folder()
    name = "_llm_data"
    full_prompt_list = []

    for file in data_path.iterdir():

        if file.is_file() and name in str(file):  # TODO: THis is garbage
            try:
                with open(file, 'r') as file:
                    content = file.read()
                    prompts = split_into_prompts(content)
                    full_prompt_list += prompts
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in file {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    return full_prompt_list

def split_into_prompts(data):
    """
    Takes a JSON string, file content, or parsed data containing prompt objects
    and splits them into individual prompts.
    
    Args:
        data: JSON string, file content, or parsed data
        
    Returns:
        list: A list of individual prompt objects
    """
    prompt_list = []
    
    # Handle the case where data is a string (file content)
    if isinstance(data, str):
        try:
            # Try to parse the string as JSON
            parsed_data = json.loads(data)
            # Now we have parsed JSON, continue with processing
            return split_into_prompts(parsed_data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return prompt_list
    
    # At this point, data should be a parsed JSON object
    if isinstance(data, list):
        # Process each item in the list
        for item in data:
            # Check if this is a valid prompt object
            if isinstance(item, dict) and "text_prompt" in item and "coordinate_list" in item:
                # This is a valid prompt object, add it to our list
                prompt_list.append(item)
            elif isinstance(item, list):
                # Recursively handle nested lists
                prompt_list += split_into_prompts(item)
    elif isinstance(data, dict):
        # Check if this single dictionary is a valid prompt object
        if "text_prompt" in data and "coordinate_list" in data:
            prompt_list.append(data)
    
    return prompt_list

def gather_llm_responses():

    # get the llm to do some cool stuff
    prompt = load_prompt()
    response = get_response(prompt)
    json_only = extract_json_from_llm_response(response)

    # edit it because llm generates better data being centered at 0,0
    adjusted_json = adjust_coordinates(json_only, 2.1, 2.1)
    save_json_result(adjusted_json) # save it because I want all of it!


if __name__ == "__main__":

    gather_llm_responses()

