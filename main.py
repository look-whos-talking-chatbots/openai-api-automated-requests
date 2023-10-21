"""Script to send multiple automated requests to OpenAI API.

Author: Erkan Basar <erkan.basar@ru.nl>
"""

import os
import csv
import yaml
import json
import time
import openai
import backoff
from datetime import datetime


def load_csv_to_json(csv_file_path):
    """Loads a CSV formatted file as a list of JSON objects.

    Each JSON object corresponds to a single row within the CSV file.
    The function uses the header of the CSV file as the keys of the JSON objects.

    Parameters
    ----------
    csv_file_path: str
        The folder path to the CSV file to be loaded.

    Returns
    -------
    data: list
        The list of JSON objects loaded from the file.
    """
    data = []
    with open(csv_file_path, newline="") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        for row in csv_reader:
            # Iterate through columns
            for key, value in row.items():
                try:
                    # Try to convert the value to an integer
                    row[key] = int(value)
                except (ValueError, TypeError):
                    # If conversion fails, keep the original value
                    pass
            data.append(row)
    return data


def load_jsonlines(data_path):
    """Loads a JSON-Lines formatted file as a list of JSON objects.

    Parameters
    ----------
    data_path: str
        The folder path to the CSV file to be loaded.

    Returns
    -------
    list of objects
    """
    with open(data_path, 'r') as js_file:
        return [json.loads(line) for line in js_file]


def append_jsonlines(data, file_path):
    """Appends a given dataset to a JSON-Lines file.

    Parameters
    ----------
    data: list
        The data to be appended to the targeted file.
    file_path: str
        The folder path to the file which the data should be appended.
    """
    with open(file_path, 'a') as js_file:
        for d in data:
            json.dump(d, js_file)
            js_file.write('\n')


def save_json_to_csv(json_data, file_path):
    """Saves a list of JSON objects to a CSV formatted file.

    Each JSON object corresponds to a single row within the CSV file.
    The function uses the keys of the JSON objects as the header of the CSV file.

    Parameters
    ----------
    json_data: list
        The data to be saved into the targeted CSV file.
    file_path: str
        The folder path to the file to which the data should be saved.
    """
    with open(file_path, "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=json_data[0].keys(), delimiter=';')
        csv_writer.writeheader()
        csv_writer.writerows(json_data)


def convert_data_to_csv(generated_jsonlines_path):
    """Loads the JSON-Lines data from file system and saves it to a CSV file.

    This function is specifically designed to read the generation data, which is saved as JSON-Lines,
    from the file system. It then converts this data into a CSV file with the same name for human readability.
    The metadata contained within the JSON objects is ignored and not saved to the CSV file.

    Parameters
    ----------
    generated_jsonlines_path: str
        The folder path to the JSON-Lines file that should be loaded and converted.
    """
    base, _ = os.path.splitext(generated_jsonlines_path)
    csv_output_path = f"{base}.csv"
    generated = load_jsonlines(generated_jsonlines_path)
    # Remove 'metadata' key from the generated data
    generated = [{k: v for k, v in gen.items() if k != 'metadata'}
                 for gen in generated]
    if not os.path.exists(csv_output_path):
        save_json_to_csv(generated, csv_output_path)
    else:
        print(f"WARNING: Generated csv file already exists at \"{csv_output_path}\".")


def load_config(file_path='./config.yml'):
    """Loads the data used to configure the parameters of the script from a file formatted in YAML.

    Parameters
    ----------
    file_path: str, optional (default='./config.yml')
        The folder path to the configuration file to be loaded.

    Returns
    -------
    input_path_conf: str
        The folder path to the input file that contains the prompts.
    output_path_conf: str
        The folder path to the file where the generated output data should be saved.
    openai_api_key: str
        The secret key that authorizes connection to the OpenAI API.
    max_tokens_conf: int
        The maximum number of tokens that the models are allowed to generate.
    requests_per_min_conf: int
        The number of requests that will be sent to the API per minute.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    paths = data.get('paths', {})

    input_path_conf = paths.get('input_file', 'prompts.csv')
    print(f"INFO: The expected input file path is \"{input_path_conf}\"")

    output_path_conf = paths.get('output_file', 'generations.jsonl')
    print(f"INFO: The output file path is set to \"{output_path_conf}\"")

    params = data.get('params', {})

    max_tokens_conf = params.get('max_tokens', 200)
    print(f"INFO: Maximum tokens to be generated is set to {max_tokens_conf}.")

    requests_per_min_conf = params.get('  requests_per_min', 30)
    print(f"INFO: The number of requests to API per minute is {requests_per_min_conf}.")

    try:
        openai_api_key = data['api_keys']['openai']
    except KeyError:
        raise KeyError("The keys \"api_keys\" and/or \"openai\" do not exist in the config file.")

    return input_path_conf, output_path_conf, openai_api_key, max_tokens_conf, requests_per_min_conf


def load_prompts(input_file_path):
    """Loads the data that includes both the prompts and the configurations used for the text generation process.

    The function checks for the existence of a progress file every time it runs. If the progress file is found,
    it is loaded instead of the original input file. This ensures that the script can keep track of its progress
    and avoid sending redundant requests.

    Parameters
    ----------
    input_file_path: str
        The folder path to the file that contains the prompts.

    Returns
    -------
    data: list
        The prompt and configuration data loaded from the input file.
    file_path: str
        The created folder path to the progress file.
    """
    base, extension = os.path.splitext(input_file_path)
    progress_file_path = f"{base}_progress{extension}"

    if os.path.exists(progress_file_path):
        input_file_path = progress_file_path
        print(f"INFO: Progress file found. Using the following as the input file: \"{progress_file_path}\"")

    return load_csv_to_json(input_file_path), progress_file_path


def generate(prompt_data, max_tokens=200):
    """Sends the prompts to the API and retrieves the results of the text generation process.

    The function utilizes the backoff package to handle potential rate limit errors.
    It attempts the request three times before finally throwing an error.

    Parameters
    ----------
    prompt_data: dict
        A single prompt and configuration data
    max_tokens: int
        The maximum number of tokens that the models are allowed to generate.

    Returns
    -------
    data: dict
        The data contains the result of the text generation and further information.
    """
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=3, factor=3)
    def generation_request(**kwargs):
        return openai.ChatCompletion.create(**kwargs)

    messages = []
    if 'system' in prompt_data and prompt_data['system']:
        messages.append({
            'role': 'system',
            'content': prompt_data['system']
        })
    messages.append({
        'role': 'user',
        'content': prompt_data['prompt']
    })

    if 'model' in prompt_data and prompt_data['model']:
        model = 'gpt-4' if prompt_data['model'].lower() == 'gpt-4' else 'gpt-3.5-turbo'
    else:
        model = 'gpt-3.5-turbo'

    try:
        response = generation_request(messages=messages, model=model, max_tokens=max_tokens)
    except openai.error.RateLimitError:
        raise openai.error.RateLimitError("OpenAI API rate limit exceeded.")

    return {
        'prompt': prompt_data['prompt'],
        'generated_text': response['choices'][0]['message']['content'],
        'system': prompt_data['system'] if 'system' in prompt_data else '',
        'generated_tokens': response['usage']['completion_tokens'],
        'timestamp': datetime.utcfromtimestamp(response['created']).isoformat(),
        'model': response['model'],
        'metadata': response
    }


if __name__ == "__main__":

    # Load the necessary configs from the configs file
    input_path, output_path, openai.api_key, max_tokens, requests_per_min = load_config('./config.yaml')

    # Load the prompts from the input file (or from the progress file).
    prompts, progress_path = load_prompts(input_path)

    print("\n== GENERATION REQUESTS ==")

    for prompt in prompts:

        print(f"\nRunning the prompt: \"{prompt['prompt']}\"")

        print(f"Iterations left: {str(prompt['iterations'])}")

        while prompt['iterations']:
            # Process the prompt data and send a request to API
            result = generate(prompt, max_tokens)

            # Append the generated data to the output file
            append_jsonlines([result], output_path)

            # Subtract from the number of iterations to keep the progress of the prompts
            prompt['iterations'] -= 1

            # Save the progress of the requests in a new file
            save_json_to_csv(prompts, progress_path)

            # Wait between each request
            time.sleep(60.0 / requests_per_min)

            print(f"Iterations left: {str(prompt['iterations'])}")

    print("\n== END OF GENERATION ==\n")

    # Convert the resulted data to a CSV file.
    convert_data_to_csv(output_path)
