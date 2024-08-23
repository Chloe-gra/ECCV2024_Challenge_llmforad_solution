from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from tqdm import tqdm
import time
import os
import json
import argparse
import requests
import base64


class GPTBatcher:
 
    def __init__(self, 
                 api_key, 
                 model_name="gpt-4", 
                 system_prompt="",
                 temperature=1, 
                 num_workers=32,
                 timeout_duration=60,
                 retry_attempts=2,
                 api_base_url=None):
        
        self.headers = {'Authorization': f'Bearer {api_key}',
                        'Content-Type':'application/json'}
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.num_workers = num_workers
        self.timeout_duration = timeout_duration
        self.retry_attempts = retry_attempts
        self.miss_index =[]
        self.api_base_url = api_base_url

    def upload_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def create_messages(self, content):
        ret = []
        ret.append({
            "role": "system",
            "content": self.system_prompt
        })
        ret.append({
            "role": "user", 
            "content": content
        })
        return ret

    def get_attitude(self, ask_text):
        index, ask_text = ask_text
        try:
            payload ={
                'model': self.model_name,
                'messages': ask_text,
                'temperature': self.temperature
            }
            completion = requests.post(self.api_base_url, headers=self.headers, data=json.dumps(payload))
            data = completion.json()
            return (index, data['choices'][0]['message']['content'])
        except Exception as e:
            print(f"Error occurred: {e}")
            self.miss_index.append(index)
            return (index, None)

    def process_attitude(self, message_list):
        new_list = []
        num_workers = self.num_workers
        timeout_duration = self.timeout_duration
        retry_attempts = self.retry_attempts
    
        executor = ThreadPoolExecutor(max_workers=num_workers)
        message_chunks = list(self.chunk_list(message_list, num_workers))
        try:
            for chunk in tqdm(message_chunks, desc="Processing messages"):
                future_to_message = {executor.submit(self.get_attitude, message): message for message in chunk}
                for _ in range(retry_attempts):
                    done, not_done = wait(future_to_message.keys(), timeout=timeout_duration)
                    for future in not_done:
                        future.cancel()
                    new_list.extend(future.result() for future in done if future.done())
                    if len(not_done) == 0:
                        break
                    future_to_message = {executor.submit(self.get_attitude, future_to_message[future]): future for future in not_done}
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            executor.shutdown(wait=False)
            return new_list

    def complete_attitude_list(self, attitude_list, max_length):
        completed_list = []
        current_index = 0
        for item in attitude_list:
            index, value = item
            while current_index < index:
                completed_list.append((current_index, None))
                current_index += 1
            completed_list.append(item)
            current_index = index + 1
        while current_index < max_length:
            print("Filling in missing index", current_index)
            self.miss_index.append(current_index)
            completed_list.append((current_index, None))
            current_index += 1
        return completed_list

    def chunk_list(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def handle_message_list(self, message_list):
        indexed_list = [(index, data) for index, data in enumerate(message_list)]
        max_length = len(indexed_list)
        attitude_list = self.process_attitude(indexed_list)
        attitude_list.sort(key=lambda x: x[0])
        attitude_list = self.complete_attitude_list(attitude_list, max_length)
        attitude_list = [x[1] for x in attitude_list]
        return attitude_list
    
    def get_miss_index(self):
        return self.miss_index


