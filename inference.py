from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import json
from itertools import islice
import os
import copy

from utils.gpt_utils import GPTBatcher
from utils.draft_reassign import reassign

TASK = ["general_perception", "region_perception", "driving_suggestion"]

#PATH
DATA_ROOT =  "data" 
OUTPUT_DIR = "results"   # Final output
STAGE1_OUTPUT_DIR = "results_stage1"   # save the Preliminary results of Stage 1
CHECKPOINT_DIR = "checkpoints"

# LLaVA-Next related
MAX_TOKEN = 1000
DEVICE = "cuda:0"
NUM_WORKERS = 32

'''
1. Download 'llava-hf/llava-v1.6-vicuna-7b-hf' from huggingface
2. Modify the base_model_name_or_path in the adapter_config.json in each ckpts
'''
MODEL_ID1 = f"{CHECKPOINT_DIR}/ckpt1"
MODEL_ID2 = f"{CHECKPOINT_DIR}/ckpt2"
MODEL_ID3 = f"{CHECKPOINT_DIR}/ckpt3"

# GPT-4 api 
'''
The code uses transit GPT4 api. For offical OpenAI api, the code in GPTBatcher Class may need to be modified a little.
'''
MODEL_NAME = "gpt-4o"
API_KEY =  ""
API_BASE_URL = ""
RETRY_ATTEMPTS = 16


general_perception_dataset = [json.loads(q) for q in open(os.path.expanduser(os.path.join(DATA_ROOT, "CODA-LM/Test/vqa_anno/general_perception.jsonl")), "r", encoding='utf-8')]
region_perception_dataset = [json.loads(q) for q in open(os.path.expanduser(os.path.join(DATA_ROOT, "CODA-LM/Test/vqa_anno/region_perception.jsonl")), "r", encoding='utf-8')]
driving_suggestion_dataset = [json.loads(q) for q in open(os.path.expanduser(os.path.join(DATA_ROOT, "CODA-LM/Test/vqa_anno/driving_suggestion.jsonl")), "r", encoding='utf-8')]

region_category = [json.loads(q) for q in open(os.path.expanduser(os.path.join(DATA_ROOT, "CODA-LM/Test/vqa_anno/region_perception_category.jsonl")), "r", encoding='utf-8')]

datasets = {"general_perception": general_perception_dataset,
            "region_perception": region_perception_dataset,
            "driving_suggestion": driving_suggestion_dataset}

########### Stage 1 ###########


print("Stage 1 begin")

llava_prompts = {"general_perception": "A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\n Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's color, position, status, implication, response, and how they influence ego car. EXPERT:",
                 "region_perception": "A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\Please describe the object inside the red rectangle in the image and explain why it affect ego car driving. EXPERT:",
                 "driving_suggestion": "A chat between a curious human and an autonomous driving expert, specializing in providing specific and helpful driving suggestions. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. You must not discuss any objects that not show in the image. Please provide driving suggestions for the ego car based on the current scene. EXPERT:"}

gpt_selection_prompts = {"general_perception": "Choose the answer that generally and accurately describe the objects in the image. You must choose one answer by strictly following this format: \"[[choice]]\", for example: \"Choice: [[Answer 1]]\".",
                         "region_perception": "Choose the shortest answer that only describe the object inside the red rectangle in the image among the three answers. You must choose one answer by strictly following this format: \"[[choice]]\", for example: \"Choice: [[Answer 1]]\".",
                         "driving_suggestion": "Choose the answer that only provide driving suggestions without extra description. You must choose one answer by strictly following this format: \"[[choice]]\", for example: \"Choice: [[Answer 1]]\"."}

sample_datsets = {"general_perception": general_perception_dataset[:30],
                  "region_perception": region_perception_dataset[:50],
                  "driving_suggestion": driving_suggestion_dataset[:30]}

def load_and_use_model(model_id, dataset, tasks):
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(DEVICE)

    local_dataset = copy.deepcopy(dataset)
    
    results = {}
    for task in tasks:
        answers = []
        prompt = llava_prompts[task]
        print(f"LLaVA output {task} begin")
        for item in local_dataset[task]:
            #print(f"Processing question_id: {item['question_id']} with model {model_id}")
            image = Image.open(os.path.join(DATA_ROOT, item["image"]))
            inputs = processor(prompt, image, return_tensors="pt").to(DEVICE)
            output = model.generate(**inputs, max_new_tokens=MAX_TOKEN)
            text = processor.decode(output[0], skip_special_tokens=True)
            result = text.split("EXPERT: ", 1)[1]
            item["answer"] = result
            answers.append(item)
        print(f"LLaVA output {task} end")
        results[task] = answers

    del model
    torch.cuda.empty_cache()

    return results


# LLaVA-Next output for model selection
results1 = load_and_use_model(MODEL_ID1, sample_datsets, TASK)
results2 = load_and_use_model(MODEL_ID2, sample_datsets, TASK)
results3 = load_and_use_model(MODEL_ID3, sample_datsets, TASK)


candidates = {"general_perception": [results1['general_perception'], results2['general_perception'], results3['general_perception']],
              "region_perception": [results1['region_perception'], results2['region_perception'], results3['region_perception']],
              "driving_suggestion": [results1['driving_suggestion'], results2['driving_suggestion'], results3['driving_suggestion']]}

# GPT-4 selection
selection_result = {}

for task in TASK:
    batcher = GPTBatcher(
            api_key=API_KEY, 
            system_prompt=gpt_selection_prompts[task],
            model_name=MODEL_NAME, 
            num_workers=NUM_WORKERS,
            retry_attempts=RETRY_ATTEMPTS,
            api_base_url=API_BASE_URL
        )
    candidate = candidates[task]

    rets = []
    for item1, item2, item3 in zip(candidate[0], candidate[1], candidate[2]):
        assert item1['question_id'] == item2['question_id']
        assert item2['question_id'] == item3['question_id']
        image_path = os.path.join(DATA_ROOT, item1['image']) 
        image_id = batcher.upload_image(image_path)
        
        content = [
                    {"type": "text",
                    "text": f"[The Start of Answer 1]\n{item1['answer']}\n[The End of Answer 1]\n\n[The Start of Answer 2\n{item2['answer']}\n[The End of Answer 2]\n\n[The Start of Answer 3]\n{item3['answer']}\n[The End of Answer 3]"
                    },
                    {"type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{image_id}"
                    }
                }]

        ret = batcher.create_messages(content)
        rets.append(ret)

    results = batcher.handle_message_list(rets)

    choices = {}
    for result in results:
        try:
            choice = result.split("Choice: [[")[1].split("]]")[0]
            choices[choice] = choices.get(choice, 0) + 1
        except:
            pass

    model_selected = max(choices, key=choices.get)
    selection_result[task] = model_selected

print(selection_result)


# General Stage 1 primary results

preliminary_output = {}

for task in TASK:

    if selection_result[task] == "Answer 1":
        model_id = MODEL_ID1
    elif selection_result[task] == "Answer 2":
        model_id = MODEL_ID2
    elif selection_result[task] == "Answer 3":
        model_id = MODEL_ID3
    else:
        raise ValueError("The selection result error")

    result = load_and_use_model(model_id, datasets, [task])
    preliminary_output[task] = result[task]

os.makedirs(STAGE1_OUTPUT_DIR, exist_ok=True)

for task in TASK:
    with open(f"{STAGE1_OUTPUT_DIR}/{task}_stage1.jsonl", 'w') as file:
        for entry in preliminary_output[task]:
            json_str = json.dumps(entry)
            file.write(json_str + '\n')


########### Stage 2 ###########
print("Stage 2 begin")

refine_prompts = {"general_perception": "You are an AI assistant tasked with integrating image descriptions related to autonomous driving. You will receive an image captured from autonomous driving and a draft description. Only include visible objects in the scene. Accurately describe the objects and their impact on the ego vehicle. Delete the inaccurate content. Add a comment on how much each road user affects the driving of the ego car.",
                 "region_perception": "You are an AI assistant tasked with integrating image descriptions related to autonomous driving. You will receive an image with red rectangle captured from autonomous driving and a draft description of the object in the rectangle. The task is to describe the object inside the red rectangle in the image and explain why it affect ego car. The category of the object is correct. Based on the image, modify the draft regional description. Start with \"This object is \"",
                 "driving_suggestion": "You are an AI assistant tasked with modify the driving suggestion related to autonomous driving. You will receive an image captured from autonomous driving, a general perception description and a draft driving suggestion. Accurately modify the driving suggestion based on the image and general perception."}

# Reassign the draft of region perception according to its correctness of category 
region_final_part1, region_draft = reassign(preliminary_output["region_perception"], region_category)
preliminary_output["region_perception"] = region_draft


final_output = {}

for task in TASK:
    batcher = GPTBatcher(
            api_key=API_KEY, 
            system_prompt=refine_prompts[task],
            model_name=MODEL_NAME, 
            num_workers=NUM_WORKERS,
            retry_attempts=RETRY_ATTEMPTS,
            api_base_url=API_BASE_URL
        )

    rets = []
    count = 0
    for item in preliminary_output[task]:
        image_path = os.path.join(DATA_ROOT, item["image"]) 
        image_id = batcher.upload_image(image_path)

        if task == "general_perception":
            content = [
                {"type": "text",
                 "text": f"Draft general description: {item['answer']} \n "
                },
                {"type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{image_id}"
                }
            }]
        
        elif task == "driving_suggestion":
            assert final_output["general_perception"][count]["question_id"] == item["question_id"]
            content = [
                    {"type": "text",
                    "text": f"General description: {final_output['general_perception'][count]['answer']}\n.Draft driving suggestion: {item['answer']}"
                    },
                    {"type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{image_id}"
                    }
                }]

        else:
            content = [
                {"type": "text",
                 "text": f"Draft region description: {item['answer']} \n "
                },
                {"type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{image_id}"
                }
            }]

        ret = batcher.create_messages(content)
        rets.append(ret)
        count += 1

    results = batcher.handle_message_list(rets)
    
    task_final_output = []
    for item, answer in zip(preliminary_output[task], results):
        item_ = copy.deepcopy(item)
        item_["answer"] = answer
        task_final_output.append(item_)
    final_output[task] = task_final_output


region_final_part2 = final_output["region_perception"]
region_final = region_final_part1 + region_final_part2
region_final.sort(key=lambda x: x['question_id'])
final_output["region_perception"] = region_final

os.makedirs(OUTPUT_DIR, exist_ok=True)

for task in TASK:
    with open(f"{OUTPUT_DIR}/{task}_answer.jsonl", 'w') as file:
        for entry in final_output[task]:
            if task == "region_perception":
                del entry["category"]
            json_str = json.dumps(entry)
            file.write(json_str + '\n')