# ECCV2024_Challenge_llmforad_solution

## File Organization
```
├── checkpoints
   ├──ckpt1
   ├──ckpt2
   ├──ckpt3
├── results   --final output
├── results_stage1
├── data
   ├── val
   │   │── images
   │   │── images_w_bboxes                
   │   │   │── *.jpg
   ├── test
   │   │── images
   │   │── images_w_bboxes               
   │   │   │── *.jpg
   ├── CODA-LM
   │   │── Test
   │   │   │── vqa_anno
   │   │   │   │── general_perception.jsonl 
   │   │   │   │── region_perception.jsonl   
   │   │   │   │── region_perception_category.jsonl   -- GT category annotations for region perception
   │   │   │   │── driving_suggestion.jsonl  
   │   │── Val
   │   │   │── vqa_anno
   │   │── Train
   │   │   │── vqa_anno
```


## Train: Stage 1 Finetune
```bash
python llava-next_finetune.py
```
The base LLaVA-Next model is llava-v1.6-vicuna-7b-hf, that can be downloaded from https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf

## Inference
```bash
python inference.py
```
For inference, there are some notes:

1. Modify the base_model_name_or_path in the adapter_config.json in each ckpts
2. The code runs with transit GPT4 api. For offical OpenAI api, the code in GPTBatcher Class may need to be modified for a small portion.
