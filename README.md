# ECCV2024_Challenge_llmforad_solution

## Dataset Organization
```
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

## Inference
```bash
python inference.py
```
