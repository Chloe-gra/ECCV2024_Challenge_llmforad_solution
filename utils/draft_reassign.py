import json
import random
import os

def reassign(draft, category):
    wrong = []
    right = []
    for item, cat in zip(draft, category):
        assert item["question_id"] == cat["question_id"]
        gt_cat = cat["label_name"].replace("_", " ")
        if gt_cat == "traffic sign":
            gt_cat = "sign"
        elif gt_cat == "traffic cone":
            gt_cat = "cone"
        elif gt_cat == "car":
            gt_cat = " car "

        if gt_cat == "misc":
            right.append({"question_id": item["question_id"], "image": item["image"], "question": item["question"], "category": gt_cat, "answer": item["answer"]})
        elif gt_cat == "moped" and "moped" not in item["answer"][:200] and "motor vehicle" not in item["answer"][:200]:
            wrong.append({"question_id": item["question_id"], "image": item["image"], "question": item["question"], "category": gt_cat, "answer": item["answer"]})
        elif gt_cat not in item["answer"][:200]:
            wrong.append({"question_id": item["question_id"], "image": item["image"], "question": item["question"], "category": gt_cat, "answer": item["answer"]})
        else:
            right.append({"question_id": item["question_id"], "image": item["image"], "question": item["question"], "category": gt_cat, "answer": item["answer"]})

    reassigned_draft = []
    for wrong_item in wrong:
        gt_label = wrong_item["category"]
        drafts = list(filter(lambda item: item["category"] == gt_label, right))
        if drafts:
            random_item = random.choice(drafts)
            wrong_item["answer"] = random_item["answer"]
            reassigned_draft.append(wrong_item)
        else:
            wrong_item["answer"] = f"This is a {gt_label}"
            reassigned_draft.append(wrong_item)

    return right, reassigned_draft

