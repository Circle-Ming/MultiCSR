import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import csv
import os

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--model_name", type=str, default="google/flan-t5-xl")

parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=0)

parser.add_argument("--source", type=str, default="flan_nli")
parser.add_argument("--target", type=str, default="0")
parser.add_argument("--score", type=str, default="flan_nli")

parser.add_argument("--ngram", type=int, default=0)

parser.add_argument("--filter_type", type=int, default=15)
parser.add_argument("--positive_line", type=float, default=3.0)
parser.add_argument("--negative_line", type=float, default=3.0)
parser.add_argument("--similarity_gap", type=int, default=1)

args = parser.parse_args()
print(args)

if args.score == "0":
    score_filename = "data/score_{}_for_simcse.csv".format(args.source)
else:
    score_filename = "data/{}_for_simcse.csv".format(args.score)
is_score_filename = os.path.exists(score_filename)


if is_score_filename:
    csvfile = open(score_filename, 'r')
else:
    print("Scoring mode")
    filename = "data/{}_for_simcse.csv".format(args.source)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    csvfile = open(filename, 'r')

sentence_reader = csv.reader(csvfile)
dataset = []
for row in sentence_reader:
    if "sent0" in row and "sent1" in row and "hard_neg" in row: 
        if is_score_filename:
            assert len(row) == 5
        continue
    dataset.append(row)
print(f"dataset with {len(dataset)} sentence triplets loaded")

if args.start + args.end != 0:
    start = args.start
    end = min(len(dataset), args.end)
else:
    start = 0
    end = len(dataset)

if not is_score_filename:
    if args.start + args.end != 0:
        score_filename = "data/score_{}_{}-{}_for_simcse.csv".format(args.source, start, end)
    score_file = open(score_filename, 'w')
    score_writer = csv.writer(score_file)
    score_writer.writerow(["sent0", "sent1", "hard_neg", "positive_score", "negative_score"])

if args.target != "0":
    target_file = "data/{}_for_simcse.csv".format(args.target)
else:
    target_file = "data/filter{}_nli_for_simcse.csv".format(args.filter_type)

print(f"Writing to {target_file}")
f_target = open(target_file, 'w')
writer = csv.writer(f_target)
writer.writerow(["sent0","sent1","hard_neg"])

similarity_gap = 0
if args.filter_type != 1:
    if args.filter_type == 2:
        positive_line = 4
        negative_line = 3
    if args.filter_type == 3:
        positive_line = 4
        negative_line = 2
    if args.filter_type == 4:
        positive_line = 5
        negative_line = 3
    if args.filter_type == 5:
        positive_line = 5
        negative_line = 2
    if args.filter_type == 6:
        positive_line = 4
        negative_line = 1
    if args.filter_type == 7:
        positive_line = 4
        negative_line = 0
    if args.filter_type == 8:
        positive_line = 5
        negative_line = 1
    if args.filter_type == 9:
        positive_line = 5
        negative_line = 0
    if args.filter_type == 10:
        positive_line = 3
        negative_line = 0
    if args.filter_type == 11:
        positive_line = 3
        negative_line = 1
    if args.filter_type == 12:
        positive_line = 3
        negative_line = 2
    if args.filter_type == 13:
        positive_line = 4
        negative_line = 4
        similarity_gap = 1
    if args.filter_type == 14:
        positive_line = 5
        negative_line = 4
    if args.filter_type == 15:
        positive_line = 3
        negative_line = 3
        similarity_gap = 1
    if args.filter_type == 16:
        positive_line = 2
        negative_line = 0
    if args.filter_type == 17:
        positive_line = 2
        negative_line = 1
    if args.filter_type == 18:
        positive_line = 2
        negative_line = 2
        similarity_gap = 1
else:
    positive_line = args.positive_line
    negative_line = args.negative_line
    similarity_gap = args.similarity_gap

count = 0
for i in range(start, end):
    if not is_score_filename:
        print(i)
        if len(dataset[i]) != 3:
            continue
        prompt = ["Scoring the semantic similarity of the following sentences between 0.0 and 5.0: \
                (a) {} (b) {}".format(dataset[i][0], dataset[i][1]),
            "Scoring the semantic similarity of the following sentences between 0.0 and 5.0: \
                (a) {} (b) {}".format(dataset[i][0], dataset[i][-1])]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_length=128)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if ((len(outputs[0]) == 3 and "." in outputs[0] and outputs[0][0] in "012345" and outputs[0][-1] in "0123456789") or (len(outputs[0]) == 1 and outputs[0] in "012345")) \
            and ((len(outputs[1]) == 3 and "." in outputs[1] and outputs[1][0] in "012345" and outputs[1][-1] in "0123456789") or (len(outputs[1]) == 1 and outputs[1] in "012345")):
            if float(outputs[0]) <= float(outputs[1]):
                score_writer.writerow([dataset[i][0],dataset[i][-1],dataset[i][1],outputs[1], outputs[0]])
            else:
                score_writer.writerow([dataset[i][0],dataset[i][1],dataset[i][-1],outputs[0], outputs[1]])

            if float(outputs[0]) <= negative_line and float(outputs[1]) >= positive_line and (not similarity_gap or float(outputs[1]) != float(outputs[0])):
                writer.writerow([dataset[i][0],dataset[i][-1],dataset[i][1]])
            elif float(outputs[1]) <= negative_line and positive_line <= float(outputs[0]) and (not similarity_gap or float(outputs[1]) != float(outputs[0])):
                writer.writerow([dataset[i][0],dataset[i][1],dataset[i][-1]])
    else:
        if len(dataset[i]) != 5:
            continue
        # ["sent0", "sent1", "hard_neg", "positive_score", "negative_score"]
        if float(dataset[i][-1]) <= negative_line and positive_line <= float(dataset[i][-2]):
            if not similarity_gap or float(dataset[i][-2]) != float(dataset[i][-1]):
                count += 1
                writer.writerow([dataset[i][0],dataset[i][1],dataset[i][2]])
print(f"{count} instances are left after filtering.")

# python filter.py --positive_line 3 --negative_line 3 --similarity_gap 1 --target filter15_nli --source flan_nli
# python filter.py --score flan_nli --filter_type 2 --target filter2_nli 