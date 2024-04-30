import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from tqdm import tqdm
import csv


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="t5")

parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch_size", type=int, default=16)

parser.add_argument("--part", type=int, default=1)
parser.add_argument("--total_part", type=int, default=1)

parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=0)

parser.add_argument("--source", type=str, default="nli")
parser.add_argument("--target", type=str, default="flan_nli")

args = parser.parse_args()
print(args)

if "wiki" in args.source:
    filename = "data/wiki1m_for_simcse.txt"
    f1 = open(filename, "r")
    sentences = list(f1.readlines())
elif "nli" in args.source:
    sentences = []
    filename = "data/nli_for_simcse.csv"
    csvfile = open(filename, 'r')
    sentence_reader = csv.reader(csvfile)
    for row in sentence_reader:
        if "sent0" in row and "sent1" in row and "hard_neg" in row: 
            continue
        sentences.append(row[0])

def generate_(sentence_list, index, type):
    if type == "c":
        """
        Other contradiction prompts you can choose:
        1. Revise the provided sentence by swapping, changing, or contradicting some details in order to express a different meaning, while maintaining the general context and structure.
        2. Generate a slightly modified version of the provided sentence to express an opposing or alternate meaning by changing one or two specific elements, while maintaining the overall context and sentence structure.
        3. Transform the input sentence by adjusting, altering, or contradicting its original meaning to create a logical and sensible output sentence with a different meaning from the input sentence.
        4. Generate a sentence that conveys a altering, contrasting or opposite idea to the given input sentence, while ensuring the new sentence is logical, realistic, and grounded in common sense.
        """
        prompt = "Help me write a sentence that is negative with a given sentence. Here are some examples:\n\n"
        prompt += "Write a sentence that is negative with '{}': '{}'\n\n".format("One of our number will carry out your instructions minutely.", "We have no one free at the moment so you have to take action yourself.")
        prompt += "Write a sentence that is negative with '{}': '{}'\n\n".format("How do you know? All this is their information again.", "They have no information at all.")
        # prompt += "Write a sentence that is negative with '{}': '{}'\n\n".format("yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range", "The tennis shoes are not over hundred dollars.")
        # prompt += "Write a sentence that is negative with '{}': '{}'\n\n".format("my walkman broke so i'm upset now i just have to turn the stereo up real loud", "My walkman still works as well as it always did.")
        # prompt += "Write a sentence that is negative with '{}': '{}'\n\n".format("'But a few Christian mosaics survive above the apse is the Virgin with the infant Jesus, with the Archangel Gabriel to the right (his companion Michael, to the left, has vanished save for a few feathers from his wings).'", "There are no mosaics left in this place at all.")
        prompt += "Write a sentence that is negative with '{}': ".format(sentence_list[index].strip())
    else:
        """
        Other entailment prompts you can choose:
        1. Please paraphrase the input sentence or phrase, providing an alternative expression with the same meaning.
        2. Rewrite the following sentence or phrase using different words and sentence structure while preserving its original meaning.
        3. Create a sentence or phrase that is also true, assuming the provided input sentence or phrase is true.
        4. Please provide a concise paraphrase of the input sentence or phrase, maintaining the core meaning while altering the words and sentence structure. Feel free to omit some of the non-essential details like adjectives or adverbs.
        """
        prompt = "Help me write a sentence that is entailment with a given sentence. Here are some examples:\n\n"
        prompt += "Write a sentence that is entailment with '{}': '{}'\n\n".format("One of our number will carry out your instructions minutely.", "A member of my team will execute your orders with immense precision.")
        prompt += "Write a sentence that is entailment with '{}': '{}'\n\n".format("How do you know? All this is their information again.", "This information belongs to them.")
        # prompt += "Write a sentence that is entailment with '{}': '{}'\n\n".format("yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range", "The tennis shoes can be in the hundred dollar range.")
        # prompt += "Write a sentence that is entailment with '{}': '{}'\n\n".format("my walkman broke so i'm upset now i just have to turn the stereo up real loud", "I'm upset that my walkman broke and now I have to turn the stereo up really loud.")
        # prompt += "Write a sentence that is entailment with '{}': '{}'\n\n".format("'But a few Christian mosaics survive above the apse is the Virgin with the infant Jesus, with the Archangel Gabriel to the right (his companion Michael, to the left, has vanished save for a few feathers from his wings).'", "There are a few Christian mosaics left in the building.")
        prompt += "Write a sentence that is entailment with '{}': ".format(sentence_list[index].strip())
    return prompt

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

contraditory = []
entailment = []

if args.target != "0":
    target_file = "data/{}_for_simcse.csv".format(args.target)
else:
    target_file = "data/{}_for_simcse.csv".format(str(args.part))
print(f"Writing to {target_file}")


model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to(device)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")


f_target = open(target_file, 'w')
writer = csv.writer(f_target)
writer.writerow(["sent0","sent1","hard_neg","positive_score","negative_score"])

sentence_per_part = len(sentences)//(args.total_part)
start_index = (args.part-1)*sentence_per_part
end_index = (args.part)*sentence_per_part
if args.part == args.total_part:
    end_index = len(sentences)

if args.start != 0 or args.end != 0:
    start_index = args.start
    end_index = min(args.end, len(sentences))
print(f"Writing from {start_index} to {end_index}")

for i in tqdm(range((end_index-start_index)//args.batch_size+1)):
    start = start_index+i*args.batch_size
    end = min(start_index+(i+1)*args.batch_size, end_index)
    # print(start, end)
    c_inputs_ = []
    e_inputs_ = []
    contraditory = []
    entailment = []
    for j in range(start, end):
        c_inputs_.append(generate_(sentences, j, "c"))
        e_inputs_.append(generate_(sentences, j, "e"))

    c_inputs = tokenizer(c_inputs_, return_tensors="pt", padding=True, truncation=True).to(device)
    c_outputs = model.generate(**c_inputs, max_length=256)
    contraditory.extend(tokenizer.batch_decode(c_outputs, skip_special_tokens=True))

    e_inputs = tokenizer(e_inputs_, return_tensors="pt", padding=True, truncation=True).to(device)
    e_outputs = model.generate(**e_inputs, max_length=256)
    entailment.extend(tokenizer.batch_decode(e_outputs, skip_special_tokens=True))
    
    c_scores_ = []
    e_scores_ = []
    for j in range(start, end):
        prompt = ["Scoring the semantic similarity of the following sentences between 0.0 and 5.0: \
                (a) {} (b) {}".format(sentences[j], contraditory[j-start]),
            "Scoring the semantic similarity of the following sentences between 0.0 and 5.0: \
                (a) {} (b) {}".format(sentences[j], entailment[j-start])]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_length=256)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if ((len(outputs[0]) == 3 and "." in outputs[0] and outputs[0][0] in "012345" and outputs[0][-1] in "0123456789") or (len(outputs[0]) == 1 and outputs[0] in "012345")) \
            and ((len(outputs[1]) == 3 and "." in outputs[1] and outputs[1][0] in "012345" and outputs[1][-1] in "0123456789") or (len(outputs[1]) == 1 and outputs[1] in "012345")):
            if float(outputs[0]) <= 4.0 and float(outputs[1]) >= float(outputs[0]):
                writer.writerow([sentences[j].strip(),entailment[j-start],contraditory[j-start],outputs[1],outputs[0]])
            elif float(outputs[1]) <= 4.0 and float(outputs[1]) <= float(outputs[0]):
                writer.writerow([sentences[j].strip(),contraditory[j-start],entailment[j-start],outputs[0],outputs[1]])
