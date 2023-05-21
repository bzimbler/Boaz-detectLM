import numpy as np
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm


def merge_to_min_length(output_path, dict_ds, min_no_sents = 100):
    no_sents = 0
    itr = 0
    text = ""
    for k in tqdm(dict_ds.keys()):
        textr = dict_ds[k]['text']
        textr = "\n".join([t for t in textr.split('\n') if len(t) > 2])
        no_sents += len(textr.split('\n'))
        text += '\n\n' + textr
        if no_sents >= min_no_sents:
            itr += 1
            outfile = output_path + f"_{itr}.txt"
            with open(outfile, "wt") as f:
                f.write(text)
            text = ""
            no_sents = 0

def merge_to_min_length_json(output_file, dict_ds, min_no_sents = 100):
    no_sents = 0
    itr = 0
    text = ""
    output = {}
    for k in tqdm(dict_ds.keys()):
        textr = dict_ds[k]['text']
        eps = dict_ds[k]['eps']
        source = dict_ds[k]['source']
        textr = "\n".join([t for t in textr.split('\n') if len(t) > 2])
        no_sents += len(textr.split('\n'))
        text += '\n\n' + textr
        if no_sents >= min_no_sents:
            itr += 1
            output[k] = dict(itr=itr,
                            text=text,
                            eps=eps,
                            no_sentences=no_sents,
                            source= source
                             )
            with open(output_file, "w") as f:
                json.dump(output, f, indent=True)

            text = ""
            no_sents = 0

def main():
    parser = argparse.ArgumentParser(description='Merge multiple texts to docs of minimal length')
    parser.add_argument('-i', type=str, help='input file', default="")
    parser.add_argument('-n', type=int, help='minimal number of sentences per file', default=100)
    parser.add_argument('-o', type=str, help='output path', default="./atleast")
    parser.add_argument('--files', action='store_true')
    args = parser.parse_args()

    input_file = args.i
    min_no_sents = args.n
    output_path = args.o

    with open(input_file, "r") as file:
        mixed_dataset = json.load(file)

    print(f"Merging {input_file} to documents with at least {min_no_sents} sentences")

    if args.files:
        print(f"Creating files in {output_path}...")
        merge_to_min_length(output_path, mixed_dataset, min_no_sents)
    else:
        print(f"Storing to {output_path}...")
        merge_to_min_length_json(output_path, mixed_dataset, min_no_sents)


if __name__ == '__main__':
    main()
