import argparse
import re
from mosestokenizer import MosesTokenizer

en_word_tokenize = MosesTokenizer('en')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_file_path')
    parser.add_argument('output_file_path')
    args = parser.parse_args()

    with open(args.input_file_path, 'r') as reader, open(args.output_file_path, encoding='utf-8', mode='w') as writer:
        out = []
        for line in reader.readlines():

            out.append(' '.join(en_word_tokenize(re.sub(r'$\n', '', line))) + '\n')

        writer.writelines(out)
