import argparse
import re

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_file_path')
    parser.add_argument('output_file_path')
    args = parser.parse_args()

    with open(args.input_file_path, 'r') as reader, open(args.output_file_path, encoding='utf-8', mode='w') as writer:
        out = []
        for line in reader.readlines():

            out.append(
                ''.join(list(map(lambda x: ' ' if x == '' else x, line.split(' ')))))

        writer.writelines(out)
