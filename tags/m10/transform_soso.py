#! /usr/bin/env python
# -*-coding:utf-8 -*-

import argparse


def main():
    with open(FLAGS.output_soso_dict_file, "w") as f:
        for index, line in enumerate(open(FLAGS.input_soso_dict_file)):
            try:
                line = unicode(line, 'utf-8')
                tokens = line.strip().split('\t')
                tagname = tokens[0].replace("\x1a", '')
                tagid = tokens[1]
                f.write(tagname.encode('utf-8'))
                f.write('\t')
                f.write(tagid)
                f.write('\n')
            except Exception:
                print("Exception: {}".format(line.encode('utf-8')))

            if index % 500000:
                print("{} lines processed".format(index))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_soso_dict_file',
        type=str,
        default='',
        help=''
    )
    parser.add_argument(
        '--output_soso_dict_file',
        type=str,
        default='',
        help=''
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()
