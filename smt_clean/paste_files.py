# coding=utf8

import argparse


def merge(f1, f2, out_f):
    with open(f1) as in_1:
        with open(f2) as in_2:
            with open(out_f, "w") as out_:
                lines1 = in_1.readlines()
                lines2 = in_2.readlines()
                out_lines = [" ||| ".join([l1.rstrip("\n"), l2]) for l1, l2 in zip(lines1, lines2)]
                out_.writelines(out_lines)


def main(f1, f2, out_f):
    merge(f1, f2, out_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sou")
    parser.add_argument("--tgt")
    parser.add_argument("--out")
    args = parser.parse_args()
    main(args.sou, args.tgt, args.out)

