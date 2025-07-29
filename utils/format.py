
# coding: utf-8
# create by tongshiwei on 2019-8-14

import io
import json

from tqdm import tqdm

__all__ = ["tl2json", "json2tl"]


def tl2json(src: str, tar: str, to_int=True, left_shift=False):
    """
    convert the dataset in `tl` sequence into `json` sequence

    .tl format
    The first line is the number of exercises a student attempted.
    The second line is the exercise tag sequence.
    The third line is the response sequence. ::

        15
        1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
        0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    .json format
    Each sample contains several response elements, and each element is a two-element list.
    The first is the exercise tag and the second is the response. ::

        [[1,0],[1,1],[1,1],[1,1],[7,1],[7,1],[9,0],[10,0],[10,1],[10,1],[10,1],[11,1],[11,1],[45,0],[54,0]]

    """
    with open(src) as f, io.open(tar, "w", encoding="utf-8") as wf:
        for _ in tqdm(f):
            exercise_tags = f.readline().strip().strip(",").split(",")
            response_sequence = f.readline().strip().strip(",").split(",")
            if to_int:
                if not left_shift:
                    exercise_tags = list(map(int, exercise_tags))
                else:
                    exercise_tags = list(map(lambda x: int(x) - 1, exercise_tags))
                response_sequence = list(map(int, response_sequence))
            responses = list(zip(exercise_tags, response_sequence))
            print(json.dumps(responses), file=wf)

def tl2json_extend(src: str, tar: str, to_int=True, left_shift=False, exercise_line=1, response_line=2, line_per_group=3, write_mode="w"):
    """
    convert the dataset in `tl` sequence into `json` sequence

    .tl format
    The first line is the number of exercises a student attempted.
    The second line is the exercise tag sequence.
    The third line is the response sequence. ::

        15
        1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
        0,1,1,1,1,1,0,0,1,1,1,1,1,0,0

    .json format
    Each sample contains several response elements, and each element is a two-element list.
    The first is the exercise tag and the second is the response. ::

        [[1,0],[1,1],[1,1],[1,1],[7,1],[7,1],[9,0],[10,0],[10,1],[10,1],[10,1],[11,1],[11,1],[45,0],[54,0]]

    """
    with open(src) as f, io.open(tar, write_mode, encoding="utf-8") as wf:
        for num_exer in tqdm(f):
            for lineID in range(1, line_per_group): # 第一行已经被读取，所以少读一行
                line = f.readline()
                if int(num_exer) == 0:
                    continue
                if lineID != (exercise_line % line_per_group) and lineID != (response_line % line_per_group): 
                    continue
                line_data = line.strip("\n").split(",")
                if len(line_data[len(line_data) - 1]) == 0:
                    line_data = line_data[:-1]
                if lineID == exercise_line % line_per_group:
                    exercise_tags = line_data
                if lineID == response_line % line_per_group:
                    response_sequence = line_data
            if to_int and len(exercise_tags) > 1:
                if not left_shift:
                    exercise_tags = list(map(int, exercise_tags))
                else:
                    exercise_tags = list(map(lambda x: int(x) - 1, exercise_tags))
                response_sequence = list(map(int, response_sequence))
            responses = list(zip(exercise_tags, response_sequence))
            print(json.dumps(responses), file=wf)

def json2tl(src, tar):
    with open(src) as f, io.open(tar, "w", encoding="utf-8") as wf:
        for line in tqdm(f):
            responses = json.loads(line)
            exercise_tags, response_sequence = zip(*responses)
            print(len(exercise_tags), file=wf)
            print(",".join(list(map(str, exercise_tags))), file=wf)
            print(",".join(list(map(str, response_sequence))), file=wf)


if __name__ == '__main__':
    json2tl("../data/junyi/student_log_kt.json.small.train", "../data/junyi/student_log_kt.json.small.train.tl")
    json2tl("../data/junyi/student_log_kt.json.small.test", "../data/junyi/student_log_kt.json.small.test.tl")
