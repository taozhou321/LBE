import numpy as np
import math

class DATAWithBehavior(object):
    def __init__(self, seqlen, separate_char, min_seq_len):
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.min_seq_len = min_seq_len

    '''
    data format:
    length
    KC sequence
    answer sequence
    exercise sequence
    time taken sequence
    attempt count sequence
    hint count sequence
    time_factor sequence
    attempt factor sequence
    hint factor sequence
    '''

    def load_data(self, path):
        f_data = open(path, 'r')
        q_data = []
        a_data = []
        e_data = []
        time_taken_data = []
        attempt_cnt_data = []
        hint_cnt_data = []
        time_data = []
        attempt_data = []
        hint_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 10 != 0:
                line_data = line.split(self.separate_char)
                if len(line_data[len(line_data) - 1]) == 0:
                    line_data = line_data[:-1]
            if lineID % 10 == 1:
                Q = line_data
            elif lineID % 10 == 2:
                A = line_data
            elif lineID % 10 == 3:
                E = line_data
            elif lineID % 10 == 4:
                Time_taken = line_data
            elif lineID % 10 == 5:
                Attempt_cnt = line_data
            elif lineID % 10 == 6:
                Hint_cnt = line_data
            elif lineID % 10 == 7:
                Time = line_data
            elif lineID % 10 == 8:
                Attempt = line_data
            elif lineID % 10 == 9:
                Hint = line_data

                # start split the data after getting the final feature
                n_split = 1
                total_len = len(A)
                if total_len > self.seqlen:
                    n_split = math.floor(len(A) / self.seqlen)
                    if total_len % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    exercise_sequence = []
                    time_taken_sequence = []
                    attempt_cnt_sequence = []
                    hint_cnt_sequence = []
                    time_sequence = []
                    attempt_sequence = []
                    hint_sequence = []
                    if k == n_split - 1:
                        end_index = total_len
                    else:
                        end_index = (k + 1) * self.seqlen
                    # choose the sequence length is larger than min_seq_len
                    if end_index - k * self.seqlen > self.min_seq_len:
                        for i in range(k * self.seqlen, end_index):
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(int(A[i]))
                            exercise_sequence.append(int(E[i]))
                            time_taken_sequence.append(float(Time_taken[i]))
                            attempt_cnt_sequence.append(int(float(Attempt_cnt[i])))
                            hint_cnt_sequence.append(int(float(Hint_cnt[i])))
                            time_sequence.append(float(Time[i]))
                            attempt_sequence.append(float(Attempt[i]))
                            hint_sequence.append(float(Hint[i]))

                        # print('instance:-->', len(instance),instance)
                        q_data.append(question_sequence)
                        a_data.append(answer_sequence)
                        e_data.append(exercise_sequence)
                        time_taken_data.append(time_taken_sequence)
                        attempt_cnt_data.append(attempt_cnt_sequence)
                        hint_cnt_data.append(hint_cnt_sequence)
                        time_data.append(time_sequence)
                        attempt_data.append(attempt_sequence)
                        hint_data.append(hint_sequence)
        f_data.close()
        # data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        r_dataArray = np.ones((len(a_data), self.seqlen)) * -1
        for j in range(len(a_data)):
            dat = a_data[j]
            r_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(e_data), self.seqlen))
        for j in range(len(e_data)):
            dat = e_data[j]
            p_dataArray[j, :len(dat)] = dat

        time_taken_dataArray = np.zeros((len(time_taken_data), self.seqlen))
        for j in range(len(time_taken_data)):
            dat = time_taken_data[j]
            time_taken_dataArray[j, :len(dat)] = dat

        attempt_cnt_dataArray = np.zeros((len(attempt_cnt_data), self.seqlen))
        for j in range(len(attempt_cnt_data)):
            dat = attempt_cnt_data[j]
            attempt_cnt_dataArray[j, :len(dat)] = dat

        hint_cnt_dataArray = np.zeros((len(hint_cnt_data), self.seqlen))
        for j in range(len(hint_cnt_data)):
            dat = hint_cnt_data[j]
            hint_cnt_dataArray[j, :len(dat)] = dat

        # time_dataArray = np.zeros((len(time_data), self.seqlen))
        # for j in range(len(time_data)):
        #     dat = time_data[j]
        #     time_dataArray[j, :len(dat)] = dat

        # attempt_dataArray = np.zeros((len(attempt_data), self.seqlen))
        # for j in range(len(attempt_data)):
        #     dat = attempt_data[j]
        #     attempt_dataArray[j, :len(dat)] = dat

        # hint_dataArray = np.zeros((len(hint_data), self.seqlen))
        # for j in range(len(hint_data)):
        #     dat = hint_data[j]
        #     hint_dataArray[j, :len(dat)] = dat

        return q_dataArray, p_dataArray, r_dataArray, \
            time_taken_dataArray, attempt_cnt_dataArray, hint_cnt_dataArray,\
            # time_dataArray, attempt_dataArray, hint_dataArray
    
        # time_dataArray_shifted = np.zeros_like(time_dataArray)
        # time_dataArray_shifted[:, 1:] = time_dataArray[:, :-1]

        # attempt_dataArray_shifted = np.zeros_like(attempt_dataArray)
        # attempt_dataArray_shifted[:, 1:] = attempt_dataArray[:, :-1]

        # hint_dataArray_shifted = np.zeros_like(hint_dataArray)
        # hint_dataArray_shifted[:, 1:] = hint_dataArray[:, :-1]

        # return q_dataArray, p_dataArray, r_dataArray, time_dataArray_shifted, \
        #     attempt_dataArray_shifted, hint_dataArray_shifted
