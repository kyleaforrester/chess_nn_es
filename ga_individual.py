#!/usr/bin/env python3

import random
import chess
import chess.uci
import chess.pgn
import numpy as np
import multiprocessing as mp
import copy
import datetime
import math


class Individual:

    mutate_std_dev = 0.005
    mutate_rate = 1
    curr_positions = []
    sample_size = 1000

    def __init__(self, layers=None, weights=None, biases=None, evaluation = 0):
        self.evaluation = evaluation
        if layers is None:
            self.layers = [837, 20, 1]
        else:
            self.layers = layers

        if weights is None:
            #Initialize weights
            all_weights = []
            for layer_idx in range(1, len(self.layers)):
                layer_weights = []
                for node_idx in range(self.layers[layer_idx]):
                    w_range = 0 
                    weights = [random.random()*2*w_range - w_range for i in range(self.layers[layer_idx-1])]
                    layer_weights.append(weights)
                all_weights.append(layer_weights)
            self.weights = all_weights
        else:
            self.weights = weights

        if biases is None:
            #Initialize biases
            all_biases = []
            for layer_idx in range(1, len(self.layers)):
                layer_biases = []
                for node_idx in range(self.layers[layer_idx]):
                    biases = [0]
                    layer_biases.append(biases)
                all_biases.append(layer_biases)
            self.biases = all_biases
        else:
            self.biases = biases

        self.mutate()

    def copy(self):
        return Individual(weights=copy.deepcopy(self.weights), layers=self.layers, biases=copy.deepcopy(self.biases))
        
    def mutate(self):
        for layer in range(1, len(self.layers)):
            for node in range(self.layers[layer]):
                #Mutate the bias
                if (random.randint(1,Individual.mutate_rate) == 1):
                    self.biases[layer-1][node][0] += random.gauss(0,Individual.mutate_std_dev)
                #Mutate the weights
                for weight_idx in range(len(self.weights[layer-1][node])):
                    if (random.randint(1,Individual.mutate_rate) == 1):
                        self.weights[layer-1][node][weight_idx] += random.gauss(0,Individual.mutate_std_dev)
        self.np_weights = np.array(self.weights)
        self.np_biases = np.array(self.biases)
        return self

    def mate(ind_a, ind_b):
        wins = 0

        all_weights = []
        all_biases = []
        for layer in range(1, len(ind_a.layers)):
            layer_weights = []
            layer_biases = []
            node_count = ind_a.layers[layer]
            for node in range(node_count):
                weights = []
                bias = []
                for weight_idx in range(len(ind_a.weights[layer-1][node])):
                    ind_a_weight = ind_a.weights[layer-1][node][weight_idx]
                    ind_b_weight = ind_b.weights[layer-1][node][weight_idx]
                    rand_int = random.randint(1,3)
                    if (rand_int == 1):
                        weights.append((ind_b_weight - ind_a_weight) * (random.random()*0.5) + ind_a_weight)
                    elif (rand_int == 2):
                        weights.append(ind_a_weight)
                    else:
                        weights.append(ind_b_weight)
                ind_a_bias = ind_a.biases[layer-1][node][0]
                ind_b_bias = ind_b.biases[layer-1][node][0]
                rand_int = random.randint(1,3)
                if (rand_int == 1):
                    bias.append((ind_b_bias - ind_a_bias) * (random.random()*0.5) + ind_a_bias)
                elif (rand_int == 2):
                    bias.append(ind_a_bias)
                else:
                    bias.append(ind_b_bias)
                layer_weights.append(weights)
                layer_biases.append(bias)
            all_weights.append(layer_weights)
            all_biases.append(layer_biases)

        return Individual(ind_a.layers, all_weights, all_biases).mutate()

    def relu(np_array):
        return np.maximum(np_array, np.zeros(np_array.shape))

    def leaky_relu(np_array):
        fraction = np_array / 20
        return np.maximum(np_array, fraction)

    def sigmoid(np_array):
        return 1 / (1 + np.exp(-np_array))

    def no_act(np_array):
        return np_array

    def fire(self, l0a):
        #l0a = layer 0 activations
        last_index = len(self.biases) - 1
        act = np.array(l0a)
        for layer_idx in range(last_index):
            act = Individual.leaky_relu((self.np_weights[layer_idx] @ act) + self.np_biases[layer_idx])
            #fd_out.write('{}\n'.format(act))

        act = Individual.sigmoid((self.np_weights[last_index] @ act) + self.np_biases[last_index])
        #fd_out.write('{}\n'.format(act))

        return act[0][0]

    def create_activations(fen):
        #Each board square follows the following activations:
        #0-5: my_pawn, my_knight, my_bishop, my_rook, my_queen, my_king
        #6-11: opp_pawn, opp_knight, opp_bishop, opp_rook, opp_queen, opp_king
        #12: empty_square
        split_fen = fen.split(' ')
        side_to_move = split_fen[1]
        board_str = split_fen[0]
        castling = split_fen[2]
        en_passent = split_fen[3]
        #White to move
        if (side_to_move == 'w'):
            l0a = []
            for char in board_str:
                if (char.isdigit()):
                    digit = int(char)
                    for i in range(digit):
                        for j in range(12):
                            l0a.append([0.0])
                        l0a.append([1.0])
                elif (char == 'P'):
                    l0a.append([1.0])
                    for i in range(12):
                        l0a.append([0.0])
                elif (char == 'N'):
                    for i in range(1):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(11):
                        l0a.append([0.0])
                elif (char == 'B'):
                    for i in range(2):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(10):
                        l0a.append([0.0])
                elif (char == 'R'):
                    for i in range(3):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(9):
                        l0a.append([0.0])
                elif (char == 'Q'):
                    for i in range(4):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(8):
                        l0a.append([0.0])
                elif (char == 'K'):
                    for i in range(5):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(7):
                        l0a.append([0.0])
                elif (char == 'p'):
                    for i in range(6):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(6):
                        l0a.append([0.0])
                elif (char == 'n'):
                    for i in range(7):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(5):
                        l0a.append([0.0])
                elif (char == 'b'):
                    for i in range(8):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(4):
                        l0a.append([0.0])
                elif (char == 'r'):
                    for i in range(9):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(3):
                        l0a.append([0.0])
                elif (char == 'q'):
                    for i in range(10):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(2):
                        l0a.append([0.0])
                elif (char == 'k'):
                    for i in range(11):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(1):
                        l0a.append([0.0])

            #Castling rights:
            #my_king, my_queen, opp_king, opp_queen
            if ('K' in castling):
                l0a.append([1.0])
            else:
                l0a.append([0.0])
            if ('Q' in castling):
                l0a.append([1.0])
            else:
                l0a.append([0.0])
            if ('k' in castling):
                l0a.append([1.0])
            else:
                l0a.append([0.0])
            if ('q' in castling):
                l0a.append([1.0])
            else:
                l0a.append([0.0])

            #En Passent rights:
            #Just a yes or no value
            if (en_passent == '-'):
                l0a.append([0.0])
            else:
                l0a.append([1.0])
        #Black to move
        else:
            board_str = board_str[::-1]
            l0a = []
            for char in board_str:
                if (char.isdigit()):
                    digit = int(char)
                    for i in range(digit):
                        for j in range(12):
                            l0a.append([0.0])
                        l0a.append([1.0])
                elif (char == 'p'):
                    l0a.append([1.0])
                    for i in range(12):
                        l0a.append([0.0])
                elif (char == 'n'):
                    for i in range(1):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(11):
                        l0a.append([0.0])
                elif (char == 'b'):
                    for i in range(2):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(10):
                        l0a.append([0.0])
                elif (char == 'r'):
                    for i in range(3):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(9):
                        l0a.append([0.0])
                elif (char == 'q'):
                    for i in range(4):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(8):
                        l0a.append([0.0])
                elif (char == 'k'):
                    for i in range(5):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(7):
                        l0a.append([0.0])
                elif (char == 'P'):
                    for i in range(6):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(6):
                        l0a.append([0.0])
                elif (char == 'N'):
                    for i in range(7):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(5):
                        l0a.append([0.0])
                elif (char == 'B'):
                    for i in range(8):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(4):
                        l0a.append([0.0])
                elif (char == 'R'):
                    for i in range(9):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(3):
                        l0a.append([0.0])
                elif (char == 'Q'):
                    for i in range(10):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(2):
                        l0a.append([0.0])
                elif (char == 'K'):
                    for i in range(11):
                        l0a.append([0.0])
                    l0a.append([1.0])
                    for i in range(1):
                        l0a.append([0.0])

            #Castling rights:
            #my_king, my_queen, opp_king, opp_queen
            if ('k' in castling):
                l0a.append([1.0])
            else:
                l0a.append([0.0])
            if ('q' in castling):
                l0a.append([1.0])
            else:
                l0a.append([0.0])
            if ('K' in castling):
                l0a.append([1.0])
            else:
                l0a.append([0.0])
            if ('Q' in castling):
                l0a.append([1.0])
            else:
                l0a.append([0.0])

            #En Passent rights:
            #Just a yes or no value
            if (en_passent == '-'):
                l0a.append([0.0])
            else:
                l0a.append([1.0])

        return l0a

    def evaluate(ind):
        error = 0
        for pos in Individual.curr_positions:
            ind_eval = ind.fire(pos[2])
            error += 10**abs(ind_eval - float(pos[1])) - 1
        return error

    def play(ind_list):
        now = datetime.datetime.now()
        fd_out = open('games/{}_{}_{}_{}_{}_{}.txt'.format(now.year, now.month, now.day, now.hour, now.minute, now.second), 'w')

        positions = open('data/data.txt').readlines()

        Individual.curr_positions = copy.deepcopy(random.sample(positions, Individual.sample_size))
        Individual.curr_positions = [pos.strip().split(',') for pos in Individual.curr_positions]
        Individual.curr_positions = [(pos[0], pos[3], Individual.create_activations(pos[0])) for pos in Individual.curr_positions]
        del positions

        with mp.Pool() as pool:
            evals = pool.map(Individual.evaluate, ind_list)
            for i in range(len(ind_list)):
                ind_list[i].evaluation = evals[i]

        fd_out.write('Mutation standard deviation: {}\n'.format(Individual.mutate_std_dev))
        for ind in ind_list:
            fd_out.write('Total Error: {}\n'.format(ind.evaluation))
        fd_out.write('Best Error: {}'.format(min(ind_list, key=lambda x: x.evaluation).evaluation))

    def to_string(self):
        return '{}|{}|{}|{}'.format(self.evaluation, self.layers, self.weights, self.biases)

    def read(string):
        p_str = string.split('|')
        return Individual(eval(p_str[1]), eval(p_str[2]), eval(p_str[3]), float(p_str[0]))
