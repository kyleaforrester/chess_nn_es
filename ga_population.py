#!/usr/bin/env python3

import ga_individual as gi
import random
import os
import math

class Population:

    elite_percent = 0.01
    pop_size = 10

    def __init__(self):
        self.population = []

    def new_population(self):
        self.population = []
        for i in range(Population.pop_size):
            self.population.append(gi.Individual())
        gi.Individual.play(self.population)

    def write_population(self, fd):
        for ind in self.population:
            fd.write(ind.to_string() + '\n')

    def read_population(self, file_name):
        self.population = []
        for line in open(file_name).readlines():
            if (line.count('|') == 3):
                self.population.append(gi.Individual.read(line))
            else:
                print('Not expected format!')

    def new_generation(self):
        elites = sorted(self.population, key=lambda x: x.evaluation)[:math.ceil(len(self.population)*Population.elite_percent)+1]
        new_gen = [elite for elite in elites]

        while (len(new_gen) < len(self.population)):
            new_gen.append(random.choice(elites).copy().mutate())

        self.population = new_gen
        gi.Individual.play(self.population)
        best_ind = min(self.population, key=lambda x: x.evaluation)
        if (best_ind not in elites):
            #Found a good individual, increase mutation
            gi.Individual.mutate_std_dev *= 1.02
        else:
            #Stagnant, decrease mutation
            gi.Individual.mutate_std_dev *= 0.9

    def execute_generations(self):
        if (len(self.population) == 0):
            self.new_population()
        user_input = input('How many generations do you wish to produce? ')
        while (not user_input.isdigit()):
            user_input = input('Error: Non-Integer response.\nHow many generations do you wish to produce? ')
        for i in range(1, int(user_input) + 1):
            self.new_generation()
            print('Completed {} Generations'.format(i))

    def kill_individual(self):
        challengers = sorted(random.sample(self.population, 2), key=lambda x: x.evaluation)

        self.population.remove(challengers[1])

    def create_individual(self):
        challengers = sorted(random.sample(self.population, 4), key=lambda x: x.evaluation)

        self.population.append(gi.Individual.mate(challengers[0], challengers[1]))

    def change_pop_size(self):
        if (len(self.population) == 0):
            self.new_population()
        print('Current population size: {}'.format(len(self.population)))
        user_input = input('What should the new population size be? ')
        while (not user_input.isdigit()):
            user_input = input('Error: Non-Integer response.\nWhat should the new population size be? ')
        user_input = int(user_input)
        if (user_input < len(self.population)):
            while (user_input < len(self.population)):
                self.kill_individual()
                print('Population size is now {}'.format(len(self.population)))
        elif (user_input > len(self.population)):
            while (user_input > len(self.population)):
                self.create_individual()
                print('Population size is now {}'.format(len(self.population)))
            print('Re-evaluating population.  This may take a moment...')
            gi.Individual.play(self.population)
        else:
            print('Population size is already {}. Nothing to do.'.format(len(self.population)))

    def run(self):

        while (True):
            print('Enter an option:')
            print('1: Load population')
            print('2: Save population')
            print('3: New population')
            print('4: Change population size')
            print('5: Save best individual')
            print('6: Execute generations')
            user_input = input('User input: ')

            if (user_input == '1'):
                file_name = input('What filename do you wish to load? ')
                self.read_population('population/' + file_name)
            elif (user_input == '2'):
                file_name = input('What filename do you wish to save as? ')
                fd = open('population/' + file_name, 'w')
                self.write_population(fd)
            elif (user_input == '3'):
                pass
            elif (user_input == '4'):
                self.change_pop_size()
            elif (user_input == '5'):
                pass
            elif (user_input == '6'):
                self.execute_generations()
            else:
                print('Invalid command: {}'.format(user_input))
