from gluoncv.model_zoo import get_model
import matplotlib.pyplot as plt
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
#from PIL import Image
import io
from flask import Flask, request

#------------------------------#
#            CLASESS           #
#------------------------------#

from numpy import random as rnd
from math import log10
from typing import Tuple
import pandas as pd
import requests, zipfile, os
import random, string
import re

class fitness_calculator(object):
  def __init__(self, url: str, n_population: int):
    ''' This class implements a quadgram-based evaluator that can be used in a genetic algorithm for cryptanalysis.  '''
    self.n_population = n_population
    if not os.path.exists('english_quadgrams.txt.zip'):
      try:
        response = requests.get(url)
        with open('english_quadgrams.txt.zip', 'wb') as f:
            f.write(response.content)
      except:
        print('Error getting the file.')
        return
    with zipfile.ZipFile('english_quadgrams.txt.zip') as zip_file:
          text_str = zip_file.read('english_quadgrams.txt').decode()
    self.quadgrams_logprob = {row.split()[0]: float(row.split()[1]) for row in text_str.strip().split('\n')}
    self.totalSamples = sum(self.quadgrams_logprob.values())
    self.floor = log10(0.01 / self.totalSamples)
    for k in self.quadgrams_logprob: self.quadgrams_logprob[k] = log10(self.quadgrams_logprob[k] / self.totalSamples)

  def generate_random_key(self) -> str:
    ''' Generate a random alphabet key for decipher a cryptogram. '''
    cipherKeyList = list(string.ascii_uppercase)
    random.shuffle(cipherKeyList)
    return "".join(cipherKeyList)

  def get_logfreq(self, crypted_txt: str, n = 1) -> Tuple[dict, str]:
    ''' Get the log probability of n-grams in a text and the most frequent n-gram. '''
    subStrings = [crypted_txt[i : i + n] for i in range(len(crypted_txt) - n + 1)]
    freq_dic = {subString: log10(subStrings.count(subString) / len(subStrings)) for subString in subStrings}
    frequent_obj = max(freq_dic, key = freq_dic.get)
    return freq_dic, frequent_obj

  def get_fitness(self, txt: str) -> float:
    ''' Get the fitness of a text based on the frequency of english quagrams. '''
    txtUppr = re.sub(r'[^A-Z]', '', txt.upper())
    fitness = 0.0
    for i in range(len(txtUppr) - 4 + 1 ):
      if txtUppr[i : i + 4] in self.quadgrams_logprob: fitness += self.quadgrams_logprob[txtUppr[i : i + 4]];
      else: fitness += self.floor
    return fitness

  def replace_cipherKey(self, crypted_txt: str, cipherKey: str) -> str:
    ''' Replace letters of a crypted text based on a given cipher key. '''
    decrypted_text = ""
    #crypted_txt = re.sub(r'[^A-Z]', '', crypted_txt.upper())
    for character in crypted_txt:
      if re.match('[A-Z]', character):
        decrypted_text += chr(cipherKey.upper().index(character.upper()) + ord('A'))
      elif re.match('[a-z]', character):
        decrypted_text += chr(cipherKey.upper().index(character.upper()) + ord('a'))
      else:
        decrypted_text += character
    return decrypted_text

  def generate_population(self, crypted_txt: str) -> dict:
    ''' Generates a population of n_population keys for decrypting an encrypted text then calculates the fitness of each key. '''
    population = [self.generate_random_key() for i in range(self.n_population)]
    fitnesses = [self.get_fitness(self.replace_cipherKey(crypted_txt, k)) for k in population]
    return dict(zip(population, fitnesses))

  def select_parent(self, population_dic: dict) -> str:
    ''' Randomly selects a parent from a population dictionary based on the fitness values of each individual. '''
    max = abs(sum(population_dic.values()))
    selection_probs = [abs(population_dic[k]) / max for k in population_dic]
    return [list(population_dic.keys())[rnd.choice(len(population_dic), p=selection_probs)]][0]


#Definir clase genetic_mutator()
class genetic_mutator(object):
  def __init__(self, mutator: str):
    ''' This class implements genetic algorithms for solving a decryption problem.
        The class has methods for mutation.  '''
    self.mutator = mutator
    self.mutators = {
        'insert_mutation': self.insert_mutation,
        'swap_mutation': self.swap_mutation,
        'inversion_mutation': self.inversion_mutation,
        'scramble_mutation': self.scramble_mutation
    }

  def insert_mutation(self, parent: str) -> str:
    ''' Implements the insert mutation method to two parents. '''
    parent_list = list(parent)
    pos1, pos2 = random.sample(range(len(parent_list)), 2)
    element = parent_list.pop(pos2)
    parent_list.insert(pos1 + 1, element)
    return ''.join(parent_list)

  def swap_mutation(self, parent: str) -> str:
    ''' Implements the swap mutation method to two parents. '''
    parent_list = list(parent)
    pos1, pos2 = random.sample(range(len(parent_list)), 2)
    parent_list[pos1], parent_list[pos2] = parent_list[pos2], parent_list[pos1]
    return ''.join(parent_list)

  def scramble_mutation(self, parent: str) -> str:
    ''' Implements the scramble mutation method to two parents. '''
    l = len(parent)
    r1 = rnd.randint(0, l)
    r2 = rnd.randint(r1, l)
    parent_list = list(parent)
    subList = parent_list[r1:r2]
    random.shuffle(subList)
    parent_list[r1:r2] = subList
    return ''.join(parent_list)

  def inversion_mutation(self, parent: str) -> str:
    ''' Implements the inversion mutation method to two parents. '''
    l = len(parent)
    r1 = rnd.randint(0, l)
    r2 = rnd.randint(r1, l)
    parent_list = list(parent)
    parent_list[r1:r2] = reversed(parent_list[r1:r2])
    return ''.join(parent_list)

class genetic_crossover(object):
  def __init__(self, crossover: str):
    ''' This class implements genetic algorithms for solving a decryption problem.
        The class has methods for selection and crossover.  '''
    self.crossover = crossover
    self.crossovers = {
        'order_one_crossover': self.order_one_crossover,
        'partially_mapped_crossover': self.partially_mapped_crossover,
        'cycle_crossover': self.cycle_crossover
    }

  def order_one_crossover(self, parent1: str, parent2: str) -> str:
    '''  Implements the order one crossover method to two parents. '''
    l = len(parent1)
    r1 = rnd.randint(0, l)
    r2 = rnd.randint(r1, l)
    child = [''] * l
    for i in range(r1, r2): child[i] = parent1[i]
    not_in_child1 = [parent1[j] for j in range(l) if not parent1[j] in child]
    not_in_child2 = [parent2[(r2 + k) % l]  for k in range(l) if parent2[(r2 + k) % l] in not_in_child1]
    for q in range(len(not_in_child1)): child[(r2 + q) % l] = not_in_child2[q]
    return ''.join(child)

  def partially_mapped_crossover(self, parent1: str, parent2: str) -> str:
    ''' Implements the partially mapped crossover method to two parents. '''
    l = len(parent1)
    r1 = rnd.randint(0, l)
    r2 = rnd.randint(r1, l)
    child = [''] * l
    for i in range(r1, r2): child[i] = parent1[i]
    mapping = {child[j]: parent2[j]  for j in range(r1, r2) if not parent2[j] in child}
    for j in mapping:
      if child[parent2.index(j)] != '': child[parent2.index(child[parent2.index(j)])] = mapping[j]
      else: child[parent2.index(j)] = mapping[j]
    for k in range(l):
      if child[k] == '': child[k] = parent2[k]
    return ''.join(child)

  def cycle_crossover(self, parent1: str, parent2: str) -> str:
    ''' Implements the cycle crossover method to two parents. '''
    l = len(parent1)
    child = [''] * l
    start_pos = random.randint(0, l)
    index = start_pos
    while True:
      child[index] = parent2[index]
      index = parent1.index(parent2[index])
      if index == start_pos:
        break
    for i in range(l):
      if child[i] == '': child[i] = parent1[i]
    return ''.join(child)

class genetic_decipher(object):
  def __init__(self):
    ''' '''
    self.fitness_calc = fitness_calculator('http://practicalcryptography.com/media/cryptanalysis/files/english_quadgrams.txt.zip', 10)
    self.crossover = genetic_crossover('order_one_crossover')
    self.mutator = genetic_mutator('insert_mutation')
    self.xo_rate = 0.5
    self.mutation_rate = 0.1
    self.crypted_text = ''

  def fitness_calculator(self, url = 'http://practicalcryptography.com/media/cryptanalysis/files/english_quadgrams.txt.zip', n_population = 10):
    ''' '''
    self.fitness_calc= fitness_calculator(url, n_population)

  def genetic_crossover(self, crossover = 'order_one_crossover', xo_rate = 0.5):
    ''' '''
    self.crossover = genetic_crossover(crossover)
    self.xo_rate = xo_rate

  def genetic_mutator(self, mutator = 'insert_mutation', mutation_rate = 0.1):
    ''' '''
    self.mutator = genetic_mutator(mutator)
    self.mutation_rate = mutation_rate

  def mate_keys(self, population: dict) -> dict:
    ''' '''
    key_ = ''
    mate = ''
    new_population = []
    new_population_fitness = []
    for i in range(self.fitness_calc.n_population):
      key_ = self.fitness_calc.select_parent(population)
      mate = self.fitness_calc.select_parent(population)
      if population[key_] > population[mate]:
        target = list(key_); source = list(mate)
      else:
        target = list(mate); source = list(key_)
      target_fitness = population[''.join(target)]
      for char in range(len(target)):
        if source[char] != target[char]:
          copy = target.copy()
          temp = copy.index(source[char])
          oldValue = copy[char]
          copy[char] = source[char]
          copy[temp] = oldValue
          temp_fitness = self.fitness_calc.get_fitness(self.fitness_calc.replace_cipherKey(self.crypted_text, ''.join(copy)))
          if temp_fitness > target_fitness:
            target = copy
            target_fitness = temp_fitness
      if random.random() < self.mutation_rate:
        target = self.mutator.mutators[self.mutator.mutator](target)
        target_fitness = self.fitness_calc.get_fitness(self.fitness_calc.replace_cipherKey(self.crypted_text, ''.join(target)))
      new_population.append(''.join(target))
      new_population_fitness.append(target_fitness)
    return dict(zip(new_population, new_population_fitness))

  def evolve_population(self, parents_dict: dict) -> dict:
    ''' '''
    parent1 = ''
    parent2 = ''
    winner = ''
    loser = ''
    temp_child = ''
    new_population = [''] * self.fitness_calc.n_population
    for i in range(self.fitness_calc.n_population):
      parent1 = self.fitness_calc.select_parent(parents_dict)
      parent2 = self.fitness_calc.select_parent(parents_dict)
      if parents_dict[parent1] > parents_dict[parent2]:
        winner = parent1; loser = parent2
      else:
        winner = parent2; loser = parent1
      temp_child = winner
      if random.random() < self.xo_rate:
        new_population[i] = self.crossover.crossovers[self.crossover.crossover](winner, loser)
      else:
        new_population[i] = temp_child
      if random.random() < self.mutation_rate:
        new_population[i] = self.mutator.mutators[self.mutator.mutator](new_population[i])
    fitnesses = [self.fitness_calc.get_fitness(self.fitness_calc.replace_cipherKey(self.crypted_text, k)) for k in new_population]
    return dict(zip(new_population, fitnesses))

  def decipher(self, crypted_text: str, max_iterations = 20, fitness_target = -950):
    ''' '''
    self.crypted_text = crypted_text
    population = self.fitness_calc.generate_population(crypted_text)
    best_pop_chr = ()
    best_chr_sofar = ('', -9999)
    decrypted_text = ''
    for i in range(max_iterations):
      best_pop_chr = max(population.items(), key=lambda x: x[1])
      if best_pop_chr[1] > fitness_target:
        decrypted_text = self.fitness_calc.replace_cipherKey(crypted_text, best_pop_chr[0])
        print('Target successfully achieved!\n\tIteration:\t%d\n\tKey:\t\t%s\n\tFitness:\t%f\nDecrypted text: %s' % (i + 1, best_pop_chr[0], best_pop_chr[1], decrypted_text))
        best_chr_sofar = best_pop_chr
        break
      else:
        if best_pop_chr[1] > best_chr_sofar[1]:
          best_chr_sofar = best_pop_chr
        decrypted_text = self.fitness_calc.replace_cipherKey(crypted_text, best_chr_sofar[0])
        print('Iteration: %d\n\tBest key so far:\t%s\n\tBast fitness so far:\t%f\nDecrypted text: %s\n' % (i + 1, best_chr_sofar[0], best_chr_sofar[1], decrypted_text))
        #population = self.evolve_population(population)
        population = self.mate_keys(population)
    return (i, best_chr_sofar, decrypted_text)

#------------------------------#
#            CLASESS           #
#------------------------------#

app = Flask(__name__)


@app.route("/decipher",methods=["POST"])
def decipher():
    if request.method == "POST":
        if request.files.get("txt"):
                txt1 = request.files["txt"].read()
                if txt1 is None:
                   return 'No hay texto'
                txt = txt1.decode()
                poblacion_inicial = 100 
                tasa_de_mutacion = 0.01 
                algoritmo_de_mutacion =  "scramble_mutation"		
                decipher = genetic_decipher()
                decipher.fitness_calculator(n_population= poblacion_inicial)
                decipher.genetic_crossover(crossover= 'order_one_crossover', xo_rate= 0.6)
                decipher.genetic_mutator(mutator= algoritmo_de_mutacion, mutation_rate= tasa_de_mutacion)

                numero_de_iteraciones = 20 
                objetivo_de_puntuación = -50 
                i, best_chr_sofar, decrypted_text = decipher.decipher(crypted_text= txt, max_iterations= numero_de_iteraciones, fitness_target = objetivo_de_puntuación)
    return 'The best result was:  %s'%(decrypted_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
