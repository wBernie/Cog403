import random

import matplotlib.pyplot as plt

from pyClarion import (SGD, Agent, Input, Family, Atoms, Atom, Chunk, Train, Tanh)

from divaClasses import divaCost, Diva, Choose

import numpy as np



class Feats(Atoms):
    _1: Atom


class IO(Atoms):
    feat1: Atom; feat2: Atom; feat3: Atom

class PairedAssoc(Family):
    io: IO
    val:Feats

class Participant(Agent):
    d: PairedAssoc
    input: Input
    path1: Diva
    choice: Choose

    def __init__(self, name, mode = None):
        p = Family()
        h = Family()
        d = PairedAssoc()
        super().__init__(name, p=p, h=h, d=d)
        self.d = d
        with self:
            self.choice = Choose("choice", p, (d.io, d.val), mode=mode)
            self.path1 = Diva('path1', p=p, h=h, s1=(d.io, d.val), layers=[2], afunc=Tanh(), train=Train.ALL, lr=1)
            self.diva1 = divaCost('path1.learn', s=(d.io, d.val))
            self.diva2 = divaCost('path2.learn', s=(d.io, d.val))
            self.input = Input("input", (d.io, d.val))
        path1, input = self.path1, self.input
        input >> path1
        path1.error1, path1.error2 = path1 >> [self.diva1, self.diva2] #path1 output layer.main = error.input
        self.diva1.target = input.main
        self.diva2.target = input.main
        self.choice.input = input.main
        self.diva1.main = path1.olayer0.main
        self.diva2.main = path1.olayer1.main
        self.choice.c1 = self.path1.olayer0.main
        self.choice.c2 = self.path1.olayer1.main

def init_stimuli(d: PairedAssoc, l: list) -> list[tuple[Chunk,int]]:
    """
    Creates chunks from the list
    """
    io, val = d.io, d.val
    return [
        (s ^
        #  + float(((-1) ** 2 ** int(s[0]))) * io.feat1 ** val[f"_{s[0]}"]
        #  + float(((-1) ** 2 ** int(s[1]))) * io.feat2 ** val[f"_{s[1]}"]
        #  + float(((-1) ** 2 ** int(s[2]))) * io.feat3 ** val[f"_{s[2]}"],
         + float(((-1) ** 2 ** int(s[0]))) * io.feat1 ** val._1
         + float(((-1) ** 2 ** int(s[1]))) * io.feat2 ** val._1
         + float(((-1) ** 2 ** int(s[2]))) * io.feat3 ** val._1,
         int(s[3]))
    for s in l]

def trial(p: Participant, correct: int) -> int:
    """
    Starts one trial
    """
    done = 0
    while p.system.queue:
        event = p.system.advance()
        if event.source == p.path1.olayer0.forward or event.source == p.path1.olayer1.forward:
            done += 1
            if done == 2:
                p.choice.select()
                done = 0
        if event.source == p.choice.select:
            choice = 0 if p.choice.main[0] == p.path1.olayer0.main[0] else 1
            if correct == 0:
                p.diva1.update()
            if correct == 1:
                p.diva2.update()
    return choice

def simulate(stim: list, mode: str = None):
    """
    runs 30 random simulations with the same mode (focus or no focus)
    """
    epoch = 0
    max_epochs = 50
    trials = 0

    attempts = 0
    max_attempts = 30
    result = np.zeros((30, 50, 8))
    while attempts < max_attempts:
        epoch = 0
        p = Participant("help1", mode=mode)
        stim1 = init_stimuli(p.d, stim)
        while epoch < max_epochs:
            random.shuffle(stim1)
            while trials < 8:
                p.input.send(stim1[trials][0])
                correct = stim1[trials][1]
                p.path1.target = correct
                p.diva1.correct = not correct #0 is correct
                p.diva2.correct = correct #1 is correct
                choice = trial(p, correct)
                result[attempts, epoch, trials] = (choice == correct)
                trials+= 1
            trials = 0
            epoch += 1
        attempts += 1
    return result.copy()


# Begin simulations
stim1 = ['0000', '0011', '0100', '0111', '1000', '1011', '1100', '1111']
stim2 = ['0000', '0011', '0100', '0111', '1001', '1010', '1101', '1110']
stim3 = ['0000', '0011', '0100', '0110', '1000', '1011', '1101', '1111']
stim4 = ['0000', '0010', '0100', '0111', '1000', '1011', '1101', '1111']
stim5 = ['0000', '0011', '0100', '0111', '1000', '1011', '1101', '1110']
stim6 = ['0000', '0011', '0101', '0110', '1001', '1010', '1100', '1111']

stims = [stim1, stim2, stim3, stim4, stim5, stim6]

averages = []
for i in range(6):
    result = simulate(stims[i])
    epoch_average = 1 - np.mean(result, axis=(0,2))
    averages.append(epoch_average)


# Plot results
x = (range(1, 51))
for i in range(6):
    plt.figure()
    plt.plot(x, averages[i])
    plt.title(f'model of type {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('percent chance at success')
    plt.savefig(f'graphs/graph_{i+1}.png')
    plt.close()

plt.figure()
for i in range(6):
    plt.plot(x, averages[i], label=f"type {i+1}")
plt.title("model for each type")
plt.xlabel('Epoch')
plt.ylabel('percent chance at success')
plt.legend()
plt.savefig("graphs/graph_all.png")
plt.close()


averages_focus = []
for i in range(6):
    result = simulate(stims[i], mode='focus')
    epoch_average = 1 - np.mean(result, axis=(0,2))
    averages_focus.append(epoch_average)

x = (range(1, 51))
for i in range(6):
    plt.figure()
    plt.plot(x, averages_focus[i])
    plt.title(f'focus model of type {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('percent chance at success')
    plt.savefig(f'focus_graphs/focus_graph_{i+1}.png')
    plt.close()

plt.figure()
for i in range(6):
    plt.plot(x, averages_focus[i], label=f"type {i+1}")
plt.title("focus model for each type")
plt.xlabel('Epoch')
plt.ylabel('percent chance at success')
plt.legend()
plt.savefig("focus_graphs/focus_graph_all.png")
plt.close()

np.savetxt("averages.cvc", averages, delimiter=',')
np.savetxt("focus_averages.cvc", averages_focus, delimiter=',')