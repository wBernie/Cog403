from datetime import timedelta
import math
import random
from typing import Any, Self, Sequence, Type

import matplotlib.pyplot as plt

from pyClarion import (Activation, Adam, ErrorSignal, Event, Agent, Layer, Optimizer, Priority, Input, Choice,
    Family, Atoms, Atom, Chunk, Process, Sort, Term,
    MLP, Train, Tanh, Supervised, Site, LeastSquares)

import numpy as np

type D = Family | Sort | Term
type V = Family | Sort
type DV = tuple[D, V]

class Feats(Atoms):
    _0: Atom; _1: Atom


class IO(Atoms):
    feat1: Atom; feat2: Atom; feat3: Atom

class PairedAssoc(Family):
    io: IO
    val:Feats

class divaCost(Supervised):

    correct: bool

    def __init__(self, name, s, cost = LeastSquares()):
        super().__init__(name, s, cost)
        self.correct = False

    # def resolve(self, event: Event) -> None:
    #     updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
    #     if self.input.affected_by(*updates):
    #         self.update()
    
    def update(self, dt: timedelta = timedelta(), priority: Priority=Priority.LEARNING) -> None:
        exp_mask = self.mask[0].exp()
        if self.correct:
            main = self.cost.grad(self.input[0], self.target[0], exp_mask)
        else:
            main = 0
        self.system.schedule(self.update, self.input.update(main, grad=True), dt=dt, priority=priority)

class Choose(Choice):

    c1: Site
    c2: Site
    attention: bool

    def __init__(self, name, p, s, *, sd = 1, attention = False):
        super().__init__(name, p, s, sd=sd)
        self.c1 = Site(self.input.index, {}, 0.0)
        self.c2 = Site(self.input.index, {}, 0.0)
        self.attention = attention
    def select(self, 
        dt: timedelta = timedelta(), 
        priority=Priority.CHOICE
    ) -> None:
        
        c1 = self.c1[0]#.mul(self.input[0])
        c2 = self.c2[0]#.mul(self.input[0])
        diff1 = c1.sub(self.input[0]).pow(x=2).sum()
        diff2 = c2.sub(self.input[0]).pow(x=2).sum()
        decision = self.c1 if diff1[''] < diff2[''] else self.c2
        self.system.schedule(
            self.select,
            self.main.update(decision[0]),
            dt=dt, priority=priority)

class Diva(MLP):

    hs: list
    target: int

    def __init__(self, 
        name: str, 
        p: Family,
        h: Family, 
        s1: V | DV,
        s2: V | DV | None = None,
        layers: Sequence[int] = (),
        optimizer: Type[Optimizer] = Adam, 
        afunc: Activation | None = None,
        l: int = 0,
        train: Train = Train.ALL,
        init_sd: float = 1e-2,
        **kwargs: Any
    ) -> None:
        s2 = s1 if s2 is None else s2
        super(MLP, self).__init__(name)
        self.system.check_root(h)
        self.layers = []
        self.optimizer0 = optimizer(f"{name}.optimizer0", p, **kwargs)
        self.optimizer1 = optimizer(f"{name}.optimizer1", p, **kwargs)
        lkwargs = {"afunc": afunc, "l": l, "train": train, "init_sd": init_sd}
        with self.optimizer0:
            self.hs = []
            hs = self.hs
            for i, n in enumerate(layers):
                hs.append(self._mk_hidden_nodes(h, i, n))
            self.ilayer = Layer(f"{name}.ilayer", s1, hs[0], **lkwargs)
            hi = hs[0]
            layer = self.ilayer
            lkwargs.pop("afunc")
            self.olayer0 = layer >> Layer(f"{name}.olayer0", hi, s2, **lkwargs)
        with self.optimizer1:
            self.olayer1 = layer >> Layer(f"{name}.olayer1", hi, s2, **lkwargs)
        self.optimizer1.add(self.ilayer)
        self.input = Site(self.ilayer.input.index, {}, self.ilayer.input.const)
    
    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.update()
        if event.source == self.ilayer.backward:
            if self.target == 0:
                self.optimizer0.update()
            else:
                self.optimizer1.update()

    def __rshift__[T: Process](self: Self, other: T) -> T:
        if isinstance(other[0], ErrorSignal) and isinstance(other[1], ErrorSignal):
            return (self.olayer0 >> other[0], self.olayer1 >> other[1])
        return NotImplemented


class Participant(Agent):
    d: PairedAssoc
    input: Input
    path1: Diva
    choice: Choose

    def __init__(self, name):
        p = Family()
        h = Family()
        d = PairedAssoc()
        super().__init__(name, p=p, h=h, d=d)
        self.d = d
        with self:
            self.choice = Choose("choice", p, (d.io, d.val))
            self.path1 = Diva('path1', p=p, h=h, s1=(d.io, d.val), layers=[2], afunc=Tanh(), train=Train.WEIGHTS, lr=0.01)
            # self.path2 = Diva('path2', p=p, h=h, s1=(d.io, d.val), layers=[2], afunc=Tanh(), train=Train.WEIGHTS)
            self.diva1 = divaCost('path1.learn', s=(d.io, d.val))
            self.diva2 = divaCost('path2.learn', s=(d.io, d.val))
            self.input = Input("input", (d.io, d.val))
        path1, input = self.path1, self.input
            # path1.ilayer = path2.ilayer
            # path1.olayer.input = path1.ilayer.main
        input >> path1
        # input >> path2
        path1.error1, path1.error2 = path1 >> [self.diva1, self.diva2] #path1 output layer.main = error.input
        # path2.error = path2 >> self.diva2
        self.diva1.target = self.path1.input
        self.diva2.target = self.path1.input
        self.choice.input = input.main
        self.choice.c1 = self.path1.olayer0.main
        self.choice.c2 = self.path1.olayer1.main

def init_stimuli(d: PairedAssoc, l: list) -> list[tuple[Chunk,int]]:
    io, val = d.io, d.val
    return [
        (s ^
         + float(((-1) ** 2 ** int(s[0]))) * io.feat1 ** val[f"_{s[0]}"]
         + float(((-1) ** 2 ** int(s[1]))) * io.feat2 ** val[f"_{s[1]}"]
         + float(((-1) ** 2 ** int(s[2]))) * io.feat3 ** val[f"_{s[2]}"],
        # + io.feat1 ** val[f"_{s[0]}"]
        # + io.feat2 ** val[f"_{s[1]}"] 
        # + io.feat3 ** val[f"_{s[2]}"],
         int(s[3]))
    for s in l]

def trial(p: Participant, correct: int) -> int:
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

def simulate(stim: list):
    epoch = 0
    max_epochs = 50
    trials = 0

    attempts = 0
    max_attempts = 30
    result = np.zeros((30, 50, 8))
    while attempts < max_attempts:
        epoch = 0
        p = Participant("help1")
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
    print(np.shape(result))
    epoch_average = 1 - np.mean(result, axis=(0,2))
    averages.append(epoch_average)

x = (range(1, 51))
for i in range(6):
    plt.figure()
    plt.plot(x, averages[i])
    plt.title(f'model of type {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('percent chance at success')
    plt.savefig(f'graph_{i+1}.png')
    plt.close()

# for r in results:
#     correct = 0
#     for l in r:
#         if l == True:
#             correct += 1
#     print(correct/8)
# correct = 0



# h, d, p all seperate familys