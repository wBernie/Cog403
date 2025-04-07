from datetime import timedelta
from typing import Any, Self, Sequence, Type
from pyClarion import (NumDict, Supervised, LeastSquares, Priority, Site, Choice, MLP, Family, Tanh, Train, Optimizer, Adam,
                       Activation, Layer, Event, Process, ErrorSignal, Sort, Term)
import sys
import random
type D = Family | Sort | Term
type V = Family | Sort
type DV = tuple[D, V]

class divaCost(Supervised):
    """
    An implementation of supervised that only updates if it's the correct branch
    """

    correct: bool

    def __init__(self, name, s, cost = LeastSquares()):
        super().__init__(name, s, cost)
        self.correct = False
    
    def update(self, dt: timedelta = timedelta(), priority: Priority=Priority.LEARNING) -> None:
        exp_mask = self.mask[0].exp()
        # input = self.input[0].mul(self.target[0].abs())
        main = self.cost.grad(self.input[0], self.target[0], exp_mask)
        self.system.schedule(self.update, self.main.update(main, grad=True), dt=dt, priority=priority)

class Choose(Choice):
    """
    An implementation of Choice that checks for closest match to input between two outputs with the option of focusing
    """
    c1: Site
    c2: Site
    attention: bool

    def __init__(self, name, p, s, *, sd = 1, mode = None):
        super().__init__(name, p, s, sd=sd)
        self.c1 = Site(self.input.index, {}, 0.0)
        self.c2 = Site(self.input.index, {}, 0.0)
        self.mode = mode
    def select(self, 
        dt: timedelta = timedelta(), 
        priority=Priority.CHOICE
    ) -> None:
        c1 = self.c1[0]#.mul(self.input[0].abs())
        c2 = self.c2[0]#.mul(self.input[0].abs())
        if self.mode:
            k = Site(self.input.index, {}, max(c1.max()[''], c2.max()['']) - min(c1.min()[''], c2.min()['']))
            beta = Site(self.input.index, {}, 2)
            focus = ((c1.sub(c2)).abs().sub(k[0])).mul(beta[0]).exp()
            diff1 = (c1.sub(self.input[0])).mul(focus).pow(x=2).sum()['']
            diff2 = (c2.sub(self.input[0])).mul(focus).pow(x=2).sum()['']
        else:
            diff1 = (c1.sub(self.input[0])).pow(x=2).sum()['']
            diff2 = (c2.sub(self.input[0])).pow(x=2).sum()['']
        prob1 = (1/diff1)/((1/diff1) + (1/diff2))
        
        decision = self.c1 if random.random() < prob1 else self.c2
        self.system.schedule(
            self.select,
            self.main.update(decision[0]),
            dt=dt, priority=priority)

class Diva(MLP):
    """
    An implementation of MLP for Diva which has 2 output layers
    """

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
        self.optimizer1.add(self.ilayer)
        with self.optimizer1:
            self.olayer1 = layer >> Layer(f"{name}.olayer1", hi, s2, **lkwargs)
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
