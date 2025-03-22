from datetime import timedelta
import math
import random

from pyClarion import (Event, Agent, Priority, Input, Pool, Choice,
    ChunkStore, BaseLevel, Family, NumDict, Atoms, Atom, Chunk, ks_crawl,
    MLP)

class Feats(Atoms):
    _0: Atom; _1: Atom


class IO(Atoms):
    feat1: Atom; feat2: Atom; feat3: Atom

class PairedAssoc(Family):
    io: IO
    val:Feats

class Participant(Agent):
    d: PairedAssoc
    input: Input
    store: ChunkStore
    path1: MLP
    path2: MLP

    def __init__(self, name):
        p = Family()
        e = Family()
        d = PairedAssoc()
        super().__init__(name, p=p, e=e, d=d)
        self.d = d
        self.path1 = MLP('path1', p, p, d, layers=[3,2,3])
        self.path2 = MLP('path2', p, p, d, None, [3,2,3])

p = Participant('help')
# h, d, p all seperate familys