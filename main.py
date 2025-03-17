# This is a sample Python script.
from datetime import timedelta
import math
import random

from pyClarion import (Event, Agent, Priority, Input, Pool, Choice,
    ChunkStore, BaseLevel, Family, NumDict, Atoms, Atom, Chunk, ks_crawl)

class Feats(Atoms):
    _0: Atom; _1: Atom


class IO(Atoms):
    feat1: Atom; feat2: Atom; feat3: Atom

class pairedAssoc(Family):
    io: IO
    val:Feats

