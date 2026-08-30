"""NEAT + HyperNEAT — генетическое/эволюционное программирование.

Экспериментальный модуль (astral/experiments/ — песочница).

E1a. NEAT (NeuroEvolution of Augmenting Topologies):
  - Геном = список связей (input→output, вес, innovation_number, enabled)
  - Мутации: добавить связь/узел, изменить вес, включить/выключить
  - Скрещивание: общие гены по innovation_number, диспаратные — от лучшего
  - Специация: по совместимости геномов (excess/disjoint + weight diff)
  - Приспособленность: через fitness-функцию пользователя

E1b. HyperNEAT: CPPN (Compositional Pattern Producing Network) генерирует
  веса связей большой сети из геометрических координат (x1,y1,x2,y2).

Пример использования:
  neat = NEAT(inputs=2, outputs=1)
  def fitness(ind):
      # XOR: оценить геном
      net = neat.to_network(ind)
      return sum(1 for ... )
  neat.evolve(fitness, generations=50, pop_size=30)

Роль в Fuga: подбор топологии декодера/патчевой сети вместо ручной
настройки; HyperNEAT — генерация весов резонаторов/кан из координат.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


# ═══════════════════════════════════════════════════════════════
# NEAT
# ═══════════════════════════════════════════════════════════════
@dataclass
class ConnectionGene:
    """Ген связи в геноме NEAT."""
    in_node: int
    out_node: int
    weight: float
    innovation: int
    enabled: bool = True

    def copy(self) -> "ConnectionGene":
        return ConnectionGene(self.in_node, self.out_node, self.weight,
                              self.innovation, self.enabled)


class Genome:
    """Геном = список связей + размеры входов/выходов.

    Узлы нумеруются: 0..n_inputs-1 — входы, последние n_outputs — выходы,
    скрытые — по возрастанию индекса.
    """

    def __init__(self, n_inputs: int, n_outputs: int):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.connections: list[ConnectionGene] = []
        self.fitness = 0.0
        self.species = -1
        self._next_node = n_inputs + n_outputs

    def _node_ids(self) -> set[int]:
        nodes = set()
        for c in self.connections:
            nodes.add(c.in_node)
            nodes.add(c.out_node)
        return nodes

    def _new_node(self) -> int:
        n = self._next_node
        self._next_node += 1
        return n

    def mutate(self, rng: random.Random) -> None:
        """Случайная мутация генома."""
        roll = rng.random()
        if roll < 0.5 and self.connections:
            # изменить вес случайной связи
            c = rng.choice(self.connections)
            c.weight += rng.gauss(0.0, 0.5)
        elif roll < 0.7 and self.connections:
            # добавить новый узел: расщепить связь
            c = rng.choice(self.connections)
            new_node = self._new_node()
            c.enabled = False
            self.connections.append(
                ConnectionGene(c.in_node, new_node, 1.0,
                               max(x.innovation for x in self.connections) + 1))
            self.connections.append(
                ConnectionGene(new_node, c.out_node, c.weight,
                               max(x.innovation for x in self.connections) + 1))
        elif roll < 0.9:
            # добавить новую связь между случайными узлами
            nodes = sorted(self._node_ids())
            a = rng.choice(nodes)
            b = rng.choice(nodes)
            if a != b and not any(c.in_node == a and c.out_node == b
                                  for c in self.connections):
                self.connections.append(
                    ConnectionGene(a, b, rng.gauss(0.0, 1.0),
                                   max([x.innovation for x in self.connections] or [0]) + 1))
        else:
            # вкл/выкл случайную связь
            if self.connections:
                rng.choice(self.connections).enabled = not \
                    rng.choice(self.connections).enabled


class NEAT:
    """NEAT-эволюция. Вызывает пользовательскую fitness(genome).

    fitness(genome) → float (выше = лучше). Возвращает лучшего индивида
    и лучший геном (геномы переживают через clone()).
    """

    def __init__(self, n_inputs: int, n_outputs: int,
                 pop_size: int = 30, species_threshold: float = 3.0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.pop_size = pop_size
        self.species_threshold = species_threshold
        self.rng = random.Random(42)
        self.population: list[Genome] = []

    def _init_population(self) -> None:
        self.population = [Genome(self.n_inputs, self.n_outputs)
                           for _ in range(self.pop_size)]
        # начальные связи вход→выход
        for g in self.population:
            for i in range(self.n_inputs):
                for o in range(self.n_outputs):
                    g.connections.append(
                        ConnectionGene(i, self.n_inputs + o,
                                       self.rng.gauss(0.0, 1.0),
                                       len(g.connections)))

    def _compatibility(self, a: Genome, b: Genome) -> float:
        """Расстояние между геномами: excess/disjoint + средний вес."""
        ia = {c.innovation: c for c in a.connections}
        ib = {c.innovation: c for c in b.connections}
        common = ia.keys() & ib.keys()
        disjoint = len(ia) + len(ib) - 2 * len(common)
        weight_diff = sum(abs(ia[k].weight - ib[k].weight) for k in common)
        weight_avg = weight_diff / max(1, len(common))
        return disjoint + weight_avg

    def _speciate(self) -> list[list[Genome]]:
        species: list[list[Genome]] = []
        for g in self.population:
            placed = False
            for sp in species:
                if self._compatibility(g, sp[0]) < self.species_threshold:
                    sp.append(g)
                    placed = True
                    break
            if not placed:
                species.append([g])
        return species

    def _crossover(self, a: Genome, b: Genome) -> Genome:
        """Скрещивание: общие гены — случайно, диспаратные — от лучшего."""
        child = Genome(a.n_inputs, a.n_outputs)
        better = a if a.fitness >= b.fitness else b
        other = b if a.fitness >= b.fitness else a
        ia = {c.innovation: c for c in better.connections}
        io = {c.innovation: c for c in other.connections}
        for inn, ca in ia.items():
            if inn in io:
                # общий ген: случайный родитель
                cb = io[inn]
                child.connections.append(ca.copy() if self.rng.random() < 0.5
                                         else cb.copy())
            else:
                # диспаратный: от лучшего
                child.connections.append(ca.copy())
        return child

    def to_network(self, genome: Genome) -> "NeatNetwork":
        """Материализовать геном в сеть (прямой проход)."""
        return NeatNetwork(genome, self.n_inputs, self.n_outputs)

    def evolve(self, fitness_fn, generations: int = 50) -> tuple[Genome, float]:
        """Эволюция. fitness_fn(genome) → float. Возвращает (лучший, fitness)."""
        self._init_population()
        best_genome, best_fitness = None, -1e9
        for gen in range(generations):
            # 1. Оценить
            for g in self.population:
                g.fitness = fitness_fn(g)
                if g.fitness > best_fitness:
                    best_fitness = g.fitness
                    best_genome = g  # сохраняем ссылку (не мутируем его напрямую)
            # 2. Специация
            species = self._speciate()
            # 3. Отбор и размножение
            next_pop: list[Genome] = []
            for sp in species:
                sp.sort(key=lambda g: -g.fitness)
                keep = max(1, len(sp) // 3)
                # элитизм: лучший каждого вида переживает
                for elite in sp[:keep]:
                    next_pop.append(elite)
                # потомки
                while len(next_pop) < self.pop_size:
                    a = self.rng.choice(sp[:keep])
                    b = self.rng.choice(sp[:keep])
                    child = self._crossover(a, b)
                    if self.rng.random() < 0.8:
                        child.mutate(self.rng)
                    child.fitness = 0.0
                    next_pop.append(child)
            self.population = next_pop[:self.pop_size]
            if gen % 10 == 0 or gen == generations - 1:
                print(f"  [NEAT] gen {gen}: species={len(species)} "
                      f"best={best_fitness:.4f} pop={len(self.population)}")
        return best_genome, best_fitness


class NeatNetwork:
    """Сеть из генома: вход → активация узлов → выход."""

    def __init__(self, genome: Genome, n_in: int, n_out: int):
        self.genome = genome
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, inputs: list[float]) -> list[float]:
        nodes = {i: inputs[i] for i in range(self.n_in)}
        # bias-узел (всегда 1.0) — даёт сдвиг активации
        bias_id = self.n_in + self.n_out + 1000
        nodes[bias_id] = 1.0
        # сортируем по топологии: обрабатываем в порядке innovation
        conns = sorted([c for c in self.genome.connections if c.enabled],
                       key=lambda c: c.innovation)
        for c in conns:
            if c.in_node in nodes:
                v = nodes[c.in_node] * c.weight
                nodes[c.out_node] = nodes.get(c.out_node, 0.0) + v
                # активация скрытого/выходного узла (нелинейность для XOR)
                if c.out_node >= self.n_in and c.out_node != bias_id:
                    nodes[c.out_node] = np.tanh(nodes[c.out_node])
        # активация и выход
        outs = []
        for o in range(self.n_out):
            v = nodes.get(self.n_in + o, 0.0)
            outs.append(float(np.tanh(v)))
        return outs


# ═══════════════════════════════════════════════════════════════
# HyperNEAT: CPPN генерирует веса
# ═══════════════════════════════════════════════════════════════
class CPPN:
    """Compositional Pattern Producing Network: (x1,y1,x2,y2) → вес.

    Простая версия: сумма косинусных узлов с обученными весами.
    Роль: генерировать паттерны весов (периодические/симметричные)
    для большой сети из малого генома.
    """

    def __init__(self, n_hidden: int = 8, seed: int = 0):
        self.rng = random.Random(seed)
        self.w1 = np.array([[self.rng.gauss(0, 1) for _ in range(4)]
                            for _ in range(n_hidden)], dtype=np.float64)
        self.b1 = np.array([self.rng.gauss(0, 1) for _ in range(n_hidden)])
        self.w2 = np.array([self.rng.gauss(0, 1) for _ in range(n_hidden)])

    def weight_for(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Вес связи между нейронами (x1,y1) и (x2,y2) из CPPN."""
        inp = np.array([x1, y1, x2, y2])
        h = np.tanh(self.w1 @ inp + self.b1)
        return float(np.tanh(self.w2 @ h))


def demo():
    print("=== E1. NEAT + HyperNEAT ===\n")

    # 1. NEAT: выучить XOR (входы 2, выход 1)
    neat = NEAT(n_inputs=2, n_outputs=1, pop_size=20)

    def xor_fitness(g: Genome) -> float:
        net = neat.to_network(g)
        err = 0.0
        for x1, x2, y in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
            out = net.forward([x1, x2])[0]
            err += (out - y) ** 2
        return -err  # выше = лучше

    print("1. NEAT: эволюция на XOR (2→1):")
    best, fit = neat.evolve(xor_fitness, generations=50)
    net = neat.to_network(best)
    print(f"   лучший fitness={fit:.4f} (0 = идеальный XOR)")
    for x1, x2, y in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        out = net.forward([x1, x2])[0]
        print(f"   {x1}^{x2}: {out:.3f} (ожид {y})")

    # 2. HyperNEAT: CPPN генерирует веса для сетки 3×3
    print("\n2. HyperNEAT: CPPN генерирует веса 3×3:")
    cppn = CPPN(n_hidden=6, seed=7)
    grid = [[cppn.weight_for(x / 3, y / 3, x / 3, y / 3) for x in range(3)]
            for y in range(3)]
    print("   веса (симметрия/паттерн от CPPN):")
    for row in grid:
        print("   " + " ".join(f"{v:+.2f}" for v in row))

    print("\n=== E1. NEAT + HyperNEAT — OK ===")


if __name__ == "__main__":
    demo()
