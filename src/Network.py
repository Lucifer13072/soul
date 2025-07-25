# src/Network.py

from typing import List, Tuple, Optional  # ← Добавлены импорты
import numpy as np
import random
import asyncio



class Neuron:
    def __init__(self, position: Tuple[int, int, int], max_connections: int = 8):
        self.position = position
        self.connections = []
        self.barriers = [np.random.uniform(0.5, 1.0) for _ in range(max_connections)]
        self.accumulated_signals = {}
        self.activation_count = 0
        self.last_activation_step = -1  # Время последней активации
        self.coactivation_counts = {}  # Счётчики совместных активаций

    def add_connection(self, neighbor: 'Neuron'):
        if len(self.connections) < len(self.barriers):
            self.connections.append(neighbor)
            self.coactivation_counts[neighbor] = 0  # Инициализация счётчика

    def process_signal(self, signal, incoming_connection_index: int, current_step: int):
        thought_id = signal.id_thought
        accumulated = self.accumulated_signals.get(thought_id, 0.0)
        new_accumulated = accumulated + signal.ampl
        self.accumulated_signals[thought_id] = new_accumulated
        
        if new_accumulated >= self.barriers[incoming_connection_index]:
            self.activation_count += 1
            self.last_activation_step = current_step  # Записываем шаг активации
            for idx, neighbor in enumerate(self.connections):
                if idx != incoming_connection_index:
                    neighbor.process_signal(signal, idx, current_step)
            self.accumulated_signals[thought_id] = 0.0

    def update_coactivations(self, current_step: int, window: int = 10, decay_rate=0.01):
        for neighbor in self.connections:
            if abs(self.last_activation_step - neighbor.last_activation_step) <= window:
                self.coactivation_counts[neighbor] += 1
            else:
                # Уменьшаем счётчик, если активации не совпадают
                self.coactivation_counts[neighbor] = max(0, self.coactivation_counts[neighbor] - decay_rate)

    def adapt_barriers(self, rate=0.01, min_barrier=0.1, max_barrier=1.0):
        # Адаптируем пороги на основе корреляции
        for idx, neighbor in enumerate(self.connections):
            count = self.coactivation_counts[neighbor]
            if count > 5:  # Высокая корреляция
                self.barriers[idx] = max(self.barriers[idx] - rate, min_barrier)
            elif count < 2:  # Низкая корреляция
                self.barriers[idx] = min(self.barriers[idx] + rate, max_barrier)
            self.coactivation_counts[neighbor] = 0  # Сбрасываем счётчик

class Signal:
    def __init__(self, ampl: float, id: int, id_thought: int) -> None:
        self.ampl = ampl  # Amplitude between -1 and 1
        self.id = id
        self.id_thought = id_thought  # Identifier for the thought group

class Receptor:
    def __init__(self, id_thought: int):
        self.id_thought = id_thought  # Идентификатор группы сигналов

class RGBReceptor(Receptor):
    def generate_signal(self, r: float, g: float, b: float) -> Signal:
        avg = (r + g + b) / 3  # Среднее значение
        return Signal(ampl=avg * 2 - 1, id=id(self), id_thought=self.id_thought)

class TextReceptor(Receptor):
    def generate_signal(self, text: str) -> Signal:
        hash_val = hash(text) % 1000  # Упрощенная кодировка
        return Signal(ampl=(hash_val / 500) - 1, id=id(self), id_thought=self.id_thought)

class SoundReceptor(Receptor):
    def generate_signal(self, amplitude: float) -> Signal:
        return Signal(ampl=amplitude, id=id(self), id_thought=self.id_thought)

class MemoryReceptor(Receptor):
    def generate_signal(self) -> Signal:
        return Signal(ampl=0, id=id(self), id_thought=self.id_thought)

class Network3D:
    def __init__(self, depth: int, height: int, width: int):
        self.depth = depth
        self.height = height
        self.width = width
        self.grid = self._create_3d_grid()
        self._connect_neurons()
        
        # Инициализация рецепторов
        self.rgb_receptor = RGBReceptor(id_thought=100)
        self.text_receptor = TextReceptor(id_thought=200)
        self.sound_receptor = SoundReceptor(id_thought=300)
        self.memory_receptor = MemoryReceptor(id_thought=400)
        
        self.step = 0  # Счётчик шагов
        self.running = True

    def _create_3d_grid(self) -> List[List[List['Neuron']]]:
        grid = []
        for d in range(self.depth):
            layer = []
            for h in range(self.height):
                row = []
                for w in range(self.width):
                    row.append(Neuron((d, h, w)))
                layer.append(row)
            grid.append(layer)
        return grid

    def _connect_neurons(self):
        for d in range(self.depth):
            for h in range(self.height):
                for w in range(self.width):
                    neuron = self.grid[d][h][w]
                    
                    # Основные соседи по осям (6 направлений)
                    neighbors = []
                    
                    # Глубина
                    if d > 0:
                        neighbors.append(self.grid[d-1][h][w])  # Вверх
                    if d < self.depth - 1:
                        neighbors.append(self.grid[d+1][h][w])  # Вниз
                    
                    # Высота
                    if h > 0:
                        neighbors.append(self.grid[d][h-1][w])  # Вперед
                    if h < self.height - 1:
                        neighbors.append(self.grid[d][h+1][w])  # Назад
                    
                    # Ширина
                    if w > 0:
                        neighbors.append(self.grid[d][h][w-1])  # Влево
                    if w < self.width - 1:
                        neighbors.append(self.grid[d][h][w+1])  # Вправо
                    
                    diagonal_directions = [
                        (-1,-1,0), (-1,1,0), (1,-1,0), (1,1,0), 
                        (-1,0,-1), (-1,0,1), (1,0,-1), (1,0,1), 
                        (0,-1,-1), (0,-1,1), (0,1,-1), (0,1,1)
                    ]
                    
                    for dd, dh, dw in diagonal_directions:
                        nd, nh, nw = d + dd, h + dh, w + dw
                        if (0 <= nd < self.depth and 
                            0 <= nh < self.height and 
                            0 <= nw < self.width):
                            neighbor = self.grid[nd][nh][nw]
                            if random.random() > 0.5:
                                neighbors.append(neighbor)
                    
                    random.shuffle(neighbors)
                    for neighbor in neighbors:
                        if len(neuron.connections) < len(neuron.barriers):
                            neuron.add_connection(neighbor)

    async def process_receptor_signal(self, receptor, current_step, *args):
        signal = receptor.generate_signal(*args)
        # Отправка сигнала в центр сети
        center_neuron = self.grid[self.depth//2][self.height//2][self.width//2]
        center_neuron.process_signal(signal, 0, current_step)
    
    async def update(self):
    # Обработка сигналов от рецепторов
        await asyncio.gather(
            self.process_receptor_signal(self.rgb_receptor, self.step, 1.0, 0.5, 0.0),
            self.process_receptor_signal(self.text_receptor, self.step, "Hello"),
            self.process_receptor_signal(self.sound_receptor, self.step, 0.75),
            self.process_receptor_signal(self.memory_receptor, self.step)
        )
        self.step += 1
            
        # Обновляем счётчики совместных активаций
        for d in range(self.depth):
            for h in range(self.height):
                for w in range(self.width):
                    self.grid[d][h][w].update_coactivations(self.step)
            
        # Каждые 100 шагов адаптируем пороги
        if self.step % 100 == 0:
            for d in range(self.depth):
                for h in range(self.height):
                    for w in range(self.width):
                        self.grid[d][h][w].adapt_barriers()
                        

    async def run_network(self):
        while self.running:
            await self.update()
            await asyncio.sleep(1)

network = Network3D(depth=30, height=50, width=100)

asyncio.run(network.run_network())