import collections
import random
import numpy as np
from utils import preprocess

class ReplayBuffer:
    def __init__(self, capacity=10000):
        """
        Inicializa o Experience Replay Buffer.

        Args:
            capacity (int): Capacidade máxima do buffer.
        """
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Adiciona uma nova experiência ao buffer.

        Args:
            state (np.array): O estado atual (pré-processado).
            action (int): A ação tomada (índice 0, 1 ou 2 para -1, 0, 1).
            reward (float): A recompensa recebida.
            next_state (np.array): O próximo estado (pré-processado).
            done (bool): True se o episódio terminou, False caso contrário.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Amostra um lote aleatório de experiências do buffer.

        Args:
            batch_size (int): O tamanho do lote a ser amostrado.

        Returns:
            tuple: Um tuple contendo arrays NumPy de (states, actions, rewards, next_states, dones).
        """
        # Garante que não tentamos amostrar mais do que o que temos no buffer
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        # Separar a tupla de experiências em lotes individuais
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Converter para arrays NumPy
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states),
                np.array(dones))

    def __len__(self):
        """
        Retorna o número atual de experiências no buffer.
        """
        return len(self.buffer)

    def populate(self, env, policy_func, num_steps):
        """
        Popula o buffer com experiências usando uma política específica.
        Útil para warm-start.

        Args:
            env (SnakeGame): O ambiente do jogo.
            policy_func (callable): A função da política a ser usada para gerar ações (e.g., random_policy, heuristic_policy).
            num_steps (int): O número de passos a serem executados para popular o buffer.
        """
        print(f"A popular o buffer de replay com {num_steps} passos...")
        state_raw, _, _, _ = env.reset()
        preprocessed_state = preprocess(state_raw)
        
        for _ in range(num_steps):
            # Obtém a ação da política. Assumimos que policy_func retorna -1, 0 ou 1.
            action = policy_func(env,pathfind="astar") 
            
            next_state_raw, reward, done, _ = env.step(action)
            preprocessed_next_state = preprocess(next_state_raw)
            
            # Converte a ação para o índice 0, 1, 2 antes de adicionar ao buffer
            action_idx = action + 1 
            self.add(preprocessed_state, action_idx, reward, preprocessed_next_state, done)

            if done:
                state_raw, _, _, _ = env.reset()
                preprocessed_state = preprocess(state_raw)
            else:
                preprocessed_state = preprocessed_next_state
        print(f"Buffer populado. Tamanho atual: {len(self)}")