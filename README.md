# AP-Assignment 2

## About the Project

This project focuses on training a deep neural network to play the classic Snake game using visual input. The model learns to make decisions based solely on images representing the current state of the game board. The board is set to a 30x30 grid with a 1-pixel border, resulting in 32x32 RGB images used as input for the neural network.

The goal of the agent is to maximize its performance by surviving and collecting as much food as possible, with each game capped at a maximum of 1000 steps. Through deep reinforcement learning, the agent gradually learns effective strategies for navigating the board, avoiding collisions, and optimizing its movements. The video included showcases ten games played by the trained agent.

## Authors
- Diogo Almeida 70140: dmh.almeida@campus.fct.unl.pt 
- Duarte Rodrigues 70150: dms.rodrigues@campus.fct.unl.pt
- Afonso Franco 70406: ao.franco@campus.fct.unl.pt
- Simão Mota 70659: sj.mota@campus.fct.unl.pt

## Setting Up the Environment

To get started with this project, follow these steps to set up your environment:

1. **Create a Virtual Environment**  
   Create a virtual environment to isolate project dependencies:
   ```
   python -m venv venv
   ```

2. **Activate the Virtual Environment**  
   Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

3. **Install Dependencies**  
   Install the required dependencies listed in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```