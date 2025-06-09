#from train import *
from snake_game import SnakeGame
from train import *

def id_from_env(env):
    """
    Generates a unique identifier for the environment based on its parameters.
    """
    return f"{env.width}x{env.height}_b{env.border}_g{env.grass_growth}_{env.max_grass}_"


def get_model(env, force_train=False, file_name=''):
    """
    Returns the trained model, optionally retraining it.
    """
    if not file_name:
        file_name = f"{id_from_env(env)}{file_name}.pth"

    if not force_train:
        try:
            return CNN_QNet.load(env, file_name).to(device)
        except FileNotFoundError:
            print("Modelo não encontrado, iniciando treino...")
        except:
            print("Erro ao dar load do modelo, iniciando treino...")
    
    model = train(env, file_name)
    model.save(file_name)
    print("Modelo salvo em:" + file_name)
        
    return model


if __name__ == '__main__':
    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
        
    board_size = (14, 14)
    border = 1
    env = SnakeGame(*board_size, border=border,grass_growth=0,max_grass=0, food_amount=1)

    model = get_model(env, force_train=False, file_name='m8.pth')
    
    # evaluate_heuristic_baseline(env, num_episodes=100)

    # model.eval()

    # play_with_heuristic(env, scale=10, fps=30, num_episodes=5)
    evaluate_and_show(env, model, top_k=5)
    