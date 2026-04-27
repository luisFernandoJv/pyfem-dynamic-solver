import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.animation as animation
import math
import time
import os

class FEMLaplaceSolver:
    def __init__(self, x_coords, y_coords, elements, boundary_nodes, boundary_values):
        self.X = np.array(x_coords)
        self.Y = np.array(y_coords)
        self.NL = np.array(elements)
        self.NDP = np.array(boundary_nodes)
        self.VAL = np.array(boundary_values)
        self.ND = len(self.X)
        self.NE = len(self.NL)
        self.potentials = None

    def assemble_and_solve(self, verbose=False):
        C = lil_matrix((self.ND, self.ND))
        B = np.zeros(self.ND)
        bc_dict = dict(zip(self.NDP, self.VAL))

        for I in range(self.NE):
            K = self.NL[I, :]
            XL, YL = self.X[K], self.Y[K]
            P = np.array([YL[1] - YL[2], YL[2] - YL[0], YL[0] - YL[1]])
            Q = np.array([XL[2] - XL[1], XL[0] - XL[2], XL[1] - XL[0]])
            AREA = 0.5 * abs(P[1] * Q[2] - Q[1] * P[2])
            if AREA <= 0: continue
            CE = (np.outer(P, P) + np.outer(Q, Q)) / (4.0 * AREA)

            for J in range(3):
                IR = K[J]
                if IR in bc_dict:
                    C[IR, IR] = 1.0
                    B[IR] = bc_dict[IR]
                else:
                    for L in range(3):
                        IC = K[L]
                        if IC in bc_dict:
                            B[IR] -= CE[J, L] * bc_dict[IC]
                        else:
                            C[IR, IC] += CE[J, L]
        
        self.potentials = spsolve(C.tocsr(), B)
        return self.potentials

def get_toy_dataset():
    x_coords = np.tile(np.linspace(0, 1, 6), 6)
    y_coords = np.repeat(np.linspace(0, 1, 6), 6)
    elements = []
    for j in range(5):
        for i in range(5):
            n1 = j * 6 + i
            n2 = n1 + 1; n3 = n1 + 6; n4 = n3 + 1
            elements.extend([[n1, n2, n3], [n2, n4, n3]])
    boundary_nodes = list(range(6)) + list(range(6, 30, 6)) + list(range(11, 35, 6)) + list(range(30, 36))
    boundary_values = [0.0]*14 + [100.0]*6 
    return x_coords, y_coords, elements, boundary_nodes, boundary_values

if __name__ == "__main__":
    X, Y, NL, NDP, base_VAL = get_toy_dataset()
    triang = Triangulation(X, Y, NL)
    fig, ax = plt.subplots(figsize=(8, 6))
    frames_totais = 60
    
    def update(frame):
        ax.clear()
        posicao_fonte_x = 0.5 + 0.45 * math.sin(frame * 0.15) 
        valores_dinamicos = np.array(base_VAL)
        for i in range(6):
            idx_valor = 14 + i
            x_do_no = X[NDP[idx_valor]]
            distancia = x_do_no - posicao_fonte_x
            valores_dinamicos[idx_valor] = 100.0 * math.exp(-(distancia**2) / 0.04)
            
        solver = FEMLaplaceSolver(X, Y, NL, NDP, valores_dinamicos)
        potenciais = solver.assemble_and_solve()
        
        cont = ax.tricontourf(triang, potenciais, levels=30, cmap='inferno', vmin=0, vmax=100)
        ax.scatter([posicao_fonte_x], [1.0], color='cyan', s=100, edgecolors='white', label='Heat Source')
        ax.set_title(f"Simulação Dinâmica FEM - Potencial de Laplace\nPosição Fonte: {posicao_fonte_x:.2f}")
        ax.axis('off')
        return cont,

    print("Renderizando animação...")
    ani = animation.FuncAnimation(fig, update, frames=frames_totais, interval=50)
    
    nome_arquivo = 'simulacao_fem_linkedin.gif'
    ani.save(nome_arquivo, writer='pillow', fps=20)
    print(f"Arquivo guardado localmente como: {nome_arquivo}")

    try:
        from google.colab import files
        print("Ambiente Colab detectado. A iniciar download automático...")
        files.download(nome_arquivo)
    except ImportError:
        print("Ambiente local detectado. O ficheiro está na tua pasta actual.")