# =====================================
# UPGF-Pro — versão melhorada ao máximo
# Estável + adaptativo + momentum
# =====================================

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)

# -----------------------------
# DADOS
# -----------------------------
x = torch.linspace(-10, 10, 200).unsqueeze(1)
y = 2 * x + 1 + 0.5 * torch.randn_like(x)

# -----------------------------
# MODELO
# -----------------------------
def criar_modelo():
    return nn.Sequential(
        nn.Linear(1, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
    )

criterio = nn.MSELoss()

# -----------------------------
# UPGF-PRO
# -----------------------------
class UPGFPro:
    def __init__(self, params, lr=0.05, beta=0.9, clip=5.0):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.clip = clip
        self.vel = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                # MOMENTUM
                self.vel[i] = self.beta * self.vel[i] + (1 - self.beta) * p.grad

                # PASSO ADAPTATIVO (normalização)
                grad_norm = torch.norm(self.vel[i]) + 1e-8
                passo = self.lr * self.vel[i] / grad_norm

                # ATUALIZAÇÃO
                p_temp = p - passo

                # PROJEÇÃO SUAVE (estabilidade sem travar)
                p.copy_(torch.clamp(p_temp, -self.clip, self.clip))

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# -----------------------------
# TREINAMENTO UPGF-PRO
# -----------------------------
modelo_upgf = criar_modelo()
opt_upgf = UPGFPro(modelo_upgf.parameters())
losses_upgf = []

for _ in range(300):
    y_pred = modelo_upgf(x)
    loss = criterio(y_pred, y)
    loss.backward()
    opt_upgf.step()
    opt_upgf.zero_grad()
    losses_upgf.append(loss.item())

# -----------------------------
# TREINAMENTO ADAM (referência)
# -----------------------------
modelo_adam = criar_modelo()
opt_adam = optim.Adam(modelo_adam.parameters(), lr=0.01)
losses_adam = []

for _ in range(300):
    opt_adam.zero_grad()
    y_pred = modelo_adam(x)
    loss = criterio(y_pred, y)
    loss.backward()
    opt_adam.step()
    losses_adam.append(loss.item())

# -----------------------------
# RESULTADOS
# -----------------------------
print("Loss final UPGF-Pro:", losses_upgf[-1])
print("Loss final Adam    :", losses_adam[-1])

plt.plot(losses_upgf, label="UPGF-Pro (estável)")
plt.plot(losses_adam, label="Adam (referência)")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.title("UPGF-Pro vs Adam")
plt.show()