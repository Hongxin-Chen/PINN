"""
pinn_shg_1d.py — 物理信息神经网络(PINN)求解一维连续波二次谐波产生(SHG)

这是构建完整三维脉冲高斯光束 SHG 仿真的第一步。
目标：验证 PINN 框架能准确求解非线性耦合波方程。

耦合波方程（归一化形式）：
    dA₁/dζ = -i·Γα · A₂ · A₁* · exp(-i·σ·ζ)     （基频光）
    dA₂/dζ = -i·Γβ · A₁² · exp(+i·σ·ζ)            （倍频光）

其中：
    ζ = z/L ∈ [0, 1]        归一化传播坐标
    Γα, Γβ                   无量纲非线性耦合强度
    σ = Δk·L                 无量纲相位失配量
    A₁(0) = 1, A₂(0) = 0    初始条件

将复振幅分解为实部和虚部 A_j = u_j + i·v_j，得到 4 个实数 ODE，
用 PyTorch autograd 计算精确导数，作为 PINN 的物理约束。

完美相位匹配 (σ=0, Γα=Γβ=Γ) 解析解：
    A₁(ζ) = sech(Γζ)           （基频光强度下降）
    A₂(ζ) = -i·tanh(Γζ)        （倍频光强度上升）
    转换效率 η = tanh²(Γ)

运行方式：
    python pinn_shg_1d.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import os

# 使用 float64 提高 PDE 残差的计算精度
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"运算设备: {device}")


# ============================================================
#  第一部分：物理参数
# ============================================================
class SHGParams:
    """
    SHG 物理参数

    支持两种模式：
      'physical'   —— 从真实晶体参数（KTP）出发自动计算无量纲耦合系数
      'normalized' —— 直接指定无量纲参数，方便快速测试
    """

    def __init__(self, mode='normalized'):
        if mode == 'physical':
            self._from_physical()
        else:
            self._from_normalized()

    # ---------- 模式 1: 真实晶体参数 ----------
    def _from_physical(self):
        self.crystal = "KTP"
        self.L       = 10e-3          # 晶体长度 [m]
        self.deff    = 3.18e-12       # 有效非线性系数 [m/V]
        self.n1      = 1.830          # 基频折射率 (1064 nm)
        self.n2      = 1.889          # 倍频折射率 (532 nm)
        self.lam1    = 1064e-9        # 基频波长 [m]
        self.c0      = 3e8            # 光速 [m/s]
        self.eps0    = 8.854e-12      # 真空介电常数 [F/m]
        self.I0      = 5e13           # 输入峰值光强 [W/m²] ≈ 5 GW/cm²

        omega1  = 2 * np.pi * self.c0 / self.lam1
        k1      = 2 * np.pi * self.n1 / self.lam1
        k2      = 2 * np.pi * self.n2 / (self.lam1 / 2)
        E0      = np.sqrt(2 * self.I0 / (self.n1 * self.eps0 * self.c0))
        alpha   = 2 * omega1 * self.deff / (self.n1 * self.c0)     # dA₁/dz 耦合
        beta    = 2 * omega1 * self.deff / (self.n2 * self.c0)    # dA₂/dz 耦合

        self.Gamma_alpha = alpha * E0 * self.L
        self.Gamma_beta  = beta  * E0 * self.L
        self.sigma       = (k2 - 2 * k1) * self.L
        self.mr_coeff    = self.n2 / self.n1   # Manley-Rowe 系数

    # ---------- 模式 2: 直接给无量纲参数 ----------
    def _from_normalized(self):
        self.crystal = "Normalized"
        self.Gamma_alpha = 3.0        # 耦合强度 (≈π 时达到 ~100% 转换)
        self.Gamma_beta  = 3.0
        self.sigma       = 0.0        # 0 = 完美相位匹配；试试 20, 50
        self.n1 = self.n2 = 1.0
        self.mr_coeff    = 1.0        # n₂/n₁

    def summary(self):
        print("=" * 50)
        print(f"  SHG 参数  [{self.crystal}]")
        print("=" * 50)
        print(f"  Γα  = {self.Gamma_alpha:.4f}   (基频耦合)")
        print(f"  Γβ  = {self.Gamma_beta:.4f}   (倍频耦合)")
        print(f"  σ   = {self.sigma:.4f}   (相位失配 ΔkL)")
        print(f"  n₂/n₁ = {self.mr_coeff:.4f} (Manley-Rowe)")
        if self.sigma == 0 and self.Gamma_alpha == self.Gamma_beta:
            eta = np.tanh(self.Gamma_alpha) ** 2
            print(f"  理论转换效率 η = tanh²(Γ) = {eta*100:.2f}%")
        print("=" * 50)


# ============================================================
#  第二部分：RK45 参考解（scipy）
# ============================================================
def solve_shg_rk45(params, N=1000):
    """
    用 scipy RK45 求解 SHG 耦合波方程，作为"真值"对比参考。

    将复方程拆为 4 个实数 ODE:
        y = [u₁, v₁, u₂, v₂],  A_j = u_j + i·v_j

    Returns: (zeta, u1, v1, u2, v2)  各为 shape (N,) 的 numpy 数组
    """
    Ga, Gb, sig = params.Gamma_alpha, params.Gamma_beta, params.sigma

    def rhs(zeta, y):
        u1, v1, u2, v2 = y
        c, s = np.cos(sig * zeta), np.sin(sig * zeta)

        # A₂·A₁* 的实部 R1 和虚部 I1
        R1 = u2 * u1 + v2 * v1
        I1 = v2 * u1 - u2 * v1

        # A₁² 的实部 R2 和虚部 I2
        R2 = u1**2 - v1**2
        I2 = 2 * u1 * v1

        # 方程 1:  dA₁/dζ = -iΓα · A₂A₁* · exp(-iσζ)
        du1 =  Ga * (I1 * c - R1 * s)
        dv1 = -Ga * (R1 * c + I1 * s)

        # 方程 2:  dA₂/dζ = -iΓβ · A₁² · exp(+iσζ)
        du2 =  Gb * (I2 * c + R2 * s)
        dv2 = -Gb * (R2 * c - I2 * s)

        return [du1, dv1, du2, dv2]

    zeta_eval = np.linspace(0, 1, N)
    sol = solve_ivp(rhs, [0, 1], [1.0, 0.0, 0.0, 0.0],
                    method='RK45', t_eval=zeta_eval, rtol=1e-12, atol=1e-14)
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


# ============================================================
#  第三部分：神经网络架构
# ============================================================
class FourierFeatures(nn.Module):
    """
    傅里叶特征映射 —— 克服神经网络的"谱偏置 (Spectral Bias)"

    标准 MLP 天然倾向于学习低频函数，难以拟合 exp(iΔkz) 这类高频振荡。
    将输入 z 映射到 [sin(2π·B·z), cos(2π·B·z)] 可以直接暴露高频信息。

    Ref: Tancik et al., NeurIPS 2020 — "Fourier Features Let Networks
         Learn High Frequency Functions in Low Dimensional Domains"
    """

    def __init__(self, num_features=64, scale=4.0):
        super().__init__()
        # 随机频率矩阵，训练时冻结
        B = torch.randn(1, num_features) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        proj = 2 * np.pi * x @ self.B          # [batch, num_features]
        return torch.cat([torch.sin(proj),
                          torch.cos(proj)], dim=-1)  # [batch, 2*num_features]


class PINN_SHG(nn.Module):
    """
    SHG 物理信息神经网络

    结构:
        输入 ζ ∈ [0,1]
        → 傅里叶特征映射
        → 全连接隐藏层 × N (tanh 激活)
        → 4 个输出 [u₁, v₁, u₂, v₂]

    关键设计 —— 硬边界约束 (Hard Boundary Constraint):
        u₁ = 1 + ζ · f₁(ζ)      → 保证 u₁(0) = 1
        v₁ = ζ · f₂(ζ)          → 保证 v₁(0) = 0
        u₂ = ζ · f₃(ζ)          → 保证 u₂(0) = 0
        v₂ = ζ · f₄(ζ)          → 保证 v₂(0) = 0

    通过结构设计精确满足初始条件，无需额外的 BC 损失项，
    大幅减轻"多目标平衡"的训练难度。
    """

    def __init__(self, hidden_dim=128, num_layers=5,
                 num_fourier=64, fourier_scale=4.0,
                 hard_bc=True):
        super().__init__()
        self.hard_bc = hard_bc

        self.fourier = FourierFeatures(num_fourier, fourier_scale)
        in_dim = 2 * num_fourier

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 4))

        self.net = nn.Sequential(*layers)

        # Xavier 初始化
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, zeta):
        """
        zeta : [batch, 1]
        returns : [batch, 4] → [u₁, v₁, u₂, v₂]
        """
        raw = self.net(self.fourier(zeta))            # [batch, 4]

        if self.hard_bc:
            # 硬边界约束：网络输出 f_i(ζ)，乘以 ζ 自动满足 IC
            f1, f2, f3, f4 = raw[:, 0:1], raw[:, 1:2], raw[:, 2:3], raw[:, 3:4]
            u1 = 1.0 + zeta * f1      # A₁(0) = 1+0i
            v1 = zeta * f2
            u2 = zeta * f3             # A₂(0) = 0+0i
            v2 = zeta * f4
            return torch.cat([u1, v1, u2, v2], dim=1)
        else:
            return raw


# ============================================================
#  第四部分：PINN 损失函数（核心物理约束）
# ============================================================
class SHGLoss:
    """
    SHG PINN 的损失函数

    总损失 = λ_pde · L_pde  +  λ_bc · L_bc  +  λ_mr · L_mr

    L_pde:  PDE 残差   —— 耦合波方程在配置点(collocation points)上的残差
    L_bc:   初始条件   —— A₁(0)=1, A₂(0)=0  (硬约束模式下此项为 0)
    L_mr:   Manley-Rowe 能量守恒  —— 软正则化项
    """

    def __init__(self, params, lambda_pde=1.0, lambda_bc=100.0, lambda_mr=10.0):
        self.Ga    = params.Gamma_alpha
        self.Gb    = params.Gamma_beta
        self.sigma = params.sigma
        self.mr_c  = params.mr_coeff       # n₂/n₁

        self.lambda_pde = lambda_pde
        self.lambda_bc  = lambda_bc
        self.lambda_mr  = lambda_mr

    # ---- PDE 残差 ----
    def pde_residual(self, model, zeta):
        """
        利用 autograd 精确微分，计算耦合波方程的残差。

        这就是 PINN 的精髓：
            用 自动微分 取代 有限差分，获得解析级别的导数精度，
            彻底消灭截断误差和数值色散。

        Args:
            model : 神经网络
            zeta  : [N, 1]  需要 requires_grad
        Returns:
            [N, 4]  四个实数方程的残差
        """
        zeta = zeta.detach().requires_grad_(True)
        out  = model(zeta)

        u1, v1 = out[:, 0:1], out[:, 1:2]
        u2, v2 = out[:, 2:3], out[:, 3:4]

        # ---------- autograd 求 dA/dζ ----------
        ones = torch.ones_like(u1)
        du1 = torch.autograd.grad(u1, zeta, ones, create_graph=True)[0]
        dv1 = torch.autograd.grad(v1, zeta, ones, create_graph=True)[0]
        du2 = torch.autograd.grad(u2, zeta, ones, create_graph=True)[0]
        dv2 = torch.autograd.grad(v2, zeta, ones, create_graph=True)[0]

        # ---------- 三角函数 ----------
        c = torch.cos(self.sigma * zeta)
        s = torch.sin(self.sigma * zeta)

        # ---------- 方程 1: dA₁/dζ = -iΓα · (A₂·A₁*) · exp(-iσζ) ----------
        #   A₂·A₁* = (u₂+iv₂)(u₁-iv₁) = R₁ + iI₁
        R1 = u2 * u1 + v2 * v1
        I1 = v2 * u1 - u2 * v1
        #   乘以 exp(-iσζ) 再乘以 -i → 实部/虚部
        rhs_u1 =  self.Ga * (I1 * c - R1 * s)
        rhs_v1 = -self.Ga * (R1 * c + I1 * s)

        # ---------- 方程 2: dA₂/dζ = -iΓβ · A₁² · exp(+iσζ) ----------
        #   A₁² = (u₁+iv₁)² = R₂ + iI₂
        R2 = u1**2 - v1**2
        I2 = 2 * u1 * v1
        rhs_u2 =  self.Gb * (I2 * c + R2 * s)
        rhs_v2 = -self.Gb * (R2 * c - I2 * s)

        # ---------- 残差 = 左端(autograd导数) - 右端(物理方程) ----------
        return torch.cat([du1 - rhs_u1, dv1 - rhs_v1,
                          du2 - rhs_u2, dv2 - rhs_v2], dim=1)

    # ---- 初始条件 (软约束，hard_bc 模式下可关闭) ----
    def bc_loss(self, model):
        z0  = torch.zeros(1, 1, device=device)
        out = model(z0)
        return ((out[0, 0] - 1.0)**2 + out[0, 1]**2
                + out[0, 2]**2 + out[0, 3]**2)

    # ---- Manley-Rowe 能量守恒 ----
    def mr_loss(self, model, zeta):
        """
        |A₁|² + (n₂/n₁)|A₂|² = 1  ∀ ζ

        这是耦合波方程的数学推论，作为额外软约束可以加速收敛。
        """
        with torch.no_grad():
            zeta_eval = zeta.detach()
        out = model(zeta_eval)
        I1 = out[:, 0]**2 + out[:, 1]**2
        I2 = out[:, 2]**2 + out[:, 3]**2
        return torch.mean((I1 + self.mr_c * I2 - 1.0)**2)

    # ---- 总损失 ----
    def total_loss(self, model, zeta):
        res  = self.pde_residual(model, zeta)
        l_pde = torch.mean(res**2)
        l_bc  = self.bc_loss(model)
        l_mr  = self.mr_loss(model, zeta)

        total = (self.lambda_pde * l_pde
                 + self.lambda_bc * l_bc
                 + self.lambda_mr * l_mr)
        return total, l_pde.item(), l_bc.item(), l_mr.item()


# ============================================================
#  第五部分：训练流程
# ============================================================
def train(model, loss_fn, n_adam=5000, n_lbfgs=2000,
          lr_adam=1e-3, lr_lbfgs=0.5, n_pts=256):
    """
    两阶段训练策略：
      阶段 1  Adam   —— 快速全局探索，跳出初始化附近的局部极小
      阶段 2  L-BFGS —— 精细局部优化，达到高精度拟合

    配置点 (collocation points) 采用 均匀网格 + 随机扰动 + 边界加密。
    """
    model.to(device)
    hist = {'total': [], 'pde': [], 'bc': [], 'mr': []}

    # ---- 配置点 ----
    z_uniform  = torch.linspace(0, 1, n_pts, device=device).reshape(-1, 1)
    z_boundary = torch.linspace(0, 0.05, n_pts // 4, device=device).reshape(-1, 1)
    z_base     = torch.cat([z_uniform, z_boundary], dim=0)

    n_total = z_base.shape[0]
    print(f"\n配置点: {n_total} (均匀 {n_pts} + 边界 {n_pts // 4})")

    # ========== 阶段 1: Adam ==========
    print(f"\n{'='*45}")
    print(f" 阶段 1: Adam  (epochs={n_adam}, lr={lr_adam})")
    print(f"{'='*45}")

    opt = optim.Adam(model.parameters(), lr=lr_adam)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_adam,
                                                  eta_min=lr_adam * 0.01)
    t0 = time.time()
    for ep in range(1, n_adam + 1):
        opt.zero_grad()

        # 每个 epoch 加入随机配置点，增强泛化
        z_rand = torch.rand(n_pts // 2, 1, device=device)
        z = torch.cat([z_base, z_rand], dim=0)

        total, pde, bc, mr = loss_fn.total_loss(model, z)
        total.backward()
        opt.step()
        sched.step()

        hist['total'].append(total.item())
        hist['pde'].append(pde)
        hist['bc'].append(bc)
        hist['mr'].append(mr)

        if ep % 1000 == 0 or ep == 1:
            print(f"  [{ep:5d}/{n_adam}]  Loss={total.item():.3e}  "
                  f"PDE={pde:.3e}  BC={bc:.3e}  MR={mr:.3e}")

    print(f"  Adam 完成，耗时 {time.time()-t0:.1f}s")

    # ========== 阶段 2: L-BFGS ==========
    print(f"\n{'='*45}")
    print(f" 阶段 2: L-BFGS  (epochs={n_lbfgs}, lr={lr_lbfgs})")
    print(f"{'='*45}")

    opt2 = optim.LBFGS(model.parameters(), lr=lr_lbfgs,
                        max_iter=20, history_size=50,
                        line_search_fn='strong_wolfe')
    cache = {}

    t0 = time.time()
    for ep in range(1, n_lbfgs + 1):
        def closure():
            opt2.zero_grad()
            total, pde, bc, mr = loss_fn.total_loss(model, z_base)
            total.backward()
            cache['t'] = total.item()
            cache['p'] = pde
            cache['b'] = bc
            cache['m'] = mr
            return total

        opt2.step(closure)

        hist['total'].append(cache['t'])
        hist['pde'].append(cache['p'])
        hist['bc'].append(cache['b'])
        hist['mr'].append(cache['m'])

        if ep % 500 == 0 or ep == 1:
            print(f"  [{ep:5d}/{n_lbfgs}]  Loss={cache['t']:.3e}  "
                  f"PDE={cache['p']:.3e}  BC={cache['b']:.3e}  MR={cache['m']:.3e}")

    print(f"  L-BFGS 完成，耗时 {time.time()-t0:.1f}s")
    return hist


# ============================================================
#  第六部分：可视化
# ============================================================
def visualize(model, params, hist, save_dir='./results'):
    """比较 PINN 预测 vs RK45 参考解 vs 解析解(σ=0时)"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # --- 参考解 (RK45) ---
    z_ref, u1r, v1r, u2r, v2r = solve_shg_rk45(params)
    I1_ref = u1r**2 + v1r**2
    I2_ref = u2r**2 + v2r**2

    # --- PINN 预测 ---
    z_t = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
    with torch.no_grad():
        pred = model(z_t).cpu().numpy()
    z_p = z_t.cpu().numpy().flatten()
    I1_p = pred[:, 0]**2 + pred[:, 1]**2
    I2_p = pred[:, 2]**2 + pred[:, 3]**2

    # --- 解析解 (仅 σ=0 且 Γα=Γβ) ---
    has_analytic = (params.sigma == 0
                    and params.Gamma_alpha == params.Gamma_beta)
    if has_analytic:
        G = params.Gamma_alpha
        I1_a = 1.0 / np.cosh(G * z_ref)**2      # sech²
        I2_a = np.tanh(G * z_ref)**2             # tanh²

    mr_c = params.mr_coeff

    # ---- 绘图 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PINN for 1-D CW Second Harmonic Generation',
                 fontsize=14, fontweight='bold')

    # (a) 归一化光强
    ax = axes[0, 0]
    ax.plot(z_ref, I1_ref, 'b-',  lw=2, alpha=0.6, label=r'$|A_1|^2$ RK45')
    ax.plot(z_ref, I2_ref, 'r-',  lw=2, alpha=0.6, label=r'$|A_2|^2$ RK45')
    ax.plot(z_p,   I1_p,   'b--', lw=2, label=r'$|A_1|^2$ PINN')
    ax.plot(z_p,   I2_p,   'r--', lw=2, label=r'$|A_2|^2$ PINN')
    if has_analytic:
        ax.plot(z_ref, I1_a, 'c:', lw=1.5, label=r'$|A_1|^2$ Analytic')
        ax.plot(z_ref, I2_a, 'm:', lw=1.5, label=r'$|A_2|^2$ Analytic')
    ax.set_xlabel(r'$\zeta = z/L$')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('(a) Intensity Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) 转换效率
    ax = axes[0, 1]
    eta_ref  = mr_c * I2_ref * 100
    eta_pinn = mr_c * I2_p   * 100
    ax.plot(z_ref, eta_ref,  'k-',  lw=2, label='RK45')
    ax.plot(z_p,   eta_pinn, 'r--', lw=2, label='PINN')
    if has_analytic:
        ax.plot(z_ref, mr_c * I2_a * 100, 'g:', lw=1.5, label='Analytic')
    ax.set_xlabel(r'$\zeta = z/L$')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title(r'(b) SHG Conversion Efficiency $\eta$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) 复振幅分量
    ax = axes[1, 0]
    ax.plot(z_ref, u1r,       'b-',  lw=1.5, alpha=0.6, label=r'Re$(A_1)$ RK45')
    ax.plot(z_ref, v2r,       'r-',  lw=1.5, alpha=0.6, label=r'Im$(A_2)$ RK45')
    ax.plot(z_p,   pred[:,0], 'b--', lw=1.5, label=r'Re$(A_1)$ PINN')
    ax.plot(z_p,   pred[:,3], 'r--', lw=1.5, label=r'Im$(A_2)$ PINN')
    ax.set_xlabel(r'$\zeta = z/L$')
    ax.set_ylabel('Field Amplitude')
    ax.set_title('(c) Complex Amplitude Components')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) 训练损失曲线
    ax = axes[1, 1]
    ep = np.arange(1, len(hist['total']) + 1)
    ax.semilogy(ep, hist['total'], 'k-', lw=0.8, alpha=0.8, label='Total')
    ax.semilogy(ep, hist['pde'],   'b-', lw=0.8, alpha=0.6, label='PDE')
    ax.semilogy(ep, hist['bc'],    'r-', lw=0.8, alpha=0.6, label='BC')
    ax.semilogy(ep, hist['mr'],    'g-', lw=0.8, alpha=0.6, label='Manley-Rowe')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(d) Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'pinn_shg_1d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n图片已保存: {path}")

    # ---- 精度评估 ----
    from scipy.interpolate import interp1d
    I2_interp = interp1d(z_p, I2_p)(z_ref)
    rel_err = np.abs(I2_interp - I2_ref) / (np.max(I2_ref) + 1e-15)
    print(f"\n{'='*40}")
    print(f"  精度评估")
    print(f"{'='*40}")
    print(f"  |A₂|² 最大相对误差:  {np.max(rel_err)*100:.4f}%")
    print(f"  |A₂|² 平均相对误差:  {np.mean(rel_err)*100:.4f}%")
    print(f"  最终转换效率 (RK45):  {eta_ref[-1]:.2f}%")
    print(f"  最终转换效率 (PINN):  {eta_pinn[-1]:.2f}%")
    mr_pinn = I1_p + mr_c * I2_p
    print(f"  Manley-Rowe 偏差: max|M-R - 1| = {np.max(np.abs(mr_pinn - 1)):.2e}")
    print(f"{'='*40}")


# ============================================================
#  第七部分：主程序
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  PINN for 1-D CW Second Harmonic Generation (SHG)")
    print("  物理信息神经网络 · 一维连续波二次谐波产生")
    print("=" * 60)

    # ============ 可调参数 ============
    # 如需使用真实 KTP 参数，改为 mode='physical'
    params = SHGParams(mode='normalized')
    params.Gamma_alpha = 3.0      # 非线性耦合强度
    params.Gamma_beta  = 3.0
    params.sigma       = 0.0      # 相位失配 (0=完美匹配, 试试 20, 50)
    # ==================================

    params.summary()

    # ---- 构建网络 ----
    model = PINN_SHG(
        hidden_dim=128,           # 隐藏层宽度
        num_layers=5,             # 隐藏层数
        num_fourier=64,           # 傅里叶特征维度
        fourier_scale=4.0,        # 傅里叶尺度 (σ>0 时可适当增大)
        hard_bc=True              # True: 硬边界约束; False: 软约束
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n网络参数量: {n_params:,}")

    # ---- 损失函数 ----
    loss_fn = SHGLoss(
        params,
        lambda_pde=1.0,           # PDE 残差权重
        lambda_bc=0.0 if model.hard_bc else 100.0,   # 硬约束时 BC=0
        lambda_mr=10.0            # Manley-Rowe 权重
    )

    # ---- 训练 ----
    hist = train(
        model, loss_fn,
        n_adam=5000,              # Adam 迭代数
        n_lbfgs=2000,             # L-BFGS 迭代数
        lr_adam=1e-3,
        lr_lbfgs=0.5,
        n_pts=256                 # 配置点数
    )

    # ---- 可视化 ----
    visualize(model, params, hist)

    # ---- 保存模型 ----
    save_path = './results/pinn_shg_1d_model.pth'
    torch.save({
        'model_state': model.state_dict(),
        'params': {'Ga': params.Gamma_alpha,
                   'Gb': params.Gamma_beta,
                   'sigma': params.sigma},
        'history': hist
    }, save_path)
    print(f"模型已保存: {save_path}")

    # ---- 路线图 ----
    print(f"\n{'='*60}")
    print("  下一步路线图:")
    print("    Step 2 → 加入 x 维度走离效应       →  2D PINN")
    print("    Step 3 → 加入横向衍射 ∇⊥²A         →  3D PINN")
    print("    Step 4 → 加入时间脉冲演化           →  脉冲 SHG PINN")
    print("    Step 5 → 实验数据同化 (Data Loss)   →  反问题求解")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
