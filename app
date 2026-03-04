"""
Streamlit 可视化界面 —— PINN 求解一维 SHG（二次谐波产生）

运行方式:
    streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
import time, os, sys

# ── 导入原始模块中的类 ──────────────────────────────────
# 把原脚本所在目录加入 path，然后直接复用核心代码
sys.path.insert(0, os.path.dirname(__file__))
from pinn_shg_1d import SHGParams, PINN_SHG, SHGLoss, solve_shg_rk45

# ══════════════════════════════════════════════════════════
#  页面基础配置
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PINN · SHG 1-D 仿真",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 PINN 一维二次谐波产生（SHG）仿真")
st.markdown("""
利用 **物理信息神经网络 (PINN)** 求解一维连续波 SHG 耦合波方程，
并与 RK45 数值解 / 解析解对比。调整左侧参数后点击 **开始训练** 即可。
""")

# ── 设备 ────────────────────────────────────────────────
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════
#  侧边栏：参数面板
# ══════════════════════════════════════════════════════════
st.sidebar.header("⚙️ 物理参数")
mode = st.sidebar.selectbox("参数模式", ["normalized", "physical"],
                            format_func=lambda x: "归一化 (Normalized)" if x == "normalized" else "真实晶体 (KTP)")

if mode == "normalized":
    Gamma_alpha = st.sidebar.slider("Γα（基频耦合强度）", 0.5, 10.0, 3.0, 0.1)
    Gamma_beta  = st.sidebar.slider("Γβ（倍频耦合强度）", 0.5, 10.0, 3.0, 0.1)
    sigma       = st.sidebar.slider("σ（相位失配 ΔkL）", 0.0, 60.0, 0.0, 1.0)
else:
    Gamma_alpha = Gamma_beta = sigma = None  # 由物理参数自动计算

st.sidebar.markdown("---")
st.sidebar.header("🧠 网络结构")
hidden_dim    = st.sidebar.select_slider("隐藏层宽度", [64, 128, 256, 512], value=128)
num_layers    = st.sidebar.slider("隐藏层数", 2, 8, 5)
num_fourier   = st.sidebar.select_slider("傅里叶特征数", [32, 64, 128], value=64)
fourier_scale = st.sidebar.slider("傅里叶尺度", 1.0, 10.0, 4.0, 0.5)
hard_bc       = st.sidebar.checkbox("使用硬边界约束", value=True)

st.sidebar.markdown("---")
st.sidebar.header("🏋️ 训练配置")
n_adam  = st.sidebar.slider("Adam 迭代次数", 500, 10000, 5000, 500)
n_lbfgs = st.sidebar.slider("L-BFGS 迭代次数", 0, 5000, 2000, 500)
lr_adam  = st.sidebar.select_slider("Adam 学习率", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
n_pts    = st.sidebar.select_slider("配置点数", [128, 256, 512, 1024], value=256)
lambda_mr = st.sidebar.slider("Manley-Rowe 权重", 0.0, 50.0, 10.0, 1.0)

# ══════════════════════════════════════════════════════════
#  构建参数对象
# ══════════════════════════════════════════════════════════
params = SHGParams(mode=mode)
if mode == "normalized":
    params.Gamma_alpha = Gamma_alpha
    params.Gamma_beta  = Gamma_beta
    params.sigma       = sigma

# ── 显示参数摘要 ─────────────────────────────────────────
col_info1, col_info2 = st.columns(2)
with col_info1:
    st.markdown("### 📋 当前参数")
    st.markdown(f"""
| 参数 | 值 |
|------|-----|
| Γα | `{params.Gamma_alpha:.4f}` |
| Γβ | `{params.Gamma_beta:.4f}` |
| σ (ΔkL) | `{params.sigma:.4f}` |
| n₂/n₁ | `{params.mr_coeff:.4f}` |
""")
with col_info2:
    has_analytic = (params.sigma == 0 and params.Gamma_alpha == params.Gamma_beta)
    if has_analytic:
        eta_theory = np.tanh(params.Gamma_alpha) ** 2
        st.markdown("### 📐 理论解析解")
        st.latex(r"A_1(\zeta) = \mathrm{sech}(\Gamma\zeta)")
        st.latex(r"A_2(\zeta) = -i\,\tanh(\Gamma\zeta)")
        st.metric("理论转换效率 η", f"{eta_theory*100:.2f}%")
    else:
        st.markdown("### ℹ️ 说明")
        st.info("σ ≠ 0 或 Γα ≠ Γβ 时没有简单解析解，仅对比 RK45。")


# ══════════════════════════════════════════════════════════
#  训练 & 可视化
# ══════════════════════════════════════════════════════════
if st.button("🚀 开始训练", type="primary", use_container_width=True):

    # -------- 构建模型 --------
    torch.manual_seed(42)
    np.random.seed(42)
    model = PINN_SHG(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_fourier=num_fourier,
        fourier_scale=fourier_scale,
        hard_bc=hard_bc,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    st.info(f"网络参数量: **{n_params:,}**　|　设备: **{device}**")

    loss_fn = SHGLoss(
        params,
        lambda_pde=1.0,
        lambda_bc=0.0 if hard_bc else 100.0,
        lambda_mr=lambda_mr,
    )

    # -------- 配置点 --------
    z_uniform  = torch.linspace(0, 1, n_pts, device=device).reshape(-1, 1)
    z_boundary = torch.linspace(0, 0.05, n_pts // 4, device=device).reshape(-1, 1)
    z_base     = torch.cat([z_uniform, z_boundary], dim=0)

    hist = {'total': [], 'pde': [], 'bc': [], 'mr': []}

    # -------- 进度条 & 占位 --------
    progress_bar  = st.progress(0, text="准备中...")
    status_text   = st.empty()
    loss_chart    = st.empty()           # 实时损失曲线

    total_epochs = n_adam + n_lbfgs
    t_start = time.time()

    # ======== 阶段 1: Adam ========
    opt   = optim.Adam(model.parameters(), lr=lr_adam)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_adam,
                                                  eta_min=lr_adam * 0.01)

    for ep in range(1, n_adam + 1):
        opt.zero_grad()
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

        # 更新 UI（每 100 步或最后一步）
        if ep % 100 == 0 or ep == n_adam:
            pct = int(ep / total_epochs * 100)
            progress_bar.progress(pct, text=f"Adam {ep}/{n_adam}")
            status_text.text(
                f"Adam [{ep}/{n_adam}]  Loss={total.item():.3e}  "
                f"PDE={pde:.3e}  BC={bc:.3e}  MR={mr:.3e}  "
                f"({time.time()-t_start:.1f}s)")

            # 实时损失小图
            fig_loss_live = go.Figure()
            epochs_arr = list(range(1, len(hist['total']) + 1))
            fig_loss_live.add_trace(go.Scatter(x=epochs_arr, y=hist['total'],
                                               mode='lines', name='Total'))
            fig_loss_live.add_trace(go.Scatter(x=epochs_arr, y=hist['pde'],
                                               mode='lines', name='PDE'))
            fig_loss_live.update_layout(
                yaxis_type='log', height=250,
                margin=dict(l=40, r=20, t=30, b=30),
                title_text="训练损失 (实时)", title_font_size=13)
            loss_chart.plotly_chart(fig_loss_live, use_container_width=True)

    # ======== 阶段 2: L-BFGS ========
    if n_lbfgs > 0:
        opt2 = optim.LBFGS(model.parameters(), lr=0.5,
                            max_iter=20, history_size=50,
                            line_search_fn='strong_wolfe')
        cache = {}

        for ep in range(1, n_lbfgs + 1):
            def closure():
                opt2.zero_grad()
                tot, p, b, m = loss_fn.total_loss(model, z_base)
                tot.backward()
                cache.update(t=tot.item(), p=p, b=b, m=m)
                return tot

            opt2.step(closure)
            hist['total'].append(cache['t'])
            hist['pde'].append(cache['p'])
            hist['bc'].append(cache['b'])
            hist['mr'].append(cache['m'])

            if ep % 100 == 0 or ep == n_lbfgs:
                pct = int((n_adam + ep) / total_epochs * 100)
                progress_bar.progress(min(pct, 100),
                                      text=f"L-BFGS {ep}/{n_lbfgs}")
                status_text.text(
                    f"L-BFGS [{ep}/{n_lbfgs}]  Loss={cache['t']:.3e}  "
                    f"PDE={cache['p']:.3e}  BC={cache['b']:.3e}  MR={cache['m']:.3e}  "
                    f"({time.time()-t_start:.1f}s)")

                epochs_arr = list(range(1, len(hist['total']) + 1))
                fig_loss_live = go.Figure()
                fig_loss_live.add_trace(go.Scatter(x=epochs_arr, y=hist['total'],
                                                   mode='lines', name='Total'))
                fig_loss_live.add_trace(go.Scatter(x=epochs_arr, y=hist['pde'],
                                                   mode='lines', name='PDE'))
                fig_loss_live.update_layout(
                    yaxis_type='log', height=250,
                    margin=dict(l=40, r=20, t=30, b=30),
                    title_text="训练损失 (实时)", title_font_size=13)
                loss_chart.plotly_chart(fig_loss_live, use_container_width=True)

    elapsed = time.time() - t_start
    progress_bar.progress(100, text="✅ 训练完成!")
    status_text.success(f"训练完成！总耗时 {elapsed:.1f}s，最终损失 {hist['total'][-1]:.3e}")

    # ══════════════════════════════════════════════════════
    #  结果可视化
    # ══════════════════════════════════════════════════════
    st.markdown("---")
    st.header("📊 结果对比")

    model.eval()

    # ---- RK45 参考解 ----
    z_ref, u1r, v1r, u2r, v2r = solve_shg_rk45(params)
    I1_ref = u1r**2 + v1r**2
    I2_ref = u2r**2 + v2r**2

    # ---- PINN 预测 ----
    z_t = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
    with torch.no_grad():
        pred = model(z_t).cpu().numpy()
    z_p = z_t.cpu().numpy().flatten()
    I1_p = pred[:, 0]**2 + pred[:, 1]**2
    I2_p = pred[:, 2]**2 + pred[:, 3]**2

    has_analytic = (params.sigma == 0 and params.Gamma_alpha == params.Gamma_beta)
    if has_analytic:
        G = params.Gamma_alpha
        I1_a = 1.0 / np.cosh(G * z_ref)**2
        I2_a = np.tanh(G * z_ref)**2

    mr_c = params.mr_coeff

    # ════════════ 图 1: 归一化光强 ════════════
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=z_ref, y=I1_ref, mode='lines',
                              name='|A₁|² RK45', line=dict(color='royalblue', width=3)))
    fig1.add_trace(go.Scatter(x=z_ref, y=I2_ref, mode='lines',
                              name='|A₂|² RK45', line=dict(color='crimson', width=3)))
    fig1.add_trace(go.Scatter(x=z_p, y=I1_p, mode='lines',
                              name='|A₁|² PINN', line=dict(color='lime', width=2, dash='dash')))
    fig1.add_trace(go.Scatter(x=z_p, y=I2_p, mode='lines',
                              name='|A₂|² PINN', line=dict(color='orange', width=2, dash='dash')))
    if has_analytic:
        fig1.add_trace(go.Scatter(x=z_ref, y=I1_a, mode='lines',
                                  name='|A₁|² Analytic', line=dict(color='cyan', width=1.5, dash='dot')))
        fig1.add_trace(go.Scatter(x=z_ref, y=I2_a, mode='lines',
                                  name='|A₂|² Analytic', line=dict(color='magenta', width=1.5, dash='dot')))
    fig1.update_layout(
        title="(a) 归一化光强演化",
        xaxis_title="ζ = z/L",
        yaxis_title="Normalized Intensity",
        height=450, template="plotly_white")

    # ════════════ 图 2: 转换效率 ════════════
    eta_ref  = mr_c * I2_ref * 100
    eta_pinn = mr_c * I2_p * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=z_ref, y=eta_ref, mode='lines',
                              name='RK45', line=dict(color='black', width=2.5)))
    fig2.add_trace(go.Scatter(x=z_p, y=eta_pinn, mode='lines',
                              name='PINN', line=dict(color='red', width=2, dash='dash')))
    if has_analytic:
        fig2.add_trace(go.Scatter(x=z_ref, y=mr_c * I2_a * 100, mode='lines',
                                  name='Analytic', line=dict(color='green', width=1.5, dash='dot')))
    fig2.update_layout(
        title="(b) SHG 转换效率 η",
        xaxis_title="ζ = z/L",
        yaxis_title="Efficiency (%)",
        height=450, template="plotly_white")

    # ════════════ 图 3: 复振幅全部 4 分量 ════════════
    fig3 = go.Figure()
    # --- RK45 参考 (实线) ---
    fig3.add_trace(go.Scatter(x=z_ref, y=u1r, mode='lines',
                              name='u₁ = Re(A₁) RK45', line=dict(color='royalblue', width=2.5)))
    fig3.add_trace(go.Scatter(x=z_ref, y=v1r, mode='lines',
                              name='v₁ = Im(A₁) RK45', line=dict(color='dodgerblue', width=2.5, dash='dot')))
    fig3.add_trace(go.Scatter(x=z_ref, y=u2r, mode='lines',
                              name='u₂ = Re(A₂) RK45', line=dict(color='crimson', width=2.5)))
    fig3.add_trace(go.Scatter(x=z_ref, y=v2r, mode='lines',
                              name='v₂ = Im(A₂) RK45', line=dict(color='orangered', width=2.5, dash='dot')))
    # --- PINN 预测 (虚线) ---
    fig3.add_trace(go.Scatter(x=z_p, y=pred[:, 0], mode='lines',
                              name='u₁ = Re(A₁) PINN', line=dict(color='lime', width=2, dash='dash')))
    fig3.add_trace(go.Scatter(x=z_p, y=pred[:, 1], mode='lines',
                              name='v₁ = Im(A₁) PINN', line=dict(color='springgreen', width=2, dash='dashdot')))
    fig3.add_trace(go.Scatter(x=z_p, y=pred[:, 2], mode='lines',
                              name='u₂ = Re(A₂) PINN', line=dict(color='orange', width=2, dash='dash')))
    fig3.add_trace(go.Scatter(x=z_p, y=pred[:, 3], mode='lines',
                              name='v₂ = Im(A₂) PINN', line=dict(color='gold', width=2, dash='dashdot')))
    fig3.update_layout(
        title="(c) 复振幅全部 4 分量 (u₁, v₁, u₂, v₂)",
        xaxis_title="ζ = z/L",
        yaxis_title="Field Amplitude",
        height=500, template="plotly_white",
        legend=dict(font=dict(size=10)))

    # ════════════ 图 4: 训练损失 ════════════
    epochs_arr = list(range(1, len(hist['total']) + 1))
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=epochs_arr, y=hist['total'], mode='lines',
                              name='Total', line=dict(color='black', width=1.5)))
    fig4.add_trace(go.Scatter(x=epochs_arr, y=hist['pde'], mode='lines',
                              name='PDE', line=dict(color='royalblue', width=1)))
    fig4.add_trace(go.Scatter(x=epochs_arr, y=hist['bc'], mode='lines',
                              name='BC', line=dict(color='crimson', width=1)))
    fig4.add_trace(go.Scatter(x=epochs_arr, y=hist['mr'], mode='lines',
                              name='Manley-Rowe', line=dict(color='green', width=1)))
    fig4.update_layout(
        title="(d) 训练收敛曲线",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",
        height=450, template="plotly_white")

    # ---- 2×2 布局显示 ----
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)

    # ══════════════════════════════════════════════════════
    #  精度评估
    # ══════════════════════════════════════════════════════
    st.markdown("---")
    st.header("📏 精度评估")

    from scipy.interpolate import interp1d
    I2_interp = interp1d(z_p, I2_p)(z_ref)
    rel_err = np.abs(I2_interp - I2_ref) / (np.max(I2_ref) + 1e-15)
    mr_pinn = I1_p + mr_c * I2_p

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("|A₂|² 最大相对误差", f"{np.max(rel_err)*100:.4f}%")
    mc2.metric("|A₂|² 平均相对误差", f"{np.mean(rel_err)*100:.4f}%")
    mc3.metric("最终转换效率 (RK45)", f"{eta_ref[-1]:.2f}%")
    mc4.metric("最终转换效率 (PINN)", f"{eta_pinn[-1]:.2f}%")

    st.markdown(f"**Manley-Rowe 守恒偏差**: max|M-R − 1| = `{np.max(np.abs(mr_pinn - 1)):.2e}`")

    # ---- 误差分布图 ----
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(x=z_ref, y=rel_err * 100, mode='lines',
                                 fill='tozeroy',
                                 line=dict(color='orange', width=1.5),
                                 name='Relative Error'))
    fig_err.update_layout(
        title="|A₂|² 相对误差沿传播方向的分布",
        xaxis_title="ζ = z/L",
        yaxis_title="Relative Error (%)",
        height=350, template="plotly_white")
    st.plotly_chart(fig_err, use_container_width=True)

    # ══════════════════════════════════════════════════════
    #  Manley-Rowe 守恒验证
    # ══════════════════════════════════════════════════════
    fig_mr = go.Figure()
    fig_mr.add_trace(go.Scatter(x=z_p, y=mr_pinn, mode='lines',
                                name='|A₁|² + (n₂/n₁)|A₂|²',
                                line=dict(color='teal', width=2)))
    fig_mr.add_hline(y=1.0, line_dash="dash", line_color="gray",
                     annotation_text="理论值 = 1")
    fig_mr.update_layout(
        title="Manley-Rowe 能量守恒验证",
        xaxis_title="ζ = z/L",
        yaxis_title="|A₁|² + (n₂/n₁)|A₂|²",
        height=350, template="plotly_white")
    st.plotly_chart(fig_mr, use_container_width=True)

    st.balloons()
