# plot_lambda_sensitivity.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# ===============================
# 1. 构造数据（直接来自你的实验结果）
# ===============================
data = {
    'dataset': ['Instruments'] * 6 + ['Scientific'] * 6 + ['Video Games'] * 6,
    'lambda':  [0, 4, 8, 12, 16, 20] * 3,

    'recall@10': [
        # Instruments
        0.04271, 0.04232, 0.03976, 0.03983, 0.03954, 0.03938,
        # Scientific
        0.04913, 0.04748, 0.04429, 0.03985, 0.03507, 0.03258,
        # Video Games
        0.07261, 0.06830, 0.06488, 0.06282, 0.06051, 0.05975
    ],

    'diversity@10': [
        # Instruments
        0.4060, 0.4778, 0.5440, 0.5795, 0.5935, 0.5941,
        # Scientific
        0.4112, 0.4488, 0.4928, 0.5336, 0.5633, 0.5811,
        # Video Games
        0.2622, 0.3185, 0.3773, 0.4293, 0.4613, 0.4783
    ]
}

df = pd.DataFrame(data)

# ===============================
# 2. 风格与全局参数（严格对齐你给的模板）
# ===============================
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

color_recall = '#1f77b4'     # 蓝色
color_diversity = '#ff7f0e'  # 橙色

# ===============================
# 3. 核心绘图函数（双轴）
# ===============================
def create_dual_axis_subplot(ax, data, title,
                             show_y_left=True, show_y_right=True):

    x = data['lambda']

    # 左轴：Recall@10
    ax1 = ax
    ax1.plot(
        x, data['recall@10'],
        marker='o', linewidth=2.0, markersize=5,
        color=color_recall, label='Recall@10'
    )
    if show_y_left:
        ax1.set_ylabel('Recall@10 (↑)', fontsize=16)
    ax1.set_xlabel(r'Guidance Strength $\lambda$', fontsize=16)
    ax1.set_xticks(x)  # 或者 ax1.set_xticks(data['lambda'].tolist())

    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_title(title, fontsize=16, pad=10)

    # 右轴：Diversity@10
    ax2 = ax1.twinx()
    ax2.plot(
        x, data['diversity@10'],
        marker='s', linestyle='--', linewidth=2.0, markersize=5,
        color=color_diversity, label='Diversity@10'
    )

    if show_y_right:
        ax2.set_ylabel('Diversity@10 (↑)', fontsize=16)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.grid(False)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    return ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()

# ===============================
# 4. 画布（1×3，单栏论文友好）
# ===============================
fig, axes = plt.subplots(
    1, 3,
    figsize=(10, 2.6),
    dpi=300,
    sharey=False
)

datasets = ['Instruments', 'Scientific', 'Video Games']

# 第一个子图：显示左轴
(l1, la1), (l2, la2) = create_dual_axis_subplot(
    axes[0],
    df[df['dataset'] == 'Instruments'],
    'Instruments',
    show_y_right=False
)

# 第二个子图：不显示 y 轴 label
create_dual_axis_subplot(
    axes[1],
    df[df['dataset'] == 'Scientific'],
    'Scientific',
    show_y_left=False,
    show_y_right=False
)

# 第三个子图：显示右轴
create_dual_axis_subplot(
    axes[2],
    df[df['dataset'] == 'Video Games'],
    'Video Games',
    show_y_left=False
)

# ===============================
# 5. 统一图例（底部居中，IJCAI 友好）
# ===============================
fig.legend(
    l1 + l2, la1 + la2,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=2,
    fontsize=16,
    frameon=False
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.subplots_adjust(wspace=0.5)

# ===============================
# 6. 保存
# ===============================
plt.savefig(
    'lambda_sensitivity_1x3.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.close()

print("lambda_sensitivity_1x3.pdf generated successfully")
