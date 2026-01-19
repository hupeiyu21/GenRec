import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# ===============================
# 1. 数据准备
# ===============================
steps = [1, 2, 3, 4, 6, 8]

# Instruments
inst_recall = [0.0182, 0.0140, 0.0356, 0.0361, 0.0340, 0.0352]
inst_div    = [0.4926, 0.8489, 0.5665, 0.5589, 0.5607, 0.5608]
inst_time   = ["41m", "49m", "64m", "72m", "94m", "114m"]
# Scientific
sci_recall = [0.0392, 0.0319, 0.0441, 0.0422, 0.0414, 0.0420]
sci_div    = [0.5283, 0.7065, 0.5097, 0.5046, 0.5051, 0.5057]
sci_time   = ["37m", "45m", "54m", "65m", "89m", "102m"]
# Video Games
game_recall = [0.0544, 0.0583, 0.0642, 0.0627, 0.0636, 0.0632]
game_div    = [0.3677, 0.4780, 0.3432, 0.3419, 0.3417, 0.3413]
game_time   = ["77m", "79m", "99m", "128m", "152m", "189m"]
data = {
    'dataset': ['Instruments'] * 6 + ['Scientific'] * 6 + ['Video Games'] * 6,
    'steps': steps * 3,
    'recall@10': inst_recall + sci_recall + game_recall,
    'diversity@10': inst_div + sci_div + game_div
}

df = pd.DataFrame(data)
# 时间数据字典，用于绘图时查找
time_map = {
    'Instruments': inst_time,
    'Scientific': sci_time,
    'Video Games': game_time
}
# ===============================
# 2. 全局绘图风格
# ===============================
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

color_recall = '#1f77b4'    # 蓝色 (Accuracy)
color_diversity = '#ff7f0e'  # 橙色 (Diversity)

# ===============================
# 3. 核心绘图函数
# ===============================
def create_dual_axis_subplot(ax, data, title, time_labels, show_y_left=True, show_y_right=True):
    # 关键修改：使用 range(len(steps)) 生成等间距坐标
    x_indices = range(len(steps))
    y_rec = data['recall@10']
    y_div = data['diversity@10']

    # --- 左轴：Recall@10 ---
    ax1 = ax
    l1 = ax1.plot(
        x_indices, y_rec, # 使用 index 作为 x 坐标
        marker='o', linewidth=2.0, markersize=6,
        color=color_recall, label='Recall@10'
    )
    
    # Y轴范围设置
    max_rec = max(y_rec)
    ax1.set_ylim(0, max_rec * 1.35) 
    
    # 关键修改：根据 show_y_left 决定是否显示 Label
    if show_y_left:
        ax1.set_ylabel('Recall@10 (↑)', fontsize=16)
    # 始终显示刻度颜色，方便对应
    ax1.tick_params(axis='y', labelsize=14)

    ax1.set_xlabel(r'Inference Steps $T$', fontsize=16)
    
    # 设置刻度为 0, 1, 2... 并映射回 Step+Time
    ax1.set_xticks(x_indices) 
    
    # 构造双行标签: "Step\nTime" (去掉括号，字号缩小)
    new_xticklabels = [f"{s}\n{t}" for s, t in zip(steps, time_labels)]
    ax1.set_xticklabels(new_xticklabels, fontsize=14) 

    ax1.set_title(title, fontsize=16, pad=10)

    # 标记 Sweet Spot (T=4)，现在它的 index 是 3 (0,1,2,3)
    ax1.axvline(x=3, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)

    # --- 右轴：Diversity@10 ---
    ax2 = ax1.twinx()
    l2 = ax2.plot(
        x_indices, y_div, # 使用 index 作为 x 坐标
        marker='s', linestyle='--', linewidth=2.0, markersize=6,
        color=color_diversity, label='Diversity@10'
    )

    # 关键修改：根据 show_y_right 决定是否显示 Label
    if show_y_right:
        ax2.set_ylabel('Diversity@10 (↑)', fontsize=16)
    
    ax2.tick_params(axis='y',labelsize=14)
    # X轴不需要重复设置，因为共享
    ax2.grid(False)

    max_div = max(y_div)
    min_div = min(y_div)
    ax2.set_ylim(min_div * 0.8, max_div * 1.2)

    return l1 + l2, [l.get_label() for l in l1 + l2]

# ===============================
# 4. 创建画布 (1x3)
# ===============================
fig, axes = plt.subplots(
    1, 3,
    figsize=(11, 2.9),
    dpi=300,
    sharey=False
)
plt.subplots_adjust(wspace=0.1) 

datasets = ['Instruments', 'Scientific', 'Video Games']
lines, labels = [], []

for i, ds_name in enumerate(datasets):
    # 逻辑修改：只有第一个图显示左Label，只有最后一个图显示右Label
    show_left = (i == 0)
    show_right = (i == 2)
    
    ds_data = df[df['dataset'] == ds_name]
    # 获取对应数据集的时间标签
    current_time_labels = time_map[ds_name]

    l, lbl = create_dual_axis_subplot(
        axes[i], ds_data, ds_name, 
        current_time_labels, # 传入时间
        show_y_left=show_left, 
        show_y_right=show_right
    )
    
    if i == 0:
        lines, labels = l, lbl

# ===============================
# 5. 统一底部图例
# ===============================
fig.legend(
    lines, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    fontsize=16,
    frameon=False
)

plt.tight_layout(rect=[0, 0.08, 1, 1])

# ===============================
# 6. 保存
# ===============================
save_path = 'refline_steps_sensitivity_1x3.pdf'
plt.savefig(save_path, format='pdf', bbox_inches='tight')
print(f"Successfully generated: {save_path}")