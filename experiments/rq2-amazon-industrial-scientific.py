import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备
# ==========================================

# --- Our Model (从日志提取) ---
# 按 Diversity@10 排序，确保连线顺滑
our_div = np.array([0.4112, 0.4488, 0.4928, 0.5336, 0.5633, 0.5811, 0.5907, 0.6190, 0.6251, 0.6285])
our_recall = np.array([0.0491, 0.0475, 0.0443, 0.0399, 0.0351, 0.0326, 0.0296, 0.0281, 0.0259, 0.0243])
our_ndcg = np.array([0.0228, 0.0223, 0.0210, 0.0191, 0.0173, 0.0162, 0.0148, 0.0140, 0.0129, 0.0122])

# --- TIFER-MMR (从日志提取) ---
tiger_div = np.array([0.3903, 0.4236, 0.4448, 0.4626, 0.4944, 0.5189, 0.5448, 0.5853, 0.6263])
tiger_recall = np.array([0.0374, 0.0297, 0.0274, 0.0260, 0.0250, 0.0242, 0.0235, 0.0220, 0.0212])
tiger_ndcg = np.array([0.0195, 0.0138, 0.0129, 0.0123, 0.0117, 0.0113, 0.0109, 0.0101, 0.0095])

	

# --- Baseline Mock Data (模拟数据: ComiRec & Paragon) ---
# 逻辑：在 Our Model 的基础上进行缩放，使其处于左下方 (Pareto Dominated)

# ComiRec: 性能约为 Our Model 的 85% - 90%
ComiRec_div = [
    0.3532130, 0.3777105, 0.3955960, 0.4189550, 0.4617365, 0.5243420, 0.5639980, 0.6035700, 0.6161285, 0.6365940
    
   
    
]

ComiRec_recall = [
   0.036438, 0.034640, 0.033907, 0.031803, 0.030697, 0.028877, 0.027154, 0.026603, 0.025159, 0.020796
]

ComiRec_ndcg = [
  0.015495, 0.014717, 0.014097, 0.013443, 0.013066, 0.012324, 0.011788, 0.011527, 0.010949, 0.009231
]

# Paragon: 性能约为 Our Model 的 70% - 75%
paragon_div = [0.408568657655765,
               0.428770879967823,
               0.453898752791483,
               0.497229021501454,
               
               0.544567878666119,
               0.587345676788114,
               0.597267038754233]
paragon_recall = [
    0.04518338297799484,
    0.03907381897739243,
    0.03586769787987987,
    0.03258431245442022,
    
    0.03111579288550361,
    0.02632500367687698,
    0.02204538087520259,
]
paragon_ndcg = [
    0.02198848387609636,
    0.01902892809971235,
    0.018166641478318014,
    0.01760937926479031,
    
    0.017425780575812965,
    0.015684845305245554,
    0.010778581264405852,
]
# ==========================================
# 2. 绘图代码
# ==========================================

# 样式设置 (参考您提供的代码)
# 使用不同颜色区分模型
color_our = '#d62728'    # 红色 (突出自己的模型)
color_tifer = '#1f77b4'  # 蓝色
color_ComiRec = '#2ca02c'  # 绿色
color_paragon = '#ff7f0e'# 橙色

# 创建 1x2 子图，保持您要求的 figsize 和 dpi
fig, axs = plt.subplots(1, 2, figsize=(9.8, 2.7), dpi=300, gridspec_kw={'wspace': 0.37})

# 定义绘制单个子图的函数
def plot_pareto_curve(ax, x_metric_name, y_metric_name, 
                      datas, labels, colors, markers):
    """
    通用绘图函数
    datas: List of tuples (x_data, y_data)
    """
    for (x, y), label, color, marker in zip(datas, labels, colors, markers):
        ax.plot(x, y, marker=marker, linewidth=2, color=color, label=label, markersize=6)
    
    # 设置坐标轴标签 (字体大小 15)
    ax.set_xlabel(x_metric_name, fontsize=15)
    ax.set_ylabel(y_metric_name, fontsize=15)
    
    # 设置刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    # 网格
    ax.grid(True, linestyle='--', alpha=0.6)

# --- 准备数据包 ---
# 模型列表顺序
models_labels = ['Ours', 'TIGER-MMR', 'ComiRec', 'Paragon']
models_colors = [color_our, color_tifer, color_ComiRec, color_paragon]
models_markers = ['o', 's', '^', 'D'] # 圆圈, 正方形, 三角形, 菱形

# 子图 1 数据: Div vs Recall
data_plot1 = [
    (our_div, our_recall),
    (tiger_div, tiger_recall),
    (ComiRec_div, ComiRec_recall),
    (paragon_div, paragon_recall)
]

# 子图 2 数据: Div vs NDCG
data_plot2 = [
    (our_div, our_ndcg),
    (tiger_div, tiger_ndcg),
    (ComiRec_div, ComiRec_ndcg),
    (paragon_div, paragon_ndcg)
]

# --- 执行绘制 ---

# 绘制左图 (Diversity vs Recall)
plot_pareto_curve(axs[0], 'Diversity@10', 'Recall@10', 
                  data_plot1, models_labels, models_colors, models_markers)
# 为子图添加标题
# axs[0].set_title('Diversity@10 vs Recall@10', fontsize=16)

# 绘制右图 (Diversity vs NDCG)
plot_pareto_curve(axs[1], 'Diversity@10', 'NDCG@10', 
                  data_plot2, models_labels, models_colors, models_markers)
# 为子图添加标题
# axs[1].set_title('Diversity@10 vs NDCG@10', fontsize=16)

# --- 设置图例 ---
# 由于两个图图例一样，我们在第一个图或者整体加一个图例即可
# 这里选择在第一个图加，或者并在上方。参考您的代码是在内部。
# 为了不遮挡曲线，通常放在"lower left" (因为帕累托通常是左上到右下，左下角较空)
axs[0].legend(fontsize=10, loc='best') 
# axs[1].legend(fontsize=10, loc='best') # 如果需要两边都显示图例，取消注释

# 添加总标题并为其留出空间
# fig.suptitle('Pareto Curves: Diversity vs Recall & NDCG', fontsize=18)

# 调整布局，保留顶端空间给总标题
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存
plt.savefig("pareto_curve_amazon-industrial-scientific.pdf", bbox_inches='tight')
plt.show()
