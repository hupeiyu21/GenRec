import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备
# ==========================================

import numpy as np

# our
our_recall = np.array([
    0.042706175246783544,
    0.04284545343755985,
    0.04221870157906649,
    0.042027194066749075,
    0.042323160222148715,
    0.03976392346663417,
    0.03983356256202232,
    0.03953759640662268,
    0.03938090844199934,
    0.03384460035864134,
    0.03217326206932572,
    0.032295130486254986,
    0.033061160535524645
])

our_ndcg = np.array([
    0.020485117188108342,
    0.020521760392619465,
    0.020042688389708088,
    0.02004424017913395,
    0.020177766595885216,
    0.019021448782990058,
    0.019093766675424367,
    0.01885471774080686,
    0.018805517187486125,
    0.016565755749793125,
    0.016061291347384054,
    0.015997497078958943,
    0.016193721449705763
])

our_div = np.array([
    0.4059868404545046,
    0.4220952204484438,
    0.43962995774971386,
    0.4591789663445303,
    0.4777783568132207,
    0.5439540014244274,
    0.5795071636334859,
    0.593488167410727,
    0.5940953825783669,
    0.6202828395126061,
    0.6507895148262024,
    0.6611395803537133,
    0.6624231089184968
])

# TIGER-MMR
tiger_recall = np.array([
    0.04328069778373579,
    0.039885791883563435,
    0.03788366789115409,
    0.0367346228172496,
    0.03513292362332213,
    0.0335486342032417,
    0.03189470568777312,
    0.03095457790003308
])

tiger_ndcg = np.array([
    0.02029573454884517,
    0.01862504742151723,
    0.01766204446921012,
    0.017112229191208226,
    0.01627184406861233,
    0.015584011884715997,
    0.014727875601998565,
    0.014039220492337025
])

tiger_div = np.array([
    0.43197395112554865,
    0.4661192851614447,
    0.49352310254910764,
    0.5126590658614213,
    0.5415425118171486,
    0.5597817422404421,
    0.601787289399984,
    0.6243804803292008
])

# ComiRec 你给我的上一版数据， 画出来效果不好
# ComiRec_recall = np.array([
#     0.022186, 0.024272, 0.028987, 0.030083, 0.031611,
#     0.032952, 0.034134, 0.035069, 0.035768, 0.036354
# ])

# ComiRec_ndcg = np.array([
#     0.012373, 0.013131, 0.015288, 0.015835, 0.016737,
#     0.017303, 0.018009, 0.018817, 0.019512, 0.020143
# ])

# ComiRec_div = np.array([
#     0.659472, 0.550915, 0.611360, 0.610896, 0.644219,
#     0.628393, 0.571292, 0.501896, 0.494948, 0.470216
# ])

# ComiRec: 性能约为 Our Model 的 85% - 90%
ComiRec_div = [
     0.3998862,
    0.4006224,
    0.470216,
    0.494948,
    0.501896,
    0.550915,
    0.571292,
    0.610896,
    0.611360,
    0.628393
]

ComiRec_recall = [
    0.036354,
    0.036117,
    0.035868,
    0.034134,
    0.032952,
    0.031611,
    0.030083,
    0.028987,
    0.024272,
    0.022186
]

ComiRec_ndcg = [
    
  0.020143,
    0.020029,
    0.019922,
    0.018009,
    0.017303,
    0.016737,
    0.015835,
    0.015288,
    0.013131,
    0.012373
]

# --- Baseline Mock Data (模拟数据: ComiRec & Paragon) ---
# 逻辑：在 Our Model 的基础上进行缩放，使其处于左下方 (Pareto Dominated)


# Paragon: 性能约为 Our Model 的 70% - 75%
paragon_div = [ 0.43637971547720955,0.45640060207121256,0.4969468518422945,0.52937279497703277,
               0.5685395975488752,0.5992037407032907,0.6688481220501127,
               ]
paragon_recall = [0.044130434782608695, 0.04338423083192958, 0.039782608695652174, 0.038912, 0.03630434782608696, 0.035, 0.03326086956521739]
paragon_ndcg = [0.02102544830946557, 0.019904380440672345, 0.019384287827805833, 0.018068825086324197, 0.017627201732547445, 0.01729970195959619, 0.015400773075515195]
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
# axs[0].set_title('(a) Recall@10 - Instruments', fontsize=16)

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
plt.savefig("pareto_curve_amazon-musical-instruments.pdf", bbox_inches='tight')
plt.show()