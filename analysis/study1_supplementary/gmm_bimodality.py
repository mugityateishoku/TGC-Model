import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats

filename = 'tgc_master_dataset.csv'

try:
    df = pd.read_csv(filename)
    print(f"'{filename}' の読み込みに成功しました！")
except FileNotFoundError:
    print(f"エラー: {filename} が見つかりません。")
    exit()

# ==========================================
# ★ここが最重要：時系列を壊さずに負荷(E)を計算する
# Condition (1 or 2) の移動平均を「Cognitive Load (E)」とする
# ==========================================
WINDOW_SIZE = 10
df['E'] = df['Condition'].rolling(window=WINDOW_SIZE, min_periods=1).mean()

# E と RT が揃っている試行だけを抽出（時間軸はEが保持している）
valid_data = df.dropna(subset=['RT', 'E']).copy()
valid_rt = valid_data['RT'].values
valid_E = valid_data['E'].values

print(f"解析対象データ数: {len(valid_rt)} 試行")

# ==========================================
# 1. GMM & BIC 分析
# ==========================================
print("\n--- 1. GMM & BIC 分析 ---")
rt_reshaped = valid_rt.reshape(-1, 1)

gmm1 = GaussianMixture(n_components=1, random_state=42).fit(rt_reshaped)
bic1 = gmm1.bic(rt_reshaped)

gmm2 = GaussianMixture(n_components=2, random_state=42).fit(rt_reshaped)
bic2 = gmm2.bic(rt_reshaped)

print(f"BIC (1-Component) : {bic1:.2f}")
print(f"BIC (2-Components): {bic2:.2f}")
if bic2 < bic1:
    print("結論: データは圧倒的に『2つの状態（二峰性）』を支持しています！")
else:
    print("結論: 全体としては単一の分布で説明可能です。")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ==========================================
# 2. 経験的ポテンシャル地形 V(x)
# ==========================================
hist, bin_edges = np.histogram(valid_rt, bins=25, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

p_x = np.where(hist > 0, hist, 1e-9)
v_x = -np.log(p_x)

ax1.plot(bin_centers, v_x, color='purple', linewidth=3, marker='o')
ax1.set_title('Empirical Potential Landscape $V(x)$')
ax1.set_xlabel('Reaction Time (State $x$)')
ax1.set_ylabel('Potential Energy $V(x) \propto -\ln P(x)$')
ax1.grid(True, linestyle='--', alpha=0.5)

# ==========================================
# 3. 厳密なヒステリシス検定
# ==========================================
print("\n--- 3. 厳密なヒステリシス検定 ---")

def calculate_hysteresis_area(E_seq, beta_seq, n_bins=8):
    delta_E = np.diff(E_seq)
    delta_E = np.insert(delta_E, 0, 0)
    
    asc_mask = delta_E > 0
    desc_mask = delta_E <= 0
    
    bins = np.linspace(E_seq.min(), E_seq.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    y_asc, _, _ = stats.binned_statistic(E_seq[asc_mask], beta_seq[asc_mask], statistic='mean', bins=bins)
    y_desc, _, _ = stats.binned_statistic(E_seq[desc_mask], beta_seq[desc_mask], statistic='mean', bins=bins)
    
    valid_asc = ~np.isnan(y_asc)
    valid_desc = ~np.isnan(y_desc)
    
    # 共通区間のみを抽出
    common = valid_asc & valid_desc
    
    area = 0.0
    if np.sum(common) >= 2:
        try:
            area = np.trapezoid(y_asc[common] - y_desc[common], bin_centers[common])
        except AttributeError:
            area = np.trapz(y_asc[common] - y_desc[common], bin_centers[common])
            
    return area, bin_centers, y_asc, y_desc, valid_asc, valid_desc, common

actual_area, centers, y_asc, y_desc, v_asc, v_desc, common_mask = calculate_hysteresis_area(valid_E, valid_rt)
print(f"実際のヒステリシス面積 (共通区間): {actual_area:.2f}")

# Circular Shift 検定
np.random.seed(42)
n_permutations = 1000
null_areas = []
n_trials = len(valid_rt)

for _ in range(n_permutations):
    shift = np.random.randint(1, n_trials)
    shuffled_rt = np.roll(valid_rt, shift) 
    shuffled_area, _, _, _, _, _, _ = calculate_hysteresis_area(valid_E, shuffled_rt)
    null_areas.append(shuffled_area)

null_areas = np.array(null_areas)
p_value = np.mean(np.abs(null_areas) >= np.abs(actual_area))
print(f"厳密な Circular Shift 検定 p値 : {p_value:.4f}")

# プロット
ax2.plot(centers[v_asc], y_asc[v_asc], color='red', marker='^', linewidth=2.5, markersize=8, label='Ascending ($\Delta E > 0$)')
ax2.plot(centers[v_desc], y_desc[v_desc], color='blue', marker='v', linewidth=2.5, markersize=8, label='Descending ($\Delta E \leq 0$)')

if np.sum(common_mask) >= 2:
    ax2.fill_between(centers[common_mask], y_asc[common_mask], y_desc[common_mask], color='gray', alpha=0.3, label=f'Area = {actual_area:.1f}\n$p = {p_value:.3f}$')

ax2.set_title('Hysteresis Loop: Reaction Time over Cognitive Load $E$')
ax2.set_xlabel('Cognitive Load $E$ (10-Trial Moving Average of Condition)')
ax2.set_ylabel('Reaction Time $\\beta$ (ms)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()