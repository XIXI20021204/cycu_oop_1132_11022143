import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 導入字體管理器

# --- 設定Matplotlib中文顯示 ---
# 選擇一個支持中文字符的字體。您可以根據您的操作系統調整。
# Windows 範例: 'Microsoft YaHei', 'SimHei', 'FangSong'
# macOS 範例: 'PingFang HK', 'Heiti TC'
# Linux 範例: 'DejaVu Sans' (但通常需要安裝中文字體包，如 ttf-wqy-zenhei)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 將字體設定為微軟雅黑
plt.rcParams['axes.unicode_minus'] = False # 解決負號 '-' 顯示為方塊的問題

# --- 雙線性滯後模型類 ---
class HystereticModel:
    def __init__(self, k1, k2, xy):
        self.k1 = k1  # 初始剛度
        self.k2 = k2  # 屈服後剛度
        self.xy = xy  # 屈服位移
        self.Fy = k1 * xy  # 屈服力 (正向)
        self.Fy_neg = -k1 * xy # 屈服力 (負向)

        # 滯後狀態變數 - 這些追蹤系統的實際狀態
        # 'elastic': 彈性區
        # 'yield_pos': 正向屈服中
        # 'yield_neg': 負向屈服中
        # 'unload_pos': 從正向屈服點卸載中
        # 'unload_neg': 從負向屈服點卸載中
        self.current_branch = 'elastic' 
        self.x_reversal_point = 0.0  # 上一個反向點的位移
        self.Fs_reversal_point = 0.0 # 上一個反向點的力

    def get_Fs_and_Kt(self, x_trial, x_prev_actual, Fs_prev_actual):
        """
        根據試探位移 (x_trial) 和上一個時間步的實際狀態 (x_prev_actual, Fs_prev_actual, current_branch)，
        計算試探的彈簧力 (Fs_trial) 和切線剛度 (Kt_trial)。
        此函數在迭代過程中不改變模型的內部狀態。
        """
        
        # 根據前一個實際點和試探點的位移方向
        dx = x_trial - x_prev_actual

        Fs_trial = 0.0
        Kt_trial = 0.0

        # 判斷試探點所在的載入路徑
        if self.current_branch == 'elastic':
            if x_trial >= self.xy: # 進入正向屈服區
                Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                Kt_trial = self.k2
            elif x_trial <= -self.xy: # 進入負向屈服區
                Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                Kt_trial = self.k2
            else: # 仍在彈性區
                Fs_trial = self.k1 * x_trial
                Kt_trial = self.k1
        elif self.current_branch == 'yield_pos':
            if dx >= 0: # 繼續在正向屈服區載入
                Fs_trial = self.Fs_reversal_point + self.k2 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k2
            else: # 從正向屈服點開始卸載
                Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k1
                # 檢查是否穿過負向屈服點
                if Fs_trial < self.Fy_neg: # 重新進入負向屈服
                    Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                    Kt_trial = self.k2
        elif self.current_branch == 'yield_neg':
            if dx <= 0: # 繼續在負向屈服區載入
                Fs_trial = self.Fs_reversal_point + self.k2 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k2
            else: # 從負向屈服點開始卸載
                Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k1
                # 檢查是否穿過正向屈服點
                if Fs_trial > self.Fy: # 重新進入正向屈服
                    Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                    Kt_trial = self.k2
        elif self.current_branch == 'unload_pos': # 從正向反向點卸載中
            Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
            Kt_trial = self.k1
            if Fs_trial > self.Fy: # 重新進入正向屈服
                Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                Kt_trial = self.k2
            elif Fs_trial < self.Fy_neg: # 穿過負向屈服點
                Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                Kt_trial = self.k2
        elif self.current_branch == 'unload_neg': # 從負向反向點卸載中
            Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
            Kt_trial = self.k1
            if Fs_trial < self.Fy_neg: # 重新進入負向屈服
                Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                Kt_trial = self.k2
            elif Fs_trial > self.Fy: # 穿過正向屈服點
                Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                Kt_trial = self.k2
        
        return Fs_trial, Kt_trial

    def update_state(self, x_current_actual, Fs_current_actual, x_prev_actual, Fs_prev_actual):
        """
        在牛頓-拉夫森迭代收斂後，更新模型的內部狀態。
        這將確定下一個時間步的起始狀態。
        """
        dx = x_current_actual - x_prev_actual

        # 根據新的位移和力，判斷分支狀態
        if self.current_branch == 'elastic':
            if x_current_actual >= self.xy:
                self.current_branch = 'yield_pos'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual 
            elif x_current_actual <= -self.xy:
                self.current_branch = 'yield_neg'
                self.x_reversal_point = x_current_actual
                self.Fs_reversal_point = Fs_current_actual 
        elif self.current_branch == 'yield_pos':
            if dx < 0: # 位移方向改變，開始卸載
                self.current_branch = 'unload_pos'
                self.x_reversal_point = x_prev_actual # 反向點是前一個實際點
                self.Fs_reversal_point = Fs_prev_actual
        elif self.current_branch == 'yield_neg':
            if dx > 0: # 位移方向改變，開始卸載
                self.current_branch = 'unload_neg'
                self.x_reversal_point = x_prev_actual # 反向點是前一個實際點
                self.Fs_reversal_point = Fs_prev_actual
        elif self.current_branch == 'unload_pos':
            # 判斷是否重新屈服或回到彈性區
            if dx >= 0: # 重新載入
                if Fs_current_actual >= self.Fy:
                    self.current_branch = 'yield_pos'
                    self.x_reversal_point = x_current_actual
                    self.Fs_reversal_point = Fs_current_actual
            else: # 繼續卸載
                 # 如果位移趨近於零，且力也趨近於零，可能回到彈性區
                if abs(x_current_actual) < self.xy and abs(Fs_current_actual) < self.Fy:
                     self.current_branch = 'elastic'
                     self.x_reversal_point = x_current_actual
                     self.Fs_reversal_point = Fs_current_actual
        elif self.current_branch == 'unload_neg':
            if dx <= 0: # 重新載入
                if Fs_current_actual <= self.Fy_neg:
                    self.current_branch = 'yield_neg'
                    self.x_reversal_point = x_current_actual
                    self.Fs_reversal_point = Fs_current_actual
            else: # 繼續卸載
                if abs(x_current_actual) < self.xy and abs(Fs_current_actual) < self.Fy:
                     self.current_branch = 'elastic'
                     self.x_reversal_point = x_current_actual
                     self.Fs_reversal_point = Fs_current_actual

        # 確保反向點在屈服時是正確的
        if self.current_branch == 'yield_pos' and x_current_actual > self.x_reversal_point:
            self.x_reversal_point = x_current_actual
            self.Fs_reversal_point = Fs_current_actual
        elif self.current_branch == 'yield_neg' and x_current_actual < self.x_reversal_point:
            self.x_reversal_point = x_current_actual
            self.Fs_reversal_point = Fs_current_actual


# --- 參數設定 ---
m = 1.0  # 質量 [k·s²/in]
xi = 0.05  # 阻尼比
k1 = 631.65  # 初始剛度 [k/in]
k2 = 126.33  # 屈服後剛度 [k/in]
xy = 1.0  # 屈服位移 [in]
dt = 0.005  # 時間步長 [s]
total_time = 2.0 # 模擬總時間，需要足夠長以觀察第一個迴圈
num_steps = int(total_time / dt) # 計算總步數

# 導出參數
wn = np.sqrt(k1 / m)  # 系統初始自然頻率 (基於 k1)
c = 2 * m * wn * xi  # 阻尼係數 [k·s/in]

# --- 初始條件 ---
x0 = 0.0  # 初始位移 [in]
x_dot0 = 40.0  # 初始速度 [in/s] (修正為題目給定值)

# 根據運動方程 m*ẍ + c*ẋ + F_s(x) = 0 導出初始加速度 ẍ(0)
# 在 t=0: m*ẍ(0) + c*x_dot(0) + F_s(x(0)) = 0
# 由於 x(0)=0, F_s(x(0))=0 (在彈性區)
# 所以 ẍ(0) = -c * x_dot(0) / m
x_ddot0 = -c * x_dot0 / m # 修正後的初始加速度 [in/s²]

print(f"計算得出的初始阻尼係數 c = {c:.4f} k·s/in")
print(f"計算得出的初始加速度 ẍ(0) = {x_ddot0:.4f} in/s²")


# 地面加速度（此問題假定為自由振動，無持續外部輸入）
F_external_input = 0.0 # 外部輸入力為0，響應僅由初始條件驅動

# --- 初始化陣列 ---
# 這些陣列將儲存每個時間步的結果
t = np.linspace(0, total_time, num_steps + 1) # 時間向量
x = np.zeros(num_steps + 1)
x_dot = np.zeros(num_steps + 1)
x_ddot = np.zeros(num_steps + 1)
Fs = np.zeros(num_steps + 1) # 彈簧力
Kt = np.zeros(num_steps + 1) # 切線剛度

# 設定初始值
x[0] = x0
x_dot[0] = x_dot0
x_ddot[0] = x_ddot0
# 計算初始彈簧力
hysteretic_model = HystereticModel(k1, k2, xy)
# 初始時模型狀態為彈性，所以 Fs[0] 和 Kt[0] 會是 k1 * x[0] 和 k1
Fs[0], Kt[0] = hysteretic_model.get_Fs_and_Kt(x[0], x[0], Fs[0])


# --- 平均加速度法 (非線性求解) ---
# 迭代求解的容忍度
tolerance = 1e-6
max_iterations = 100

# 追蹤特定點
points_of_interest = {} # 儲存點 a, b, c, d, e 的時間步索引

for i in range(num_steps):
    # --- 牛頓-拉夫森迭代 ---
    # 為 x_{i+1} 提供一個初始猜測。使用 x[i] 作為起始猜測。
    x_k = x[i] 

    for iter_count in range(max_iterations):
        # 計算試探狀態下的 Fs 和 Kt
        # get_Fs_and_Kt 函數使用 HystereticModel 中儲存的實際狀態 (從前一個時間步收斂而來)
        # 結合 x_k (當前迭代猜測) 來計算 Fs_k 和 Kt_k
        Fs_k, Kt_k = hysteretic_model.get_Fs_and_Kt(x_k, x[i], Fs[i])

        # 計算殘差 R(x_k)
        # 方程: m*x_ddot_{i+1} + c*x_dot_{i+1} + F_s(x_{i+1}) = F_external(i+1)
        # 將 x_ddot 和 x_dot 表示為 x 的函數
        # x_ddot_k = (4 / dt**2) * (x_k - x[i]) - (4 / dt) * x_dot[i] - x_ddot[i]
        # x_dot_k = (2 / dt) * (x_k - x[i]) - x_dot[i]
        
        R_k = m * ((4 / dt**2) * (x_k - x[i]) - (4 / dt) * x_dot[i] - x_ddot[i]) + \
              c * ((2 / dt) * (x_k - x[i]) - x_dot[i]) + \
              Fs_k - F_external_input
        
        # 計算 R 的導數 (切線剛度) R_prime_k
        # R_prime_k = dR/dx_k = m * (4/dt^2) + c * (2/dt) + d(Fs_k)/dx_k
        R_prime_k = (4 * m) / (dt**2) + (2 * c) / dt + Kt_k

        # 更新位移增量 delta_x_k
        delta_x_k = - R_k / R_prime_k
        x_k_new = x_k + delta_x_k

        # 檢查收斂
        if abs(delta_x_k) < tolerance:
            x[i+1] = x_k_new
            break
        x_k = x_k_new
    else:
        print(f"警告: 在時間步 t={i*dt:.3f}s 處，牛頓-拉夫森未收斂。")
        x[i+1] = x_k # 如果未收斂，使用最後的迭代值

    # --- 歷史追蹤 (更新 HystereticModel 的內部狀態) ---
    # 在牛頓-拉夫森收斂後，更新 HystereticModel 的實際狀態
    # Fs[i] 是前一個時間步的彈簧力
    hysteretic_model.update_state(x[i+1], Fs[i], x[i], Fs[i]) 
    # Fs[i+1] 應該是根據 x[i+1] 和最終狀態計算
    Fs[i+1], Kt[i+1] = hysteretic_model.get_Fs_and_Kt(x[i+1], x[i], Fs[i]) 

    # --- 計算速度和加速度 ---
    x_ddot[i+1] = (4 / dt**2) * (x[i+1] - x[i]) - (4 / dt) * x_dot[i] - x_ddot[i]
    x_dot[i+1] = x_dot[i] + (dt / 2) * (x_ddot[i] + x_ddot[i+1])

    # --- 追蹤特定點 (a, b, c, d, e) ---
    # 這些點的判斷通常需要基於位移、速度和力狀態的變化
    # 這裡的邏輯是基於一個假設的響應模式，可能需要根據實際模擬結果微調
    
    # 點 a: 首次達到正向屈服 (位移 >= xy 且力 >= Fy)
    # 由於初始速度是正的，系統會向正向運動
    if 'a' not in points_of_interest and x[i+1] >= xy and Fs[i+1] >= hysteretic_model.Fy * 0.99:
        points_of_interest['a'] = t[i+1]
        print(f"點 a (首次正向屈服) 在 t = {t[i+1]:.3f}s, x={x[i+1]:.3f}, Fs={Fs[i+1]:.3f}")
    
    # 點 b: 正向最大位移點 (速度從正變負或接近零，且位移為正向最大)
    # 並且已經過了點a
    if 'a' in points_of_interest and 'b' not in points_of_interest:
        if x_dot[i] > 0 and x_dot[i+1] <= 0: # 速度從正數變成非正數 (過零點)
            points_of_interest['b'] = t[i+1]
            print(f"點 b (正向最大位移) 在 t = {t[i+1]:.3f}s, x={x[i+1]:.3f}, Fs={Fs[i+1]:.3f}")

    # 點 c: 卸載並經過零點 (位移從正變負或力從正變負，且從正向卸載)
    # 並且已經過了點b
    if 'b' in points_of_interest and 'c' not in points_of_interest:
        if x[i] >= 0 and x[i+1] < 0 and hysteretic_model.current_branch == 'unload_pos':
             points_of_interest['c'] = t[i+1]
             print(f"點 c (卸載過零點) 在 t = {t[i+1]:.3f}s, x={x[i+1]:.3f}, Fs={Fs[i+1]:.3f}")

    # 點 d: 首次達到負向屈服 (位移 <= -xy 且力 <= -Fy)
    # 並且已經過了點c
    if 'c' in points_of_interest and 'd' not in points_of_interest:
        if x[i+1] <= -xy and Fs[i+1] <= hysteretic_model.Fy_neg * 0.99:
            points_of_interest['d'] = t[i+1]
            print(f"點 d (首次負向屈服) 在 t = {t[i+1]:.3f}s, x={x[i+1]:.3f}, Fs={Fs[i+1]:.3f}")
            
    # 點 e: 負向最大位移點 (速度從負變正或接近零，且位移為負向最大)
    # 並且已經過了點d
    if 'd' in points_of_interest and 'e' not in points_of_interest:
        if x_dot[i] < 0 and x_dot[i+1] >= 0: # 速度從負數變成非負數 (過零點)
            points_of_interest['e'] = t[i+1]
            print(f"點 e (負向最大位移) 在 t = {t[i+1]:.3f}s, x={x[i+1]:.3f}, Fs={Fs[i+1]:.3f}")
            # 為確保只標記第一個迴圈，可以在這裡結束點追蹤，或更精確地定義迴圈完成的點。

# --- 列印前六個時間步的結果 ---
print("\n--- 前六個時間步的結果 ---")
print(f"{'時間 (s)':<10} {'位移 (in)':<15} {'速度 (in/s)':<15} {'加速度 (in/s²)':<15} {'彈簧力 (k)':<15}")
for i in range(min(num_steps + 1, 6)):
    print(f"{t[i]:<10.4f} {x[i]:<15.4f} {x_dot[i]:<15.4f} {x_ddot[i]:<15.4f} {Fs[i]:<15.4f}")

print("\n--- 特定時間點 ---")
# 確保點按照時間順序輸出
sorted_points = sorted(points_of_interest.items(), key=lambda item: item[1])
for point_label, time_val in sorted_points:
    idx = int(time_val / dt)
    # 檢查索引是否有效
    if idx < len(x) and idx < len(Fs):
        print(f"點 {point_label}: 時間 = {time_val:.3f}s, 位移 = {x[idx]:.4f} in, 速度 = {x_dot[idx]:.4f} in/s, 加速度 = {x_ddot[idx]:.4f} in/s², 彈簧力 = {Fs[idx]:.4f} k")
    else:
        print(f"警告: 點 {point_label} ({time_val:.3f}s) 的索引超出範圍，可能模擬時間不足。")


# --- 繪製時間歷史圖 ---
plt.figure(figsize=(12, 10))

# 位移 x(t)
plt.subplot(4, 1, 1)
plt.plot(t, x, label='Displacement x(t)')
plt.ylabel('Displacement x(t) [in]')
plt.title('System Dynamic Response Time History')
plt.grid(True)
plt.legend()

# 速度 ẋ(t)
plt.subplot(4, 1, 2)
plt.plot(t, x_dot, label='Velocity $\\dot{x}(t)$', color='orange')
plt.ylabel('Velocity $\\dot{x}(t)$ [in/s]')
plt.grid(True)
plt.legend()

# 加速度 ẍ(t)
plt.subplot(4, 1, 3)
plt.plot(t, x_ddot, label='Acceleration $\\ddot{x}(t)$', color='green')
plt.ylabel('Acceleration $\\ddot{x}(t)$ [in/s²]')
plt.grid(True)
plt.legend()

# 彈簧力 F_s(t)
plt.subplot(4, 1, 4)
plt.plot(t, Fs, label='Spring Force $F_s(t)$', color='red')
plt.ylabel('Spring Force $F_s(t)$ [k]')
plt.xlabel('Time t [s]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# --- 繪製 F_s(x) 滯後迴圈 ---
plt.figure(figsize=(8, 6))
plt.plot(x, Fs, label='Spring Force $F_s(x)$ Hysteretic Loop', color='purple')
# 標記屈服點
plt.axvline(x=xy, color='gray', linestyle='--', label='Positive Yield Displacement')
plt.axvline(x=-xy, color='gray', linestyle='--', label='Negative Yield Displacement')
plt.axhline(y=hysteretic_model.Fy, color='gray', linestyle='-.', label='Positive Yield Force')
plt.axhline(y=hysteretic_model.Fy_neg, color='gray', linestyle='-.', label='Negative Yield Force')

# 標記特定點 a, b, c, d, e
colors = ['red', 'green', 'blue', 'cyan', 'magenta']
labels_plot = ['a', 'b', 'c', 'd', 'e']
# 根據 sorted_points 繪製點
for j, (point_label, time_val) in enumerate(sorted_points):
    idx = int(time_val / dt)
    if idx < len(x) and idx < len(Fs):
        plt.scatter(x[idx], Fs[idx], color=colors[j], marker='o', s=100, zorder=5, label=f'Point {point_label}')
        # 調整文本位置，避免重疊
        plt.text(x[idx] + 0.05, Fs[idx] + 0.05 * hysteretic_model.Fy, point_label, fontsize=12, ha='left', va='bottom')

plt.xlabel('Displacement x(t) [in]')
plt.ylabel('Spring Force $F_s(t)$ [k]')
plt.title('Spring Force $F_s(x)$ - Hysteretic Loop')
plt.grid(True)
plt.legend()
plt.show()