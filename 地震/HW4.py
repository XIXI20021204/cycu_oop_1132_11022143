import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import csv

# 設定中文字型與負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class HystereticModel:
    def __init__(self, k1, k2, xy):
        self.k1 = k1
        self.k2 = k2
        self.xy = xy
        self.Fy = k1 * xy
        self.Fy_neg = -k1 * xy
        self.current_branch = 'elastic'
        self.x_reversal_point = 0.0
        self.Fs_reversal_point = 0.0

    def get_Fs_and_Kt(self, x_trial, x_prev_actual, Fs_prev_actual):
        dx = x_trial - x_prev_actual
        Fs_trial = 0.0
        Kt_trial = 0.0

        if self.current_branch == 'elastic':
            if x_trial >= self.xy:
                Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                Kt_trial = self.k2
            elif x_trial <= -self.xy:
                Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                Kt_trial = self.k2
            else:
                Fs_trial = self.k1 * x_trial
                Kt_trial = self.k1
        elif self.current_branch == 'yield_pos':
            if dx >= 0:
                Fs_trial = self.Fs_reversal_point + self.k2 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k2
            else:
                Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k1
                if Fs_trial < self.Fy_neg:
                    Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                    Kt_trial = self.k2
        elif self.current_branch == 'yield_neg':
            if dx <= 0:
                Fs_trial = self.Fs_reversal_point + self.k2 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k2
            else:
                Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
                Kt_trial = self.k1
                if Fs_trial > self.Fy:
                    Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                    Kt_trial = self.k2
        elif self.current_branch == 'unload_pos':
            Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
            Kt_trial = self.k1
            if Fs_trial > self.Fy:
                Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                Kt_trial = self.k2
            elif Fs_trial < self.Fy_neg:
                Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                Kt_trial = self.k2
        elif self.current_branch == 'unload_neg':
            Fs_trial = self.Fs_reversal_point + self.k1 * (x_trial - self.x_reversal_point)
            Kt_trial = self.k1
            if Fs_trial < self.Fy_neg:
                Fs_trial = self.Fy_neg + self.k2 * (x_trial + self.xy)
                Kt_trial = self.k2
            elif Fs_trial > self.Fy:
                Fs_trial = self.Fy + self.k2 * (x_trial - self.xy)
                Kt_trial = self.k2

        return Fs_trial, Kt_trial

    def update_state(self, x_current_actual, Fs_current_actual, x_prev_actual, Fs_prev_actual):
        dx = x_current_actual - x_prev_actual

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
            if dx < 0:
                self.current_branch = 'unload_pos'
                self.x_reversal_point = x_prev_actual
                self.Fs_reversal_point = Fs_prev_actual
        elif self.current_branch == 'yield_neg':
            if dx > 0:
                self.current_branch = 'unload_neg'
                self.x_reversal_point = x_prev_actual
                self.Fs_reversal_point = Fs_prev_actual
        elif self.current_branch == 'unload_pos':
            if dx >= 0:
                if Fs_current_actual >= self.Fy:
                    self.current_branch = 'yield_pos'
                    self.x_reversal_point = x_current_actual
                    self.Fs_reversal_point = Fs_current_actual
            else:
                if abs(x_current_actual) < self.xy and abs(Fs_current_actual) < self.Fy:
                    self.current_branch = 'elastic'
                    self.x_reversal_point = x_current_actual
                    self.Fs_reversal_point = Fs_current_actual
        elif self.current_branch == 'unload_neg':
            if dx <= 0:
                if Fs_current_actual <= self.Fy_neg:
                    self.current_branch = 'yield_neg'
                    self.x_reversal_point = x_current_actual
                    self.Fs_reversal_point = Fs_current_actual
            else:
                if abs(x_current_actual) < self.xy and abs(Fs_current_actual) < self.Fy:
                    self.current_branch = 'elastic'
                    self.x_reversal_point = x_current_actual
                    self.Fs_reversal_point = Fs_current_actual

        if self.current_branch == 'yield_pos' and x_current_actual > self.x_reversal_point:
            self.x_reversal_point = x_current_actual
            self.Fs_reversal_point = Fs_current_actual
        elif self.current_branch == 'yield_neg' and x_current_actual < self.x_reversal_point:
            self.x_reversal_point = x_current_actual
            self.Fs_reversal_point = Fs_current_actual

# 系統參數與初始條件
m = 1.0
xi = 0.05
k1 = 631.65
k2 = 126.33
xy = 1.0
dt = 0.005
total_time = 2.0
num_steps = int(total_time / dt)
wn = np.sqrt(k1 / m)
c = 2 * m * wn * xi

x0 = 0.0
x_dot0 = 40.0
x_ddot0 = -c * x_dot0 / m
F_external_input = 0.0

# 初始化陣列
t = np.linspace(0, total_time, num_steps + 1)
x = np.zeros(num_steps + 1)
x_dot = np.zeros(num_steps + 1)
x_ddot = np.zeros(num_steps + 1)
Fs = np.zeros(num_steps + 1)
Kt = np.zeros(num_steps + 1)

x[0] = x0
x_dot[0] = x_dot0
x_ddot[0] = x_ddot0
hysteretic_model = HystereticModel(k1, k2, xy)
Fs[0], Kt[0] = hysteretic_model.get_Fs_and_Kt(x[0], x[0], Fs[0])

tolerance = 1e-6
max_iterations = 100
points_of_interest = {}

# 主迴圈
for i in range(num_steps):
    x_k = x[i]
    for iter_count in range(max_iterations):
        Fs_k, Kt_k = hysteretic_model.get_Fs_and_Kt(x_k, x[i], Fs[i])
        R_k = m * ((4 / dt**2) * (x_k - x[i]) - (4 / dt) * x_dot[i] - x_ddot[i]) + \
              c * ((2 / dt) * (x_k - x[i]) - x_dot[i]) + \
              Fs_k - F_external_input
        R_prime_k = (4 * m) / (dt**2) + (2 * c) / dt + Kt_k
        delta_x_k = - R_k / R_prime_k
        x_k_new = x_k + delta_x_k
        if abs(delta_x_k) < tolerance:
            x[i+1] = x_k_new
            break
        x_k = x_k_new
    else:
        x[i+1] = x_k

    hysteretic_model.update_state(x[i+1], Fs[i], x[i], Fs[i])
    Fs[i+1], Kt[i+1] = hysteretic_model.get_Fs_and_Kt(x[i+1], x[i], Fs[i])
    x_ddot[i+1] = (4 / dt**2) * (x[i+1] - x[i]) - (4 / dt) * x_dot[i] - x_ddot[i]
    x_dot[i+1] = x_dot[i] + (dt / 2) * (x_ddot[i] + x_ddot[i+1])

    # 標記特定點
    if 'a' not in points_of_interest and x[i+1] >= xy and Fs[i+1] >= hysteretic_model.Fy * 0.99:
        points_of_interest['a'] = t[i+1]
    if 'a' in points_of_interest and 'b' not in points_of_interest:
        if x_dot[i] > 0 and x_dot[i+1] <= 0:
            points_of_interest['b'] = t[i+1]
    if 'b' in points_of_interest and 'c' not in points_of_interest:
        if x[i] >= 0 and x[i+1] < 0 and hysteretic_model.current_branch == 'unload_pos':
            points_of_interest['c'] = t[i+1]
    if 'c' in points_of_interest and 'd' not in points_of_interest:
        if x[i+1] <= -xy and Fs[i+1] <= hysteretic_model.Fy_neg * 0.99:
            points_of_interest['d'] = t[i+1]
    if 'd' in points_of_interest and 'e' not in points_of_interest:
        if x_dot[i] < 0 and x_dot[i+1] >= 0:
            points_of_interest['e'] = t[i+1]

# 儲存模擬結果為 CSV（主時間步）
with open("simulation_results.csv", mode="w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["時間 (s)", "位移 (in)", "速度 (in/s)", "加速度 (in/s²)", "彈簧力 (k)"])
    for i in range(num_steps + 1):
        writer.writerow([f"{t[i]:.4f}", f"{x[i]:.4f}", f"{x_dot[i]:.4f}", f"{x_ddot[i]:.4f}", f"{Fs[i]:.4f}"])

# 儲存特定點資訊為 CSV
with open("points_of_interest.csv", mode="w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["點位", "時間 (s)", "位移 (in)", "速度 (in/s)", "加速度 (in/s²)", "彈簧力 (k)"])
    for label, time_val in sorted(points_of_interest.items(), key=lambda item: item[1]):
        idx = int(time_val / dt)
        writer.writerow([label, f"{time_val:.4f}", f"{x[idx]:.4f}", f"{x_dot[idx]:.4f}", f"{x_ddot[idx]:.4f}", f"{Fs[idx]:.4f}"])

# 畫圖（略）可根據需要保留或刪除

print("✅ 模擬結果與特定點資訊已成功儲存為 CSV。")
