import numpy as np

# 已知參數 (根據您的圖片)
W_ksi = 290  # ksi (結構重量)
g_in_s2 = 386.4 # in/s^2 (重力加速度)
L_columns_ft = 14 # ft (柱高)
L_braces_ft = np.sqrt(14**2 + 25**2) # ft (斜撐長度)
I_columns_in4 = 209 # in^4 (柱子慣性矩)
A_braces_in2 = 0.785 # in^2 (斜撐斷面積)
E_columns_ksi = 29000 # ksi (柱子彈性模數)
E_braces_ksi = 29000 # ksi (斜撐彈性模數)
n_NS = 24 # N-S 方向柱數
n_EW = 6 # E-W 方向斜撐數

# 轉換單位
L_columns_in = L_columns_ft * 12 # in
L_braces_in = L_braces_ft * 12 # in
m_slug = W_ksi / g_in_s2 # slug (質量塊)

# N-S 方向計算
k_NS_ksiin = n_NS * (3 * E_columns_ksi * I_columns_in4) / (L_columns_in**3)
T_NS_s = 2 * np.pi * np.sqrt(m_slug / k_NS_ksiin)

# E-W 方向計算
theta_deg = np.arctan(14 / 25) * 180 / np.pi # 度
theta_rad = np.arctan(14 / 25) # 弧度
k_EW_ksiin = n_EW * (A_braces_in2 * E_braces_ksi * (np.cos(theta_rad))**2) / L_braces_in
T_EW_s = 2 * np.pi * np.sqrt(m_slug / k_EW_ksiin)

# 輸出結果
print(f"N-S 方向的勁度 (k_NS): {k_NS_ksiin:.2f} ksi/in")
print(f"N-S 方向的週期 (T_NS): {T_NS_s:.2f} s")
print(f"E-W 方向的勁度 (k_EW): {k_EW_ksiin:.2f} ksi/in")
print(f"E-W 方向的週期 (T_EW): {T_EW_s:.2f} s")