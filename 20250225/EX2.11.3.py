import math

radius = 5
volume = (4/3) * math.pi * (radius ** 3)
print(f"半徑為 {radius} 的球體體積為: {volume:.2f} 立方厘米")

x = 42  # 角度
x_rad = math.radians(x)  # 轉換為弧度
cos_x_sq = math.cos(x_rad) ** 2
sin_x_sq = math.sin(x_rad) ** 2
sum_of_squares = cos_x_sq + sin_x_sq
print(f"cos²({x}) + sin²({x}) = {sum_of_squares:.6f}")

exp1 = math.e ** 2  # 方式 1
exp2 = math.pow(math.e, 2)  # 方式 2
exp3 = math.exp(2)  # 方式 3
print(f"e^2 計算結果:")
print(f"1. math.e ** 2 = {exp1:.6f}")
print(f"2. math.pow(math.e, 2) = {exp2:.6f}")
print(f"3. math.exp(2) = {exp3:.6f}")