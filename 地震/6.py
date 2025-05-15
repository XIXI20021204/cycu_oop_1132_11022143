import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    prop = fm.FontProperties(family='DFKai-SB') # 將 'DFKai-SB' 替換為你實際找到的名稱
except:
    print("警告：找不到標楷體，可能無法正常顯示中文。")
    prop = None

# ... (其餘程式碼不變) ...

# 繪製結果
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

if prop:
    axs[0].plot(t, F_eff, label=r'$F_{eff}(t)$', color='b')
    axs[0].set_ylabel("有效力 (k)", fontproperties=prop)
    axs[0].legend(prop=prop)
    axs[1].plot(t, x, label=r'$x(t)$', color='g')
    axs[1].set_ylabel("位移 (in)", fontproperties=prop)
    axs[1].legend(prop=prop)
    axs[2].plot(t, ẋ, label=r'$\dot{x}(t)$', color='r')
    axs[2].set_ylabel("速度 (in/s)", fontproperties=prop)
    axs[2].legend(prop=prop)
    axs[3].plot(t, ẍ, label=r'$\ddot{x}(t)$', color='m')
    axs[3].set_ylabel("加速度 (in/s²)", fontproperties=prop)
    axs[3].set_xlabel("時間 (s)", fontproperties=prop)
    axs[3].legend(prop=prop)
else:
    # 如果找不到字體，仍然繪製圖表，但可能沒有中文標籤
    axs[0].plot(t, F_eff, label=r'$F_{eff}(t)$', color='b')
    axs[0].set_ylabel("有效力 (k)")
    axs[0].legend()
    axs[1].plot(t, x, label=r'$x(t)$', color='g')
    axs[1].set_ylabel("位移 (in)")
    axs[1].legend()
    axs[2].plot(t, ẋ, label=r'$\dot{x}(t)$', color='r')
    axs[2].set_ylabel("速度 (in/s)")
    axs[2].legend()
    axs[3].plot(t, ẍ, label=r'$\ddot{x}(t)$', color='m')
    axs[3].set_ylabel("加速度 (in/s²)")
    axs[3].set_xlabel("時間 (s)")
    axs[3].legend()

axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()

plt.tight_layout()
plt.show()