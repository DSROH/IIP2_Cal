# %%
import os, re, glob, threading

import tkinter as tk
import tkinter.messagebox as msgbox

import ttkbootstrap as ttkbst
from ttkbootstrap.tableview import Tableview
from ttkbootstrap.constants import *

import pandas as pd
import _Function as func

# %%
Win_GUI = ttkbst.Window(title="LSI Root IIP2 Cal Log Check V1.0", themename="cosmo")
Win_GUI.attributes("-topmost", True)
Win_GUI.geometry("1230x555")

Left_frame = ttkbst.Frame(Win_GUI)
Left_frame.place(x=0, y=0, width=400, height=555)

list_file = tk.Listbox(Left_frame, height=5)
list_file.place(x=5, y=5, width=390, height=55)

btn_add_file = ttkbst.Button(
    Left_frame,
    text="Load CP log (F1)",
    command=lambda: [func.Common_daseul_log(list_file)],
)
btn_add_file.place(x=5, y=65, width=135, height=35)

# %%
Table_frame = ttkbst.Frame(Win_GUI)
Table_frame.place(x=400, y=0, width=830, height=555)

colors = Win_GUI.style.colors

coldata = [
    {"text": "IIP2 (est)", "anchor": "center", "width": 80, "stretch": True},
    {"text": "I_dac_Start", "anchor": "center", "width": 90, "stretch": True},
    {"text": "Q_dac_Start", "anchor": "center", "width": 90, "stretch": True},
    {"text": "I_dac_End", "anchor": "center", "width": 90, "stretch": True},
    {"text": "Q_dac_End", "anchor": "center", "width": 90, "stretch": True},
    {"text": "Prms_Start", "anchor": "center", "width": 95, "stretch": True},
    {"text": "Prms_End", "anchor": "center", "width": 95, "stretch": True},
    {"text": "Prms (Est)", "anchor": "center", "width": 95, "stretch": True},
    {"text": "DAC (Est)", "anchor": "center", "width": 95, "stretch": True},
]

rowdata = []

Table = Tableview(
    master=Table_frame,
    coldata=coldata,
    rowdata=rowdata,
    paginated=False,
    searchable=False,
    autoalign=False,
    bootstyle=PRIMARY,
    height=30,
    stripecolor=(colors.light, None),
)

Table.place(x=0, y=5, width=825, height=95)
# Table.autoalign_columns()
# Table.autofit_columns()

# %%
def Start(list_file, Table):
    if list_file.size() == 0:
        msgbox.showwarning("경고", "Cal log 파일(*.txt)을 추가하세요")
        return
    dict_iip2 = {}

    file_list = list_file.get(0, tk.END)

    IIP2_DAC_MAX = 127
    IIP2_DAC_OFFSET = 7

    for file in file_list:
        Print_fname = os.path.basename(file)
        Print_Word = "Collecting Data"
        my_cols = [str(i) for i in range(20)]  # create some col names

        df_Data = pd.read_csv(file, sep="\t", names=my_cols, header=None, engine="python", encoding="utf-8")
        col_name = df_Data.iloc[0]
        df_Data = df_Data[1:]
        df_Data.columns = col_name
        df_Data.reset_index(drop=True, inplace=True)
        df_Data = df_Data[["CP Time", "Group", "Channel", "Message"]]
        df_Data = df_Data.map(lambda x: x.encode("unicode_escape").decode("utf-8") if isinstance(x, str) else x)
        df_Rfdrv = df_Data[df_Data["Group"].str.contains("RF_DRV").to_list()]
        del [[df_Data]]

        df_Rfdrv = df_Rfdrv[["Message"]]
        df_iip2c = df_Rfdrv[df_Rfdrv["Message"].str.contains(r"\[RF Cal\] IIP2_CAL").to_list()]
        Band_index = df_iip2c[df_iip2c["Message"].str.contains(r"\[RF Cal\] IIP2_CAL >>>>> Start!!").to_list()].values[1]
        Test_Band = Band_index[0].split(" ")[5]
        Test_Mixer = Band_index[0].replace(":", ",").split(",")[-1]

        dict_iip2["CASE0"] = df_iip2c[df_iip2c["Message"].str.contains(r": CASE0").to_list()]
        dict_iip2["CASE1"] = df_iip2c[df_iip2c["Message"].str.contains(r": CASE1").to_list()]
        dict_iip2["CASE2"] = df_iip2c[df_iip2c["Message"].str.contains(r": CASE2").to_list()]
        dict_iip2["CASE3"] = df_iip2c[df_iip2c["Message"].str.contains(r": CASE3").to_list()]

        for key, value in dict_iip2.items():
            df_case = value["Message"].str.split("/| : |, | ", expand=True)
            est_v = df_case[df_case[2].str.contains(r"\(est\)").to_list()]
            for num in range(len(est_v)):
                if "i_dac_start" in est_v.iloc[num, 4]:
                    i_dac_str = int(re.findall(r"\(([^)]+)", est_v.iloc[num, 4])[0])
                    q_dac_str = int(re.findall(r"\(([^)]+)", est_v.iloc[num, 5])[0])
                    p_rms_str = int(re.findall(r"\(([^)]+)", est_v.iloc[num, 6])[0])
                elif "i_dac_end" in est_v.iloc[num, 4]:
                    i_dac_end = int(re.findall(r"\(([^)]+)", est_v.iloc[num, 4])[0])
                    q_dac_end = int(re.findall(r"\(([^)]+)", est_v.iloc[num, 5])[0])
                    p_rms_end = int(re.findall(r"\(([^)]+)", est_v.iloc[num, 6])[0])

                if "dac_est" in est_v.iloc[num, 5]:
                    numerator = p_rms_str + p_rms_end
                    denominator = (IIP2_DAC_MAX - (IIP2_DAC_OFFSET * 2)) * p_rms_str
                    dac_est = int((denominator / numerator) + IIP2_DAC_OFFSET)
                    p_rms_est = int(re.findall(r"\(([^)]+)", est_v.iloc[num, 4])[0])

                    rowdata.append(
                        [
                            est_v.iloc[num, 3],
                            i_dac_str,
                            q_dac_str,
                            i_dac_end,
                            q_dac_end,
                            p_rms_str,
                            p_rms_end,
                            p_rms_est,
                            dac_est,
                        ]
                    )

    Table.destroy()

    Table = Tableview(
        master=Table_frame,
        coldata=coldata,
        rowdata=rowdata,
        paginated=False,
        searchable=False,
        autoalign=False,
        bootstyle=PRIMARY,
        height=30,
        stripecolor=(colors.light, None),
    )

    Table.place(x=0, y=5, width=825, height=95)
    Table.autofit_columns()

# %%
btn_start = ttkbst.Button(Left_frame, text="시작 (F5)", command=lambda: [Start(list_file, Table)])
btn_start.place(x=260, y=65, width=135, height=35)

# %%
Win_GUI.resizable(False, False)
Win_GUI.mainloop()

# %%
import numpy as np
import plotly.graph_objects as go

# 주어진 데이터
temp_i = [66, 66, 66, 67, 67, 67, 68, 68, 68]
temp_q = [66, 67, 68, 66, 67, 68, 66, 67, 68]
power_rms = [978, 911, 1117, 743, 412, 649, 736, 205, 748]

# temp_i와 temp_q의 범위 설정
temp_range = np.arange(min(temp_i) - 10, max(temp_i) + 11)
temp_i_range = np.intersect1d(temp_range, np.arange(1, 121))
temp_q_range = np.intersect1d(temp_range, np.arange(1, 121))

# 격자 생성
temp_i_grid, temp_q_grid = np.meshgrid(temp_i_range, temp_q_range)

# 추세선 계산을 위한 데이터 준비
temp_combined = np.concatenate((temp_i, temp_q))
power_combined = np.array(power_rms * 2)  # temp_i와 temp_q의 값이 같으므로 power_rms 데이터를 2번 반복하여 사용

# 추세선 계산
z = np.polyfit(temp_combined, power_combined, 2)
trend_surface_flat = np.polyval(z, np.concatenate((temp_i_grid.flatten(), temp_q_grid.flatten())))

# trend_surface_flat 배열을 temp_i_grid와 temp_q_grid와 같은 모양으로 재구성
trend_surface = trend_surface_flat.reshape(temp_i_grid.shape[0], -1)

# 3D surface 그래프 생성
fig = go.Figure(data=[go.Surface(z=trend_surface, x=temp_i_grid, y=temp_q_grid)])

# 그래프 레이아웃 설정
fig.update_layout(
    scene=dict(
        xaxis=dict(title="temp_i", range=[min(temp_i_range), max(temp_i_range)]),
        yaxis=dict(title="temp_q", range=[min(temp_q_range), max(temp_q_range)]),
        zaxis=dict(title="power_rms", range=[min(trend_surface.flatten()), max(trend_surface.flatten())]),
    ),
    title="3D Surface 그래프 (추세선)",
)

# 출력
fig.show()

# %%
import numpy as np
import plotly.graph_objects as go

# 주어진 데이터
temp_i = [66, 66, 66, 67, 67, 67, 68, 68, 68]
temp_q = [66, 67, 68, 66, 67, 68, 66, 67, 68]
power_rms = [978, 911, 1117, 743, 412, 649, 736, 205, 748]

# temp_i와 temp_q의 범위 설정
temp_range = np.arange(min(temp_i) - 10, max(temp_i) + 11)
temp_i_range = np.intersect1d(temp_range, np.arange(1, 121))
temp_q_range = np.intersect1d(temp_range, np.arange(1, 121))

# 격자 생성
temp_i_grid, temp_q_grid = np.meshgrid(temp_i_range, temp_q_range)

# power_rms 값 계산
max_power_rms = max(power_rms)
power_rms_surface = np.full_like(temp_i_grid, max_power_rms, dtype=float)  # 모든 값을 최대값으로 초기화
for i, (temp_i_val, temp_q_val) in enumerate(zip(temp_i, temp_q)):
    index = np.where((temp_i_grid == temp_i_val) & (temp_q_grid == temp_q_val))
    if index[0].size > 0:
        power_rms_surface[index] = power_rms[i]

# 최소값 포인트 계산
min_power_rms_index = np.argmin(power_rms)
min_temp_i = temp_i[min_power_rms_index]
min_temp_q = temp_q[min_power_rms_index]
min_power_rms = power_rms[min_power_rms_index]

# 3D surface 그래프 생성
fig = go.Figure(data=[go.Surface(z=power_rms_surface, x=temp_i_grid, y=temp_q_grid)])

# 최소값 포인트에 마커 추가
fig.add_trace(go.Scatter3d(x=[min_temp_i], y=[min_temp_q], z=[min_power_rms], mode="markers", marker=dict(size=5, color="red")))

# 그래프 레이아웃 설정
fig.update_layout(
    scene=dict(
        xaxis=dict(title="temp_i", range=[min(temp_i_range), max(temp_i_range)]),
        yaxis=dict(title="temp_q", range=[min(temp_q_range), max(temp_q_range)]),
        zaxis=dict(title="power_rms", range=[0, max_power_rms]),
    ),
    title="3D Surface 그래프",
)

# 출력
fig.show()

# %%
import numpy as np
import plotly.graph_objects as go

# 주어진 데이터
temp_i = [66, 66, 66, 67, 67, 67, 68, 68, 68]
temp_q = [66, 67, 68, 66, 67, 68, 66, 67, 68]
power_rms = [978, 911, 1117, 743, 412, 649, 736, 205, 748]

# 표면을 정의합니다.
temp_range = np.arange(min(temp_i) - 10, max(temp_i) + 11)
temp_i_range = np.intersect1d(temp_range, np.arange(1, 121))
temp_q_range = np.intersect1d(temp_range, np.arange(1, 121))

temp_i_grid, temp_q_grid = np.meshgrid(temp_i_range, temp_q_range)

max_power_rms = max(power_rms)
power_rms_surface = np.full_like(temp_i_grid, max_power_rms, dtype=float)
for i, (temp_i_val, temp_q_val) in enumerate(zip(temp_i, temp_q)):
    index = np.where((temp_i_grid == temp_i_val) & (temp_q_grid == temp_q_val))
    if index[0].size > 0:
        power_rms_surface[index] = power_rms[i]

# 최소값을 찾습니다.
min_power_rms_index = np.argmin(power_rms)
min_temp_i = temp_i[min_power_rms_index]
min_temp_q = temp_q[min_power_rms_index]
min_power_rms = power_rms[min_power_rms_index]

# 표면 그래프를 생성합니다.
fig = go.Figure(data=[go.Surface(z=power_rms_surface, x=temp_i_grid, y=temp_q_grid)])

# 최소값에 대한 주석을 추가합니다.
annotation_text = f"Temp_i: {min_temp_i}<br>Temp_q: {min_temp_q}<br>Power_rms: {min_power_rms}"
fig.add_trace(
    go.Scatter3d(
        x=[min_temp_i],
        y=[min_temp_q],
        z=[min_power_rms],
        mode="text",
        text=[annotation_text],
        textfont=dict(color="black", size=12),
        textposition="middle right",
    )
)  # 주석을 오른쪽 아래로 배치합니다.

# 레이아웃을 설정합니다.
fig.update_layout(
    scene=dict(
        xaxis=dict(title="온도_i", range=[min(temp_i_range), max(temp_i_range)]),
        yaxis=dict(title="온도_q", range=[min(temp_q_range), max(temp_q_range)]),
        zaxis=dict(title="Power_rms", range=[0, max_power_rms]),
    ),
    title="3D 표면 그래프",
)

# 그래프를 출력합니다.
fig.show()

# %%
import tkinter as tk
from tkinter import ttk
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio

# 주어진 데이터
temp_i = [66, 66, 66, 67, 67, 67, 68, 68, 68]
temp_q = [66, 67, 68, 66, 67, 68, 66, 67, 68]
power_rms = [978, 911, 1117, 743, 412, 649, 736, 205, 748]

# 표면을 정의합니다.
temp_range = np.arange(min(temp_i) - 10, max(temp_i) + 11)
temp_i_range = np.intersect1d(temp_range, np.arange(1, 121))
temp_q_range = np.intersect1d(temp_range, np.arange(1, 121))

temp_i_grid, temp_q_grid = np.meshgrid(temp_i_range, temp_q_range)

max_power_rms = max(power_rms)
power_rms_surface = np.full_like(temp_i_grid, max_power_rms, dtype=float)
for i, (temp_i_val, temp_q_val) in enumerate(zip(temp_i, temp_q)):
    index = np.where((temp_i_grid == temp_i_val) & (temp_q_grid == temp_q_val))
    if index[0].size > 0:
        power_rms_surface[index] = power_rms[i]

# 최소값을 찾습니다.
min_power_rms_index = np.argmin(power_rms)
min_temp_i = temp_i[min_power_rms_index]
min_temp_q = temp_q[min_power_rms_index]
min_power_rms = power_rms[min_power_rms_index]

# Tkinter 윈도우를 생성합니다.
root = tk.Tk()
root.title("3D 표면 그래프")

# Plotly 그래프를 생성합니다.
fig = go.Figure(data=[go.Surface(z=power_rms_surface, x=temp_i_grid, y=temp_q_grid)])

# 최소값에 대한 주석을 추가합니다.
annotation_text = f"Temp_i: {min_temp_i}<br>Temp_q: {min_temp_q}<br>Power_rms: {min_power_rms}"
fig.add_trace(
    go.Scatter3d(
        x=[min_temp_i],
        y=[min_temp_q],
        z=[min_power_rms],
        mode="text",
        text=[annotation_text],
        textfont=dict(color="black", size=12),
        showlegend=False,
    )
)

# Plotly 그래프를 Tkinter 창에 표시합니다.
pio.show(fig, validate=False)

# Tkinter 창을 실행합니다.
root.mainloop()

# %%
import tkinter as tk
from tkinter import ttk
import plotly.graph_objects as go
import numpy as np

# 주어진 데이터
temp_i = [66, 66, 66, 67, 67, 67, 68, 68, 68]
temp_q = [66, 67, 68, 66, 67, 68, 66, 67, 68]
power_rms = [978, 911, 1117, 743, 412, 649, 736, 205, 748]

# 표면을 정의합니다.
temp_range = np.arange(min(temp_i) - 10, max(temp_i) + 11)
temp_i_range = np.intersect1d(temp_range, np.arange(1, 121))
temp_q_range = np.intersect1d(temp_range, np.arange(1, 121))

temp_i_grid, temp_q_grid = np.meshgrid(temp_i_range, temp_q_range)

max_power_rms = max(power_rms)
power_rms_surface = np.full_like(temp_i_grid, max_power_rms, dtype=float)
for i, (temp_i_val, temp_q_val) in enumerate(zip(temp_i, temp_q)):
    index = np.where((temp_i_grid == temp_i_val) & (temp_q_grid == temp_q_val))
    if index[0].size > 0:
        power_rms_surface[index] = power_rms[i]

# 최소값을 찾습니다.
min_power_rms_index = np.argmin(power_rms)
min_temp_i = temp_i[min_power_rms_index]
min_temp_q = temp_q[min_power_rms_index]
min_power_rms = power_rms[min_power_rms_index]

# Tkinter 윈도우를 생성합니다.
root = tk.Tk()
root.title("3D 표면 그래프")

# Plotly 그래프를 생성합니다.
fig = go.FigureWidget(data=[go.Surface(z=power_rms_surface, x=temp_i_grid, y=temp_q_grid)])

# 최소값에 대한 주석을 추가합니다.
annotation_text = f"Temp_i: {min_temp_i}<br>Temp_q: {min_temp_q}<br>Power_rms: {min_power_rms}"
fig.add_trace(
    go.Scatter3d(
        x=[min_temp_i],
        y=[min_temp_q],
        z=[min_power_rms],
        mode="text",
        text=[annotation_text],
        textfont=dict(color="black", size=12),
        showlegend=False,
    )
)

# Tkinter 창에 그래프를 표시합니다.
fig.show()

# Tkinter 창을 실행합니다.
root.mainloop()



