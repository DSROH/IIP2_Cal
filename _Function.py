import os

import tkinter as tk
from tkinter import filedialog

import pandas as pd
from pprint import pprint as pp


def Common_save_Excel(filename, tab1, tab2):
    # Save Data to Excel
    Tabname = filename.replace("Excel_", "")
    Tabname = f"{os.path.splitext(Tabname)[0]}"
    with pd.ExcelWriter(filename) as writer:
        tab1.to_excel(writer, sheet_name=f"{Tabname}_Mean")
        tab2.to_excel(writer, sheet_name=f"{Tabname}_Data")


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("\tError: Failed to create directory.")


def Common_daseul_log(list_file):
    list_file.delete(0, tk.END)
    files = filedialog.askopenfilenames(
        title="log 파일을 선택하세요",
        filetypes=(("TXT 파일", "*.txt"), ("모든 파일", "*.*")),
        initialdir=r"D:\DATA\바탕화면\\",
    )
    # 사용자가 선택한 파일 목록
    for file in files:
        list_file.insert(tk.END, file)
