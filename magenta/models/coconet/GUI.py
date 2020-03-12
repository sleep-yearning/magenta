from tkinter import *
from tkinter import ttk
import tkinter as tk
from tkinter.ttk import *
from tkinter import filedialog, messagebox
from tkinter import Tk, RIGHT, BOTH, RAISED
from tkinter.ttk import Frame, Button, Style
import _thread
from prepare_and_train import prepare, train, main
import os
import pathlib

root = Tk()

# Variables go here

convert_folder_path = StringVar()
result_folder_path_npz = StringVar()
train_folder_path = StringVar()
result_folder_path_training = StringVar()
choosemodel = StringVar()
title_new_train_model = StringVar()
nepochs = StringVar()

is_grouped = BooleanVar()
is_grouped.set(True)

# This map contains the pre-trained models and the paths to their data as a basis for the model selection
model_map = {
    "Abba": "/some-path", # TODO put real paths in here
    "Bach": "/some-path",
    "Zelda": "/some-path"
}

# Definitions go here


def check_that_folder_contains_file(folder_path, file):
    path = os.path.join(folder_path, file)
    return os.path.isfile(path)


def open_converting_folder():
    folder_path = (filedialog.askdirectory(parent=root, title='Choose a folder containing midi files') + '/')
    convert_folder_path.set(folder_path)
    if not result_folder_path_npz.get():
        result_folder_path_npz.set(folder_path)
    print(convert_folder_path)


def open_result_folder_npz():
    folder_path = filedialog.askdirectory(parent=root, title='Choose a folder to place the results in')
    result_folder_path_npz.set(folder_path)
    print(folder_path)


def open_train_folder():
    folder_path = filedialog.askdirectory(parent=root, title='Choose a folder containing npz file')

    if check_that_folder_contains_file(folder_path, 'trainData.npz'):
        train_folder_path.set(folder_path)
        if not result_folder_path_training.get():
            result_folder_path_training.set(folder_path)
        print(folder_path)
    else:
        messagebox.showerror("Error", "There is no 'trainData.npz' in the selected folder!")

def open_result_folder_train():
    folder_path = filedialog.askdirectory(parent=root, title='Choose a folder to place the results in')
    result_folder_path_training.set(folder_path)
    print(folder_path)

def open_sample_midi():
    print("sample midi")

def start_converting():
    print(is_grouped.get())
    prepare(convert_folder_path.get(), is_grouped.get())
    print("Converting done")

def convert_in_background():
    _thread.start_new_thread(start_converting, ())

def stop_it():
    root.destroy()

def start_training():
    string_nepochs = nepochs.get()
    try:
        int_nepochs = int(string_nepochs)
    except ValueError:
        messagebox.showerror("Error", "Nepochs must be a number!")
        return
    train(train_folder_path.get(), int_nepochs, " ", is_grouped.get(), title_new_train_model.get())

def really_start_training():
    add_model_to_menu()
    start_training()

def set_group_true():
    is_grouped.set(True)


def set_group_false():
    is_grouped.set(False)

def callback(*args):
    labelTest.configure(text="The selected item is {}".format(choosemodel.get()))

def add_model_to_menu():
    val = title_new_train_model.get()
    which_model['menu'].add_command(label=val, command=tk._setit(choosemodel, val))
    model_map[val] = train_folder_path.get()

def play():
    model_name = choosemodel.get()
    model_folder_path = model_map[model_name]
    model_checkpoint_folder_path = os.path.join(model_folder_path, model_name + '_checkpoint')
    print(model_checkpoint_folder_path) # TODO use this

root.title("Music in Machine Learning")
root.minsize(640, 500)

# root.wm_iconbitmap('icon.ico')

# Create top Menu
nb = ttk.Notebook(root)
nb.pack()
f1 = ttk.Frame(nb, width=500, height=500)
nb.add(f1, text="Preprocessing")
f2 = ttk.Frame(nb, width=500, height=500)
nb.add(f2, text="Training")
f3 = ttk.Frame(nb, width=500, height=500)
nb.add(f3, text="Sampling")

#########
#
# Start of Converting Page
#
##########

lbl_frame_select_convert_folder = ttk.LabelFrame(f1, text="Select your folder with midi files")
lbl_frame_select_convert_folder.pack()

btn_select_convert_folder = ttk.Button(lbl_frame_select_convert_folder, text="Select Folder",
                                       command=open_converting_folder)
btn_select_convert_folder.pack()

lbl_convert_folder = Label(master=f1, textvariable=convert_folder_path)
lbl_convert_folder.pack()

lbl_grouped = Label(f1,
                    text="Do you want to preprocess the midis with grouped instruments or with the one which are used most frequently?")
lbl_grouped.pack()

yes = Button(f1, text="Yes", command=set_group_true)
yes.pack(pady=0)
no = Button(f1, text="No", command=set_group_false)
no.pack(pady=0)

# lbl_frame_select_result_folder = ttk.LabelFrame(f1, text="Select a folder to save the results")
# lbl_frame_select_result_folder.pack()
#
# btn_select_result_folder = ttk.Button(lbl_frame_select_result_folder, text="Select Folder",
#                                       command=open_result_folder_npz)
# btn_select_result_folder.pack()
#
# lbl_result_folder = Label(master=f1, textvariable=result_folder_path_npz)
# lbl_result_folder.pack()

lbl_convert_midis = Label(f1, text="First preprocess your midi Files to a npz File")
lbl_convert_midis.pack()

btn_start_converting = ttk.Button(f1, text="Start preprocessing", command=convert_in_background)
btn_start_converting.pack()

btn_stop_converting = ttk.Button(f1, text="Stop preprocessing", command=stop_it)
btn_stop_converting.pack()

progress = ttk.Progressbar(f1, orient=HORIZONTAL, length=200)
progress.pack()

#########
#
# Start of Training Page
#
##########

lbl_frame_select_train_folder = ttk.LabelFrame(f2, text="Select your folder with npz file")
lbl_frame_select_train_folder.pack()

btn_select_train_folder = ttk.Button(lbl_frame_select_train_folder, text="Select Folder", command=open_train_folder)
btn_select_train_folder.pack()

lbl_train_folder = Label(master=f2, textvariable=train_folder_path)
lbl_train_folder.pack()

lbl_grouped = Label(f1,
                    text="Do you want to preprocess the midis with grouped instruments or with the one which are used most frequently?")
lbl_grouped.pack()

yes = Button(f1, text="Yes", command=set_group_true)
yes.pack(pady=0)
no = Button(f1, text="No", command=set_group_false)
no.pack(pady=0)
# lbl_frame_select_result_folder = ttk.LabelFrame(f2, text="Select a folder to save the results")
# lbl_frame_select_result_folder.pack()
#
# btn_select_result_folder = ttk.Button(lbl_frame_select_result_folder, text="Select Folder",
#                                       command=open_result_folder_train)
# btn_select_result_folder.pack()
#
# lbl_result_folder = Label(master=f2, textvariable=result_folder_path_training)
# lbl_result_folder.pack()

lbl_new_model_nepoch = Label(f2,
                           text="Please enter number of epochs")
lbl_new_model_nepoch.pack()


title_new_train_model_nepoch = Entry(f2, textvariable=nepochs)
title_new_train_model_nepoch.pack()


lbl_new_model_name = Label(f2,
                           text="Please enter the name for your model")
lbl_new_model_name.pack()


title_new_train_model_widget = Entry(f2, textvariable=title_new_train_model)
title_new_train_model_widget.pack()

lbl_start_training = Label(f2,
                           text="If you already converted your midis, please choose the folder with the .npz file above and start training here.")
lbl_start_training.pack()

btn_start_training = ttk.Button(f2, text="Start Training", command=really_start_training)
btn_start_training.pack()

#########
#
# Start of Sampling Page
#
##########
lbl_choose_model = Label(f3, text="Choose your desired model here")
lbl_choose_model.pack()

OptionList = [
"Choose a model",
] + model_map.keys()

choosemodel.set(OptionList[0])

which_model = OptionMenu(f3, choosemodel, *OptionList)
which_model.config()
which_model.pack()

choosemodel.trace('w', callback)

labelTest = Label(text="")
labelTest.pack()

lbl_frame_select_sample_midi = ttk.LabelFrame(f3, text="Select a midi file to sample from")
lbl_frame_select_sample_midi.pack()

btn_select_sample_midi = ttk.Button(lbl_frame_select_sample_midi, text="Select Folder", command=open_sample_midi)
btn_select_sample_midi.pack()

base_path = str(pathlib.Path(__file__).parent.absolute())

midi_play_button = Button(f3, text="play", command=play)
img_play = PhotoImage(file=os.path.join(base_path, "play.png"))
midi_play_button.config(image=img_play)
midi_play_button.pack(padx=5, pady=10, side=LEFT)

midi_download_button = Button(f3, text="download")
img_download = PhotoImage(file=os.path.join(base_path, "download.png"))
midi_download_button.config(image=img_download)
midi_download_button.pack(padx=5, pady=10, side=LEFT)

# btn_test_training = ttk.Button(root, text="test", command=teeeeest)
# btn_test_training.grid(row=row_counter)

root.mainloop()
