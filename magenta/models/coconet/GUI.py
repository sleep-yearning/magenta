# author: Tessa Lottermann
import os

import _thread
import pathlib
import pygame
import pygame.midi
import tkinter as tk
from tkinter import *
from tkinter import Tk, RIGHT
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.ttk import *
from tkinter.ttk import Button

from magenta.models.coconet.coconet_sample import main as sample
from magenta.models.coconet.prepare_and_train import main as combo
from magenta.models.coconet.prepare_and_train import prepare, train

root = Tk()

# Variables go here
convert_folder_path = StringVar()
result_folder_path_npz = StringVar()
train_folder_path = StringVar()
result_folder_path_training = StringVar()
result_folder_path_sampling = StringVar()
sampling_folder_path = StringVar()
own_checkpoint_path = StringVar()
sample_midi_path = StringVar()
title_new_train_model = StringVar()
nepochs = StringVar()
sampled_midi = StringVar()

is_grouped_pre = BooleanVar()
is_grouped_pre.set(True)

is_grouped_train = BooleanVar()
is_grouped_train.set(True)

# Hyperparams for Training
use_residual = BooleanVar()
use_sep_conv = BooleanVar()
dilate_time_only = BooleanVar()
repeat_last_dilation_level = BooleanVar()
architecture_int = IntVar()
architecture = StringVar()
size_batch = StringVar()
nfilters = StringVar()
nlayers = StringVar()
ndilationblocks = StringVar()
npointwisesplits = StringVar()
interleavesplit = StringVar()

# Default Values for Training
use_residual.set(True)
use_sep_conv.set(True)
dilate_time_only.set(False)
repeat_last_dilation_level.set(False)
architecture_int.set(1)
architecture_list = [('straight', 1), ('dilated', 2)]
size_batch.set(10)
nfilters.set(128)
nlayers.set(64)
ndilationblocks.set(1)
npointwisesplits.set(2)
interleavesplit.set(2)

# Hyparams for Sampling
choosemodel = StringVar()
choosestrategy = StringVar()
size_batch_strategy = StringVar()
piece_length = StringVar()
temperature = StringVar()
tfsample = BooleanVar()

# default values for sampling
tfsample.set(True)
size_batch_strategy.set(3)
piece_length.set(32)
temperature.set(0.99)

# This map contains the pre-trained models and the paths to their data as a basis for the model selection

base_path = str(pathlib.Path(__file__).parent.absolute())

model_map = {
    "Pokemon": os.path.join(base_path, "trained_models/"),
    "Sonic": os.path.join(base_path, "trained_models/"),
    "RedHotChiliPeppers": os.path.join(base_path, "trained_models/"),
    "Beatles": os.path.join(base_path, "trained_models/"),
    "Zelda": os.path.join(base_path, "trained_models/")
}


# Definitions go here

# To check if train folder contains TrainData.npz
def check_that_folder_contains_file(folder_path, filename):
    path = os.path.join(folder_path, filename)
    return os.path.isfile(path)


# Set folder for preprocessing
def open_converting_folder():
    folder_path = (filedialog.askdirectory(parent=root, title='Choose a folder containing midi files') + '/')
    convert_folder_path.set(folder_path)
    result_folder_path_npz.set(folder_path)
    print(convert_folder_path)


# Set folder for training
def open_train_folder():
    folder_path = filedialog.askdirectory(parent=root, title='Choose a folder containing npz file')
    train_folder_path.set(os.path.join(folder_path, ''))
    if not result_folder_path_training.get():
        result_folder_path_training.set(folder_path)
    print(folder_path)


# Set folder for sampling
def open_result_folder_sampling():
    folder_path = filedialog.askdirectory(parent=root, title='Choose a folder to place the results in')
    sampling_folder_path.set(folder_path)
    print(folder_path)


# Set folder with own pretrained model
def open_own_checkpoint():
    folder_path = filedialog.askdirectory(parent=root, title='Choose a folder with your own Checkpoint')
    own_checkpoint_path.set(folder_path)
    print(folder_path)


# Choose midi to sample from
def open_sample_midi():
    folder_path = filedialog.askopenfilename(parent=root, filetypes=[("Midi Files", ".midi .mid")],
                                             title='Choose the midi file to sample from')
    sample_midi_path.set(folder_path)
    print(folder_path)


# Select a generated midi to play
def select_sampled_midi():
    folder_path = filedialog.askopenfilename(parent=root, filetypes=[("Midi Files", ".midi .mid")],
                                             title='Choose the midi file that you want to play')
    sampled_midi.set(folder_path)
    # print(folder_path)


# start Preprocessing
def start_preprocessing():
    prepare(convert_folder_path.get(), is_grouped_pre.get())
    print("Preprocessing done")


# Use Preprocessing with multithreading to make the GUI still usable
def convert_in_background():
    _thread.start_new_thread(start_preprocessing, ())


# Stop Button
def stop_it():
    root.destroy()


# Add the Pretrained Model to the sample page
def add_model_to_menu():
    val = title_new_train_model.get()
    which_model['menu'].add_command(label=val, command=tk._setit(choosemodel, val))
    model_map[val] = train_folder_path.get()


# Start training by checking the right type of hyperparameters first.
def start_training():
    string_nepochs = nepochs.get()
    try:
        int_nepochs = int(string_nepochs)
    except ValueError:
        messagebox.showerror("Error", "Nepochs must be a number!")
        return

    string_batches = size_batch.get()
    try:
        int_batches = int(string_batches)
    except ValueError:
        messagebox.showerror("Error", "Batch size must be a number!")
        return

    string_filters = nfilters.get()
    try:
        int_filters = int(string_filters)
    except ValueError:
        messagebox.showerror("Error", "Filters must be a number!")
        return

    string_layers = nlayers.get()
    try:
        int_layers = int(string_layers)
    except ValueError:
        messagebox.showerror("Error", "Layers must be a number!")
        return

    string_db = ndilationblocks.get()
    try:
        int_db = int(string_db)
    except ValueError:
        messagebox.showerror("Error", "Dilation Blocks must be a number!")
        return

    string_pointwise = npointwisesplits.get()
    try:
        int_pointwise = int(string_pointwise)
    except ValueError:
        messagebox.showerror("Error", "Pointwise Splits must be a number!")
        return

    string_interleave = interleavesplit.get()
    try:
        int_interleave = int(string_interleave)
    except ValueError:
        messagebox.showerror("Error", "Pointwise Splits must be a number!")
        return

    if not check_that_folder_contains_file(train_folder_path.get(), 'TrainData.npz'):
        messagebox.showerror("Error", "There is no 'TrainData.npz' in the selected folder!")

    if architecture_int == 1:
        architecture.set("straight")
    else:
        architecture.set("dilated")

    # print('architecture='.format(architecture.get()))
    # print("Folder={}".format(train_folder_path.get()))
    # print("epochs={}".format(int_nepochs))
    # print("isgrouped={}".format(is_grouped_train.get()))
    # print("use_residual={}".format(use_residual.get()))
    # print("use_sep_conv={}".format(use_sep_conv.get()))
    # print("dilate_time_only={}".format(dilate_time_only.get()))
    # print("repeat_last_dilation_level={}".format(repeat_last_dilation_level.get()))
    # print("size_batch.get()={}".format(int_batches))
    # print("int_filters={}".format(int_filters))
    # print("int_layers={}".format(int_layers))
    # print("int_db={}".format(int_db))
    # print("int_pointwise={}".format(int_pointwise))
    # print("int_interleave={}".format(int_interleave))

    train(train_folder_path.get(), int_nepochs, is_grouped_train.get(), title_new_train_model.get(),
          architecture=architecture.get(),
          use_residual=use_residual.get(),
          use_sep_conv=use_sep_conv.get(), dilate_time_only=dilate_time_only.get(),
          repeat_last_dilation_level=repeat_last_dilation_level.get(), batch_size=int_batches,
          num_filters=int_filters, num_layers=int_layers, num_dilation_blocks=int_db,
          num_pointwise_splits=int_pointwise, interleave_split_every_n_layers=int_interleave)


# combine adding model to sample page and start training for Button command
def really_start_training():
    add_model_to_menu()
    start_training()
    print("Training done")


# Run Preprocessing and training in one step
def preptrain():
    string_nepochs = nepochs.get()
    try:
        int_nepochs = int(string_nepochs)
    except ValueError:
        messagebox.showerror("Error", "Nepochs must be a number!")
        return

    combo(train_folder_path.get(), int_nepochs, is_grouped_train.get(), title_new_train_model.get())


# start sampling by checking the right input types first,
# afterwards checking if using a pretrained model or an own model
def start_sampling():
    string_batches_strategy = size_batch_strategy.get()
    try:
        int_batches_strategy = int(string_batches_strategy)
    except ValueError:
        messagebox.showerror("Error", "Batch size must be a number!")
        return

    string_piece_length = piece_length.get()
    try:
        int_piece_length = int(string_piece_length)
    except ValueError:
        messagebox.showerror("Error", "Piecelength must be a number!")
        return

    string_temperature = temperature.get()
    try:
        float_temperature = float(string_temperature)
        if float_temperature < 0 or float_temperature > 1:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Temperature size must be a number between 0 and 1!")
        return

    if own_checkpoint_path.get() != '':
        checkpoint = own_checkpoint_path.get()
    else:
        model_name = choosemodel.get()
        model_folder_path = model_map[model_name]
        model_checkpoint_folder_path = os.path.join(model_folder_path, model_name + '_checkpoint')
        checkpoint = model_checkpoint_folder_path

    sample(checkpoint=checkpoint, tfsample=tfsample.get(), strategy=choosestrategy.get(),
           gen_batch_size=int_batches_strategy, piece_length=int_piece_length, temperature=float_temperature,
           generation_output_dir=sampling_folder_path.get(), prime_midi_melody_fpath=sample_midi_path.get())

    messagebox.showinfo("Info", "Sampling done!")


# play a midi file
def play_midi():
    pygame.midi.init()
    pygame.mixer.init()
    pygame.mixer.music.load(sampled_midi.get())
    pygame.mixer.music.play()


root.title("Music in Machine Learning")
root.minsize(640, 500)

# Create top Menu
nb = ttk.Notebook(root)
nb.pack()
f1 = ttk.Frame(nb)
f1.pack(expand=True)
f2 = ttk.Frame(nb)
f2.pack(expand=True)
f3 = ttk.Frame(nb)
f3.pack(expand=True)

scrollbar = Scrollbar(f2, orient=VERTICAL)
canvas_f2 = Canvas(f2, yscrollcommand=scrollbar.set)
scrollbar.pack(side=RIGHT, fill=Y)
scrollbar.config(command=canvas_f2.yview)

nb.add(f1, text="Preprocessing")
nb.add(f2, text="Training")
nb.add(f3, text="Sampling")

#########
#
# Start of Preprocessing Page
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
                    text="Do you want to preprocess the midis with grouped instruments or with the one which are used "
                         "most frequently?")
lbl_grouped.pack()

set_is_grouped_pre = tk.Checkbutton(f1, text='preprocess grouped', var=is_grouped_pre)
set_is_grouped_pre.pack()

lbl_convert_midis = Label(f1, text="First preprocess your midi Files to a npz File")
lbl_convert_midis.pack()

btn_start_converting = ttk.Button(f1, text="Start preprocessing", command=convert_in_background)
btn_start_converting.pack()

btn_stop_converting = ttk.Button(f1, text="Stop preprocessing", command=stop_it)
btn_stop_converting.pack()

Separator(f1, orient=HORIZONTAL).pack(fill='x', pady=10)

lbl_do_preptrain = Label(f1,
                         text="If you want to preprocess and train your data in "
                              "one step you can do it in the >>Training<< Tab")
lbl_do_preptrain.pack()

#########
#
# Start of Training Page
#
##########

lbl_frame_select_train_folder = ttk.LabelFrame(f2,
                                               text="If you already converted your midis, please choose the folder "
                                                    "with the .npz file here:")
lbl_frame_select_train_folder.pack()

btn_select_train_folder = ttk.Button(lbl_frame_select_train_folder, text="Select Folder", command=open_train_folder)
btn_select_train_folder.pack()

lbl_train_folder = Label(master=f2, textvariable=train_folder_path)
lbl_train_folder.pack()

Separator(f2, orient=HORIZONTAL).pack(fill='x')

lbl_grouped = Label(f2,
                    text="Do you want to preprocess the midis with grouped instruments or with the one which are used "
                         "most frequently?")
lbl_grouped.pack()

set_is_grouped_train = tk.Checkbutton(f2, text='train grouped', var=is_grouped_train)
set_is_grouped_train.pack()

Separator(f2, orient=HORIZONTAL).pack(fill='x')

lbl_hyparams_training = Label(f2,
                              text="Here you can specify some parameters for your training. These parameters are "
                                   "optional. If you choose to leave some fields empty, the default value will be "
                                   "selected.")
lbl_hyparams_training.pack()

set_use_residual = tk.Checkbutton(f2, text='Add residual connections or not', var=use_residual)
set_use_residual.pack()

set_use_sep_conv = tk.Checkbutton(f2, text='Use depthwise separable convolutions.', var=use_sep_conv)
set_use_sep_conv.pack()

set_dilate_time_only = tk.Checkbutton(f2, text='If set, only dilates the time'
                                               'dimension and not pitch.', var=dilate_time_only)
set_dilate_time_only.pack()

set_repeat_last_delation_level = tk.Checkbutton(f2, text='If set, repeats the last dilation rate.',
                                                var=repeat_last_dilation_level)
set_repeat_last_delation_level.pack()

lbl_architecture = Label(f2, text="Choose an architecture:")
lbl_architecture.pack()

Radiobutton(f2, text="Straight", variable=architecture_int, value=1).pack()
Radiobutton(f2, text="Dilated", variable=architecture_int, value=2).pack()

lbl_new_model_batch_size = Label(f2, text="Please enter the batch size")
lbl_new_model_batch_size.pack()

size_of_batch = Entry(f2, textvariable=size_batch)
size_of_batch.pack()

lbl_new_model_nfilter = Label(f2, text="Please enter number of filters")
lbl_new_model_nfilter.pack()

filter_number = Entry(f2, textvariable=nfilters)
filter_number.pack()

lbl_new_model_nlayers = Label(f2, text="Please enter number of layers")
lbl_new_model_nlayers.pack()

layer_number = Entry(f2, textvariable=nlayers)
layer_number.pack()

lbl_new_model_ndilationblocks = Label(f2, text="Please enter number of dilation blocks")
lbl_new_model_ndilationblocks.pack()

dilation_blocks_number = Entry(f2, textvariable=ndilationblocks)
dilation_blocks_number.pack()

lbl_new_model_npointwisesplit = Label(f2, text="Please enter number of pointwise splits")
lbl_new_model_npointwisesplit.pack()

npointwisesplit_number = Entry(f2, textvariable=npointwisesplits)
npointwisesplit_number.pack()

lbl_new_model_ninterleavesplit = Label(f2,
                                       text="Num of split pointwise layers to interleave between full pointwise layers")
lbl_new_model_ninterleavesplit.pack()

interleavesplit_number = Entry(f2, textvariable=interleavesplit)
interleavesplit_number.pack()

Separator(f2, orient=HORIZONTAL).pack(fill='x')

lbl_new_model_nepoch = Label(f2, text="Please enter number of epochs")
lbl_new_model_nepoch.pack()

title_new_train_model_nepoch = Entry(f2, textvariable=nepochs)
title_new_train_model_nepoch.pack()

lbl_new_model_name = Label(f2,
                           text="Please enter the name for your model")
lbl_new_model_name.pack()

title_new_train_model_widget = Entry(f2, textvariable=title_new_train_model)
title_new_train_model_widget.pack()

Separator(f2, orient=HORIZONTAL).pack(fill='x')

lbl_start_training = Label(f2,
                           text="Start training here:")
lbl_start_training.pack()

btn_start_training = ttk.Button(f2, text="Start Training", command=really_start_training)
btn_start_training.pack()

lbl_start_preptrain = Label(f2,
                            text="Start preprocessing and training in one step here:")
lbl_start_preptrain.pack()

btn_start_preptrain = ttk.Button(f2, text="Start preprocessing and training", command=preptrain)
btn_start_preptrain.pack()

canvas_f2.config(scrollregion=canvas_f2.bbox(ALL))

#########
#
# Start of Sampling Page
#
##########

lbl_choose_model = Label(f3, text="Choose your desired model here")
lbl_choose_model.pack()

OptionList = [
                 "Choose a model",
             ] + list(model_map.keys())

choosemodel.set(OptionList[0])

which_model = OptionMenu(f3, choosemodel, *OptionList)
which_model.config()
which_model.pack()

lbl_frame_own_checkpoint_folder = ttk.LabelFrame(f3, text="OR select a folder containint your own checkpoint")
lbl_frame_own_checkpoint_folder.pack()

btn_own_checkpoint_folder = ttk.Button(lbl_frame_own_checkpoint_folder, text="Select Folder",
                                       command=open_own_checkpoint)
btn_own_checkpoint_folder.pack()

lbl_own_checkpoint_folder = Label(master=f3, textvariable=own_checkpoint_path)
lbl_own_checkpoint_folder.pack()

lbl_choose_strategy = Label(f3, text="Choose a sampling strategy")
lbl_choose_strategy.pack()

OptionList_strategy = [
    "Choose a strategy",
    'bach_upsampling',
    'scratch_upsampling',
    'revoice',
    'harmonization',
    'transition',
    'chronological',
    'orderless',
    'igibbs',
    'agibbs',
    'complete_manual'
]

choosestrategy.set(OptionList[0])

which_strategy = OptionMenu(f3, choosestrategy, *OptionList_strategy)
which_strategy.config()
which_strategy.pack()

Separator(f3, orient=HORIZONTAL).pack(fill='x', pady=10)

set_tfsample = tk.Checkbutton(f3, text='Run sampling in Tensorflow graph.', var=tfsample)
set_tfsample.pack()

sample_batch_size = Entry(f3, textvariable=size_batch_strategy)
sample_batch_size.pack()

lbl_sample_piece_length = Label(f3, text="Please enter piecelength")
lbl_sample_piece_length.pack()

sample_piece_length = Entry(f3, textvariable=piece_length)
sample_piece_length.pack()

lbl_sample_temperature = Label(f3, text="Please enter temperature")
lbl_sample_temperature.pack()

sample_temperature = Entry(f3, textvariable=temperature)
sample_temperature.pack()

Separator(f3, orient=HORIZONTAL).pack(fill='x', pady=10)

lbl_frame_select_sample_midi = ttk.LabelFrame(f3, text="Select a midi file to sample from")
lbl_frame_select_sample_midi.pack()

btn_select_sample_midi = ttk.Button(lbl_frame_select_sample_midi, text="Select File", command=open_sample_midi)
btn_select_sample_midi.pack()

lbl_sample_midi = Label(master=f3, textvariable=sample_midi_path)
lbl_sample_midi.pack()

lbl_frame_select_result_folder = ttk.LabelFrame(f3, text="Select a folder to save the results")
lbl_frame_select_result_folder.pack()

btn_select_result_folder = ttk.Button(lbl_frame_select_result_folder, text="Select Folder",
                                      command=open_result_folder_sampling)
btn_select_result_folder.pack()

lbl_result_folder = Label(master=f3, textvariable=sampling_folder_path)
lbl_result_folder.pack()

Separator(f3, orient=HORIZONTAL).pack(fill='x', pady=10)

btn_start_sample = ttk.Button(f3, text="Start Sampling", command=start_sampling)
btn_start_sample.pack()

Separator(f3, orient=HORIZONTAL).pack(fill='x', pady=10)

lbl_frame_select_sampled_midi = ttk.LabelFrame(f3, text="Which midi do you want to play?")
lbl_frame_select_sampled_midi.pack()

btn_select_sampled_midi = ttk.Button(lbl_frame_select_sampled_midi, text="Select Midi file",
                                     command=select_sampled_midi)
btn_select_sampled_midi.pack()

lbl_sampled_midi = Label(master=f3, textvariable=sampled_midi)
lbl_sampled_midi.pack()

midi_play_button = Button(f3, text="play", command=play_midi)
img_play = PhotoImage(file=os.path.join(base_path, "play.png"))
midi_play_button.config(image=img_play)
midi_play_button.pack(padx=5, pady=0)

root.mainloop()
