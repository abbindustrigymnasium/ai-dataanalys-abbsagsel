import csv
import matplotlib.pyplot as plt
import numpy as np
import os

# Kommentarer beroende på hur bra din balans och tid är
unbalance_scale = [
    "Your balance is good.",
    "Your current running form is unbalanced."
]

time_scale = [
    "Running in intervals might enable you to run for a longer period of time.",
    "Your time is pretty good. Keep it up and soon you can run a marathon!"
]

y_value = []
x_value = []
z_value = []
new_y_value = []
timestamps = []
trim_begin = 0
trim_end = 240

while True: #Hämtar den önskade filen
    filename = input("Enter a filename... \n > ")+'.csv'
    try:
        with open(filename) as data:
            data = csv.reader(data, delimiter=' ', quotechar='|')
        break
    except FileNotFoundError:
        input("ERR: File not found. Please try again.")

with open(filename) as data: #Formatterar och sorterar all data i variabler
    data = csv.reader(data, delimiter=' ', quotechar='|')
    for row in data:
        item_list = (row[0].split(','))
        for index, item in enumerate(item_list):
            item_list[index] = float(item)
        timestamps.append(item_list[0])
        x_value.append(item_list[1])
        y_value.append(item_list[2]+1)
        z_value.append(item_list[3])


def transformData(data): #Smoothing genom att använda medelvärde med ett fönster på 6
    newlist = []
    for item in range(len(data)):
        try:
            newlist.append(sum(data[item-2:item+3])/len(data[item-2:item+3]))
        except ZeroDivisionError:
            newlist.append(0)
    return newlist


def spikeData(data): #Beräknar antal spikar i datan (obalans)
    unbalance = 0
    spikes = []
    diffs = []
    for i, item in enumerate(data):
        try:
            if item > data[i-1] and item > data[i+1]:
                spikes.append(item)
        except IndexError:
            pass
    for i, item in enumerate(spikes):
        diffs.append(item - spikes[i-1])
    for i, diff in enumerate(diffs):
        if ((diff - diffs[i-1]) > 0.2) or ((diff - diffs[i-1]) < -0.2):
            unbalance += 1
    return unbalance


def unbalanceCalc(unbalance, time): #Bestämmer balans-kommentaren till resultatet
    the_unbalance = round(unbalance/len(time), 2)
    if the_unbalance >= 0.05:
        return unbalance_scale[1]
    else:
        return unbalance_scale[0]


def timeCalc(time): #Bestämmer tid-kommentaren till resultatet
    your_time = int(time[-1])*0.05
    if your_time <= 120:
        mess = time_scale[0]
    else:
        mess = time_scale[1]
    return your_time, mess

#Lägger upp 6 st fönster för diagrammen, och kallar på ovanstående funktioner
fig, axs = plt.subplots(3, 2)
x_result = transformData(x_value)
y_result = transformData(y_value)
z_result = transformData(z_value)
x_unbalance = spikeData(x_result)
time = timeCalc(timestamps)
balance = unbalanceCalc(x_unbalance, timestamps)

#Skriver ut resultatet i konsollen
os.system('cls')
print("-"*60)
print("Time: "+str(time[0])+" s")
print("-"*20)
print("Feedback:")
print("> "+str(balance))
print("> "+str(time[1]))
print("-"*60)

#Plottar ut all rå data samt formatterad data
axs[0, 0].plot(timestamps[trim_begin:trim_end],
x_value[trim_begin:trim_end], label="X")
axs[0, 0].set_title("Original X")

axs[1, 0].plot(timestamps[trim_begin:trim_end],
y_value[trim_begin:trim_end], label="Y")
axs[1, 0].set_title("Original Y")

axs[2, 0].plot(timestamps[trim_begin:trim_end],
z_value[trim_begin:trim_end], label="Z")
axs[2, 0].set_title("Original Z")

axs[0, 1].plot(timestamps[trim_begin:trim_end],
x_result[trim_begin:trim_end], label="X")
axs[0, 1].set_title("New X")

axs[1, 1].plot(timestamps[trim_begin:trim_end],
y_result[trim_begin:trim_end], label="Y")
axs[1, 1].set_title("New Y")

axs[2, 1].plot(timestamps[trim_begin:trim_end],
z_result[trim_begin:trim_end], label="Z")
axs[2, 1].set_title("New Z")

plt.show()
