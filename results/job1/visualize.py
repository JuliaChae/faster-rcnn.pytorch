import matplotlib.pyplot as plt 
import pdb 
import numpy as np

def loss(): 
   f = open("loss.txt", "r")
   lines = f.readlines()
   train = np.zeros([2,119])
   val = np.zeros([2,23])

   i=0
   for line in lines[1:]:
      train[0][i] = i +1
      train[1][i] = float(line.split("       ")[1])
      if float(line.split("       ")[2][:-2]) != 0.:
         val[0][((i+1)//5)-1] = i +1
         val[1][((i+1)//5)-1] = float(line.split("       ")[2])
      i = i +1 

   print(train)
   print(val)

   plt.figure()
   plt.title("Loss curves")
   plt.xlabel("Epochs")
   plt.ylabel("Loss")
   plt.plot(train[0], train[1], '-o', label = "Training", markersize=2)
   plt.plot(val[0], val[1], '-o', label = "Validation", markersize=2)
   plt.legend()
   plt.savefig("loss_curve.png")
   plt.show()

def mAP():
   f = open("mAP.txt", "r")
   lines = f.readlines()
   train = np.zeros([2,119])
   val = np.zeros([2,23])

   i=0
   for line in lines[1:]:
      train[0][i] = i +1
      train[1][i] = float(line.split("       ")[1])
      if float(line.split("       ")[2][:-2]) != 0.:
         val[0][((i+1)//5)-1] = i +1
         val[1][((i+1)//5)-1] = float(line.split("       ")[2])
      i = i +1 

   print(train)
   print(val)

   plt.figure()
   plt.title("mAP curves")
   plt.xlabel("Epochs")
   plt.ylabel("mAP")
   plt.ylim((0.0, 1.0))
   plt.plot(train[0][:-1], train[1][:-1], '-o', label = "Training", markersize=2)
   plt.plot(val[0][:-1], val[1][:-1], '-o', label = "Validation", markersize=2)
   plt.legend()
   plt.savefig("mAP_curve.png")
   plt.show()

def train_AP():
   classes = ['pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck']
   f = open("train_AP.txt", "r")
   lines = f.readlines()
   train = np.zeros([11,119])

   i=0
   for line in lines[1:]:
      train[0][i] = i + 1
      for x in range(1,11):
         train[x][i] = float(line.split("    ")[x])
      i = i +1 

   print(train)

   plt.figure()
   plt.title("Train AP curves")
   plt.xlabel("Epochs")
   plt.ylabel("AP")
   plt.ylim((0.0, 1.0))
   for x in range(1, 11):
      plt.plot(train[0][:-1], train[x][:-1], '-o', label = classes[x-1], markersize=2)
   plt.legend(loc='best')
   plt.savefig("train_AP_curve.png")
   plt.show()

def val_AP():
   classes = ['pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck']
   f = open("val_AP.txt", "r")
   lines = f.readlines()
   val = np.zeros([11,23])

   i=0
   for line in lines[1:]:
      val[0][i] = i + 1
      for x in range(1,11):
         val[x][i] = float(line.split("    ")[x])
      i = i +1 

   print(val)

   plt.figure()
   plt.title("Validation AP curves")
   plt.xlabel("Epochs")
   plt.ylabel("AP")
   plt.ylim((0.0, 1.0))
   for x in range(1, 11):
      plt.plot(val[0][:-1], val[x][:-1], '-o', label = classes[x-1], markersize=2)
   plt.legend(loc='best')
   plt.savefig("validation_AP_curve.png")
   plt.show()

val_AP()


