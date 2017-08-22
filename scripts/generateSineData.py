# Author. Oluwatosin V. Adewale
# Simple program to generate input and output of sine and write to file

import math

def main():
    xs = [float(i)/100 for i in range(0,400)]
    sin_x = [math.sin(x) for x in xs]

    f_out = open("trainSinX.csv", 'w')
    f_in = open("trainSinY.csv",'w')

    for i in range(len(xs)):
        f_out.write(str(xs[i]) + "\n")
        f_in.write(str(sin_x[i]) + "\n")

    f_out.close()
    f_in.close()



if __name__ == "__main__":
    main()
