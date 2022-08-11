import matplotlib.pyplot as plt


def visualize(input_file="output_file.txt"):
    print("visualizating...")
    f = open(file=input_file, mode="r")
    x_cor = []
    y_cor = []
    for line in f:
        splitted = line.split(' ')
        x = float(splitted[2])
        y = float(splitted[3])
        print("%f\t%f" % (x, y))
        x_cor.append(x)
        y_cor.append(y)
    f.close()

    plt.scatter(x=x_cor, y=y_cor)
    plt.show()


def main():
    print("main()")
    visualize()


if __name__ == "__main__":
    main()
