import matplotlib.pyplot as plt
import matplotlib as mpl

def openfile(filepath):
    file = open(filepath)
    y = []
    while 1:
        line = file.readline()
        if line.rstrip('\n') == '':
            break
        y.append(float(line.rstrip('\n')))
        if not line:
            break
        pass
    file.close()
    return y

if __name__ == '__main__':
    # mpl.use('TkAgg')
    plt.figure()
    k_array = ['0.5', '0.6', '0.7', '0.8']
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Global Round')
    for k in k_array:
        y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_NISS_epsilon_5_k_{}_drop_0.dat'.format(k))
        plt.plot(range(100), y, label=r'$\k={}$'.format(k))
    # y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_no_dp_epsilon_20.dat'.format(k))
    # plt.plot(range(100), y, label=r'$\k=+\infty$')
    plt.title('NISS')
    plt.legend()
    plt.savefig('NISS.png')

    # mpl.use('TkAgg')
    plt.figure()
    k_array = ['0.5', '0.6', '0.7', '0.8']
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Global Round')
    for k in k_array:
        y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_DpSecureAggregation_epsilon_1_k_{}_drop_0.dat'.format(k))
        plt.plot(range(100), y, label=r'$\k={}$'.format(k))
    # y = openfile('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_no_dp_epsilon_20.dat'.format(k))
    # plt.plot(range(100), y, label=r'$\k=+\infty$')
    plt.title('DpSecureAggregation')
    plt.legend()
    plt.savefig('DpSecureAggregation.png')
