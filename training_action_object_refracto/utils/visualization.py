import matplotlib.pyplot as plt

def plot_activation(data, pattern_no):
    plt.figure(figsize=(10, 5))
    plt.plot(data['sum'])
    plt.title(f"Activation Pattern {pattern_no}")
    plt.xlabel("Time")
    plt.ylabel("Activation")
    plt.show()
