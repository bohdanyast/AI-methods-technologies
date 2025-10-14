import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

# Setting functions values
x_values = np.linspace(0, 20, 100)
y_values = 7 * np.sin(2.5 * np.cos(x_values))
z_values = (x_values - 2) ** 2 * (1 - y_values**2)

# Data normalizing for lowering the value of MSE/MAE due to function value scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_z = MinMaxScaler()

x_values_norm = scaler_x.fit_transform(x_values.reshape(-1, 1)).flatten()
y_values_norm = scaler_y.fit_transform(y_values.reshape(-1, 1)).flatten()
z_values_norm = scaler_z.fit_transform(z_values.reshape(-1, 1)).flatten()

x_means = np.linspace(min(x_values_norm), max(x_values_norm), 6)
y_means = np.linspace(min(y_values_norm), max(y_values_norm), 6)
z_means = np.linspace(min(z_values_norm), max(z_values_norm), 9)

# Getting sigmas 6 inputs for x and y, and 9 outputs for z. Building plots.
x_sigma = (max(x_values_norm) - min(x_values_norm)) / 6
y_sigma = (max(y_values_norm) - min(y_values_norm)) / 6
z_sigma = (max(z_values_norm) - min(z_values_norm)) / 9

mx = [fuzz.gaussmf(x_values_norm, x_means[i], x_sigma) for i in range(6)]
my = [fuzz.gaussmf(np.linspace(min(y_values_norm), max(y_values_norm), 100), y_means[i], y_sigma) for i in range(6)]
mf = [fuzz.gaussmf(np.linspace(min(z_values_norm), max(z_values_norm), 100), z_means[i], z_sigma) for i in range(9)]

for i in range(6):
    plt.plot(x_values_norm, mx[i])
plt.title("X Gaussmf (Normalized)")
plt.show()

for i in range(6):
    plt.plot(np.linspace(min(y_values_norm), max(y_values_norm), 100), my[i])
plt.title("Y Gaussmf (Normalized)")
plt.show()

for i in range(9):
    plt.plot(np.linspace(min(z_values_norm), max(z_values_norm), 100), mf[i])
plt.title("Z Gaussmf (Normalized)")
plt.show()


# Tables of values and mfs
table = [["y\\x"] + [round(i, 2) for i in x_means]]
for y_mean in y_means:
    row = [round(y_mean, 2)]
    for x_mean in x_means:
        z_mean = (x_mean - 2) ** 2 * (1 - y_mean**2)
        row.append(round(z_mean, 2))
    table.append(row)

print(tabulate(table, tablefmt="grid"))

def get_biggest_ordinate(argument, argument_means, sigma):
    best_mf_index = -1
    best_value = -float("inf")

    for index, value in enumerate(argument_means):
        ff = fuzz.gaussmf([argument], value, sigma)
        if ff > best_value:
            best_mf_index = index
            best_value = ff

    return best_mf_index

table_new = [["y\\x"] + ["mx" + str(i) for i in range(1, 7)]]
rules = {}
for i in range(6):
    row = ["my" + str(i+1)]
    for j in range(6):
        z = (x_means[i] - 2) ** 2 * (1 - y_means[j]**2)
        best_mf = get_biggest_ordinate(z, z_means, z_sigma)
        row.append("mf" + str(best_mf + 1))
        rules[(j, i)] = best_mf
    table_new.append(row)

print(tabulate(table_new, tablefmt="grid"))


# Rules
print("\nRules: ")
for rule in rules.keys():
    print(f"If (x is mx{rule[0] + 1}) and (y is mf{rule[0] + 1}) then (f is mf{rules[rule]+1})")

# Modeling function
model = []
for x in x_values_norm:
    best_x_func = get_biggest_ordinate(x, x_means, x_sigma)
    best_y_func = get_biggest_ordinate(7 * np.sin(2.5 * np.cos(x)), y_means, y_sigma)
    best_z_func = rules[(best_x_func, best_y_func)]
    model.append(z_means[best_z_func])

plt.plot(x_values_norm, model, label="Model")
plt.plot(x_values_norm, z_values_norm, label="True")
plt.legend()
plt.show()

print(f"Mean Squared Error: {mean_squared_error(z_values_norm, model)}")
print(f"Mean Absolute Error: {mean_absolute_error(z_values_norm, model)}")
