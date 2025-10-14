import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz

num_students = 800  # Students quantity
num_subjects = 4  # Subjects quantity
num_clusters = 3  # Clusters quantity (can be "excellent student", "good student", "f student" for example)
min_grade, max_grade = 1, 12

# Setting centers for each cluster (can be edited for best choice)
clusters_centers_ratings = [4, 7, 10]
clusters_centers = [[i] * num_subjects for i in clusters_centers_ratings]

# Data for each cluster
clusters_data = []

for i in range(num_clusters):
    cluster_size = (num_students // num_clusters, num_subjects)

    cluster_data = np.random.normal(clusters_centers[i], 1.1, cluster_size)
    cluster_data = np.clip(cluster_data, min_grade, max_grade)
    clusters_data.append(cluster_data)

clusters_data = np.vstack(clusters_data)

columns = ['Алгебра', 'Англійська', 'Геометрія', 'Фізика']
df = pd.DataFrame(clusters_data, columns=columns)

# Printing test data
print("Двієчники:")
print(df.iloc[100:110])

print("\nСередні:")
print(df.iloc[400:410])

print("\nВідмінники:")
print(df.iloc[700:710])

# Plotting generated data (for Algebra and English e.g.)
plt.scatter(clusters_data[:, 0], clusters_data[:, 1])
plt.title("Згенеровані дані")
plt.show()

# FCM for data
center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    clusters_data.T, num_clusters, 3, error=0.005, maxiter=100
)

fuzzy_labels = np.argmax(u, axis=0)

for i in range(num_clusters):
    cluster_points = clusters_data[fuzzy_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1])

plt.scatter(center[:, 0], center[:, 1], marker="*", color="black")
plt.title("Кластери з центрами")
plt.show()

# Objective function plotting
plt.plot(jm)
plt.xlabel("Кількість ітерацій")
plt.ylabel("Значення цільової функції")
plt.grid(True)
plt.show()

# Fuzzy partition coefficient
print(f"Якість кластеризації: {fpc}")
