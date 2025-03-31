import os
import requests
import msgpack
import umsgpack
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import io
from sklearn.cluster import DBSCAN

BASE_URL = "http://127.0.0.1:8866/api"

def create_bench(user_id, name, test=True, queries=[0], max_batches=10, limit=1):
    """벤치 생성"""
    data = {
        "user_id": user_id,
        "name": name,
        "test": test,
        "queries": queries,
        "apitoken": 'polimi-deib',
    }
    if max_batches is not None:
        data["max_batches"] = max_batches

    response = requests.post(f"{BASE_URL}/create", json=data)

    print(f"Response Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")  # 응답이 JSON인지 확인

    # 응답이 JSON 형식인지 확인 후 처리
    if response.status_code == 200:
        return response.text.strip('"')  # JSON이 아니므로 직접 문자열 처리
    else:
        print("Bench creation failed!")
        return None

def start_bench(bench_id):
    """벤치 실행 시작"""
    response = requests.post(f"{BASE_URL}/start/{bench_id}")
    return response.status_code == 200


def get_next_batch(bench_id):
    """입력 데이터 가져오기 (MessagePack)"""
    response = requests.get(f"{BASE_URL}/next_batch/{bench_id}")

    if response.status_code == 200:
        batch_data = msgpack.unpackb(response.content, raw=False)  # MessagePack → Python 변환
        return batch_data
    elif response.status_code == 404:
        print("모든 배치 데이터를 가져왔습니다.")
        return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def compute_outliers(image3d, empty_threshold, saturation_threshold, distance_threshold, outlier_threshold):
    image3d = image3d.astype(np.float64)
    depth, width, height = image3d.shape

    def get_padded(image, d, x, y, pad=0.0):
        if d < 0 or d >= image.shape[0]:
            return pad
        if x < 0 or x >= image.shape[1]:
            return pad
        if y < 0 or y >= image.shape[2]:
            return pad
        return image[d, x, y]

    outliers = []
    # For each point
    for y in range(height):
        for x in range(width):
            if image3d[-1, x, y] <= empty_threshold or image3d[-1, x, y] >= saturation_threshold:
                continue
            # Close neighbours
            cn_sum = 0
            cn_count = 0
            for j in range(-distance_threshold, distance_threshold + 1):
                for i in range(-distance_threshold, distance_threshold + 1):
                    for d in range(depth):
                        # Manhattan distance
                        distance = abs(i) + abs(j) + abs(depth - 1 - d)
                        if distance <= distance_threshold:
                            cn_sum += get_padded(image3d, d, x+i, y+j)
                            cn_count += 1
            # Outer neighbours
            on_sum = 0
            on_count = 0
            for j in range(-2 * distance_threshold, 2 * distance_threshold + 1):
                for i in range(-2 * distance_threshold, 2 * distance_threshold + 1):
                    for d in range(depth):
                        distance = abs(i) + abs(j) + abs(depth - 1 - d)
                        if distance > distance_threshold and distance <= 2*distance_threshold:
                            on_sum += get_padded(image3d, d, x+i, y+j)
                            on_count += 1
            # Compare the mean
            close_mean = cn_sum / cn_count
            outer_mean = on_sum / on_count

            dev = abs(close_mean - outer_mean)

            # Append outliers
            if image3d[-1, x, y] > empty_threshold and image3d[-1, x, y] < saturation_threshold and dev > outlier_threshold:
                outliers.append((x, y, dev.round(2)))

    return outliers

def cluster_outliers_2d(outliers, eps=20, min_samples=5):
    if len(outliers) == 0:
        return []

    # Extract 2D positions (row, col) for clustering
    positions = np.array([(outlier[0], outlier[1]) for outlier in outliers])

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    labels = clustering.labels_  # Cluster labels (-1 means noise)

    # Group points by cluster and calculate centroids and sizes
    centroids = []
    for label in set(labels):
        if label == -1:
            continue  # Skip noise points
        # Get all points in the current cluster
        cluster_points = positions[labels == label]
        # Calculate the centroid as the mean coordinate of all points in the cluster
        centroid = cluster_points.mean(axis=0)
        centroids.append({
            'x': centroid[0],
            'y': centroid[1],
            'count': len(cluster_points)
        })

    return centroids

def plot_centroids(centroids):
    if not centroids:
        print("No centroids to plot.")
        return

    x_vals = [c['x'] for c in centroids]
    y_vals = [c['y'] for c in centroids]
    sizes = [c['count'] * 10 for c in centroids]  # 클러스터 크기 반영 (가시성 높이기 위해 5배)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, y_vals, s=sizes, c=sizes, cmap='viridis', alpha=0.75, edgecolors="k")
    plt.colorbar(label="Cluster Size")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Centroids Clustering Visualization")
    plt.gca().invert_yaxis()  # 이미지 좌표계 맞추기
    plt.show()

tile_map = dict()
window = []
def process(batch):
    global window
    EMPTY_THRESH = 5000
    SATURATION_THRESH = 65000
    DISTANCE_FACTOR = 2
    OUTLIER_THRESH = 6000
    DBSCAN_EPS = 20
    DBSCAN_MIN = 5

    print_id = batch["print_id"]
    tile_id = batch["tile_id"]
    batch_id = batch["batch_id"]
    layer = batch["layer"]
    image = Image.open(io.BytesIO(batch["tif"]))

    # logger.info(f"Processing layer {layer} of print {print_id}, tile {tile_id}")

    if len(window) == 3:
        window.pop(0)
    window.append(image)

    # 65000을 넘은 것들
    saturated = np.count_nonzero(np.array(image) > SATURATION_THRESH)

    if len(window) == 3:
        image3d = np.stack(window, axis=0) # Stack all layers on top of each other in a 3d matrix
        outliers = compute_outliers(image3d, EMPTY_THRESH, SATURATION_THRESH, DISTANCE_FACTOR, OUTLIER_THRESH)
        centroids = cluster_outliers_2d(outliers, DBSCAN_EPS, DBSCAN_MIN)
        centroids = sorted(centroids, key=lambda x: -x['count'])
    else:
        centroids = []

    result = {
        "batch_id": batch_id,
        "print_id": print_id,
        "tile_id": tile_id,
        "saturated": saturated,
        "centroids": centroids
    }
    return result

# 실행
if __name__ == "__main__":
    bench_id = create_bench(user_id="polimi-deib", name="unoptimized", max_batches=12)
    if bench_id:
        print(f"Bench created: {bench_id}")

        if start_bench(bench_id):
            print("Bench started!")

            # 모든 배치 데이터를 가져오기
            while True:
                batch = get_next_batch(bench_id)
                if batch is None:
                    break
                else:
                    result = process(batch)
                    print("batch_id : ",result['batch_id'])
                    print("tile_id : ",result['tile_id'])
                    print(result)
                    plot_centroids(result["centroids"])
                    print()
                    print()
        else:
            print("Failed to start bench.")
    else:
        print("Bench creation failed.")


