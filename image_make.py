import os
import requests
import msgpack
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import io
from sklearn.cluster import DBSCAN

BASE_URL = ""

def create_bench(user_id, name, test=True, queries=[0], max_batches=10, limit=1):
    """벤치 생성"""
    data = {
        "user_id": user_id,
        "name": name,
        "test": test,
        "queries": queries,
        "apitoken": '',
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

temp = defaultdict(list)
index = 0

def process_tif_data(layer, tif_binary):
    global index
    image = Image.open(io.BytesIO(tif_binary))  # tif 바이너리 데이터를 이미지로 변환
    image_array = np.array(image)  # numpy 배열로 변환
    temp[layer].append(image_array)
    index = max(index, layer)
    return image_array  # numpy 배열로 반환 (픽셀 데이터)

# 실행
if __name__ == "__main__":
    bench_id = create_bench(user_id="", name="", max_batches=30)
    if bench_id:
        print(f"Bench created: {bench_id}")

        if start_bench(bench_id):
            print("Bench started!")

            # 모든 배치 데이터를 가져오기
            while True:
                batch = get_next_batch(bench_id)
                if batch is None:
                    break
                tif_data = batch.get('tif')
                if tif_data:
                    image_array = process_tif_data(int(batch.get('layer')), tif_data)
                    print("tile_id",batch.get('tile_id'))
                    print("layer",batch.get('layer'))
                    print("shape", image_array.shape)
                    print("Image Array:", image_array)  # numpy 배열로 출력


            for j in range(index + 1):
                fig, axes = plt.subplots(5, 5, figsize=(35, 35))
                axes = axes.flatten()
                for i in range(5 * 5):
                    if i < len(temp[j]):
                        axes[i].imshow(temp[j][i])
                        axes[i].axis('off')
                        axes[i].set_title(f"layer={j} and title_id={i}", fontsize=40)
                    else:
                        axes[i].axis("off")
                plt.tight_layout()
                plt.show()
                
        else:
            print("Failed to start bench.")
    else:
        print("Bench creation failed.")


