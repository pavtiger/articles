В этом файле собраны все команды из видео по инференсу LLM моделей на CPU.

# Скачивание весов и подготовка моделей
С ollama.com
```shell
ollama run qwen3-coder:30b-a3b-fp16
```

Способ через `wget` для скачивания весов от unsloth на huggung face

`download.sh`
```shell
#!/bin/bash
# Check if correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_name> <quantization> <total_shards>"
    echo "Example: $0 Qwen3-Coder-480B-A35B-Instruct Q6_K 9"
    exit 1
fi

# Assign arguments to variables
MODEL_NAME="$1"
QUANTIZATION="$2"
TOTAL_SHARDS="$3"
REPO="unsloth/${MODEL_NAME}-GGUF"

# Loop through shards and download sequentially
for ((i=1; i<=TOTAL_SHARDS; i++)); do
    # Format the shard number with leading zeros (e.g., 01, 02)
    SHARD_NUM=$(printf "%05d" $i)
    TOTAL_SHARDS_PAD=$(printf "%05d" $TOTAL_SHARDS)

    # Define local filename and URL
    LOCAL_FILE="${MODEL_NAME}-${QUANTIZATION}-${SHARD_NUM}-of-${TOTAL_SHARDS_PAD}.gguf"
    URL="https://huggingface.co/${REPO}/resolve/main/${QUANTIZATION}/${MODEL_NAME}-${QUANTIZATION}-${SHARD_NUM}-of-${TOTAL_SHARDS_PAD}.gguf"

    # Skip download if file already exists
    if [ -f "$LOCAL_FILE" ]; then
        echo "Skipping $LOCAL_FILE (file already exists)"
        continue
    fi

    echo "Downloading $LOCAL_FILE..."
    wget -O "$LOCAL_FILE" "$URL"

    # Check if wget was successful
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $LOCAL_FILE"
    else
        echo "Failed to download $LOCAL_FILE"
        exit 1
    fi
done

echo "All downloads completed."
```

Пример запуска
```shell
./download.sh Qwen3-Coder-480B-A35B-Instruct Q3_K_S 5
```

Склеивание шардов в один файл для ollama
```shell
./bin/llama-gguf-split --merge \
  /path/to/Qwen3-Coder-480B-A35B-Instruct-Q6_K-00001-of-00009.gguf \
  /path/to/Qwen3-Coder-Q6_K.gguf
```

Modelfile
```
FROM Qwen3-Coder-480B-Q4_0.gguf
```
И дальше
```
ollama create Q4_0 -f /mnt/archive/qwen-coder/Q4_0/Modelfile
```

# Запуск ollama

Конфиг ollama (`/etc/systemd/system/ollama.service`) для запуска полностью на CPU 
```bash
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"
Environment="OLLAMA_KEEP_ALIVE=3h"
Environment="OLLAMA_LOAD_TIMEOUT=30m"
Environment="OLLAMA_NUM_PARALLEL=1"

Environment="OLLAMA_NO_GPU=1"
Environment="CUDA_VISIBLE_DEVICES="
Environment="HIP_VISIBLE_DEVICES="

[Install]
WantedBy=default.target
```

Конфиг ollama (`/etc/systemd/system/ollama.service`) на двухпроцессорной системе для запуска полностью на CPU 
```bash
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/bin/numactl --cpunodebind=0 --membind=0 /usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"
Environment="OLLAMA_KEEP_ALIVE=3h"
Environment="OLLAMA_LOAD_TIMEOUT=30m"
Environment="OLLAMA_NUM_PARALLEL=1"

Environment="OLLAMA_NO_GPU=1"
Environment="CUDA_VISIBLE_DEVICES="
Environment="HIP_VISIBLE_DEVICES="

Environment="OLLAMA_THREADS=64"
Environment="OMP_NUM_THREADS=64"
Environment="OMP_PROC_BIND=spread"
Environment="OMP_PLACES=threads"
Environment="KMP_AFFINITY=granularity=fine,compact"

[Install]
WantedBy=default.target
```

Перезапустить сервис ollama
```shell
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Рассчет скорости запросов в ollama
```shell
#!/bin/bash
PROMPT="Write a merge sort algorithm."
MODEL=${1:-Q5_K_M:latest}
MAX_TOKENS=300

TOKEN_COUNT=0
START=$(date +%s.%N)

# Stream and count tokens
curl -s -N -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"stream\": true,
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$PROMPT\"}
    ]
  }" | while read -r line; do
    echo $line
    echo $TOKEN_COUNT
    content=$(echo "$line" | jq -r '.message.content // empty' 2>/dev/null)
    if [[ -n "$content" ]]; then
      ((TOKEN_COUNT++))
    fi
    [[ "$TOKEN_COUNT" -ge "$MAX_TOKENS" ]] && break
  done

END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc -l)
TPS=$(echo "scale=2; $MAX_TOKENS / $ELAPSED" | bc -l)

echo
echo "-----------------------------------------"
echo "Elapsed time: $ELAPSED seconds"
echo "Estimated tokens: $MAX_TOKENS"
echo "Average tokens/sec: $TPS"
```

# Запуск llama.cpp

Билд llama.cpp под CPU
```shell
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_HIPBLAS=OFF
cmake --build build -j"$(nproc)"
```

Установка драйверов
```shell
sudo apt purge 'nvidia-*' 'libnvidia-*'
sudo apt autoremove --purge

sudo ubuntu-drivers autoinstall
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
```

Компиляция llama.cpp с поддержкой CUDA
```shell
sudo apt update
sudo apt install build-essential cmake libcurl4-openssl-dev nvidia-cuda-toolkit

# clean and configure
rm -rf build
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON

# compile
cmake --build build -j"$(nproc)"
```

Тестовый запрос в llama.cpp
```shell
curl -X POST http://192.168.88.74:11435/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3:8b", "messages": [{"role": "user", "content": "Write a merge sort algorithm."}]}'
```

Запуск llama.cpp сервера на CPU
```shell
./build/bin/llama-server \
  --model /home/serverflow/Q4_K_S/Qwen3-Coder-480B-A35B-Instruct-Q4_K_S-00001-of-00006.gguf \
  --host 0.0.0.0 \
  --port 11435 \
  --threads 64 \
  --ctx-size 16384 \
  --n-gpu-layers 0 \
  --no-mmap \
  -ot ".ffn_.*_exps.=CPU" \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --repeat-penalty 1.05
```

Запуск llama.cpp сервера на CPU на двухпроцессорной системе
```shell
export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact 
export OMP_PLACES=threads  # instead of cores
export OMP_PROC_BIND=spread

numactl --cpunodebind=0 --membind=0 ./build/bin/llama-server \
  --model /home/serverflow/Q4_K_S/Qwen3-Coder-480B-A35B-Instruct-Q4_K_S-00001-of-00006.gguf \
  --host 0.0.0.0 \
  --port 11435 \
  --threads 64 \
  --ctx-size 16384 \
  --n-gpu-layers 0 \
  --no-mmap \
  -ot ".ffn_.*_exps.=CPU" \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --repeat-penalty 1.05
```

Запуск llama.cpp сервера на CPU с GPU оптимизацией
```shell
./build/bin/llama-server \
  -m /home/serverflow/Q4_K_S/Qwen3-Coder-480B-A35B-Instruct-Q4_K_S-00001-of-00006.gguf \
  --n-cpu-moe 999 \
  --n-gpu-layers 999 \
  --host 0.0.0.0 \
  --port 11435 \
  --no-mmap \
  --threads 64 \
  --ctx-size 16384 \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --repeat-penalty 1.05
```

Запуск llama.cpp сервера на CPU с GPU оптимизацией на двухпроцессорной системе
```shell
export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact 
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

numactl --cpunodebind=0 --membind=0 ./build/bin/llama-server \
  -m /home/serverflow/Q4_K_S/Qwen3-Coder-480B-A35B-Instruct-Q4_K_S-00001-of-00006.gguf \
  --n-cpu-moe 999 \
  --n-gpu-layers 999 \
  --host 0.0.0.0 \
  --port 11435 \
  --no-mmap \
  --threads 64 \
  --ctx-size 16384 \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --repeat-penalty 1.05
```

# Open-webui

Nginx reverse proxy
```nginx
server {
    listen 80;
    server_name openwebui.example.com;

    # Redirect all HTTP to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name openwebui.example.com;

    ssl_certificate     /etc/letsencrypt/live/openwebui.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/openwebui.example.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass         http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
    }
}
```

Запуск open-webui
```shell
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

# Бенчмарки
Для проверки влияния квантизации на llama.cpp была взята модель Qwen3-Coder. Для каждой модели был запущен промпт “Write a merge sort algorithm” с температурой 0, контекстом 16384 и обрезан вывод в 300 токенов (иначе по ходу генерации нелинейно возрастает KV-cache и новые токены генерируются чуть дольше, что влияет на среднее время на токен). Эксперимент проводился 3 раза и учитывалось среднее.

![[1_vs_2_cpu.png]]
![[llamacpp_vs_ollama.png]]
![[cpu_vs_cpu_and_gpu.png]]
![[popular_models.png]]
![[GPU.png]]

## Qwen3-Coder 480B

| Model       | Quantization   | RAM requirements | VRAM requirements | 2 CPU llama.cpp | 1 CPU llama.cpp | 1 CPU Ollama | 1 CPU + GPU llama.cpp |
| ----------- | -------------- | ---------------- | ----------------- | --------------- | --------------- | ------------ | --------------------- |
| Qwen3-Coder | Q2_K           | 162              | 12.9              | 4.53            | 7.23            | 4.03         | 8.59                  |
| Qwen3-Coder | Q4_0           | 260              | 14                | 3.06            | 5.33            | 3.79         | 6.88                  |
| Qwen3-Coder | Q6_K           | 370              | 18                | 2.68            | 3.88            | 2.55         | 5.08                  |
| Qwen3-Coder | Q4_K_M, Q4_K_S | 274              | 14.8              | 2.28            | 4.95            | 3.15         | 6.57                  |
| Qwen3-Coder | Q4_K_M, Q4_K_S | 260              | 14.8              | 2.54            | 5.11            | 4.06         | 6.82                  |
| Qwen3-Coder | Q8_0           | 480              | 20                | 1.52            | 3.07            | 2.75         | 4.09                  |
| Qwen3-Coder | Q5_K_M         | 320              | 16                | 1.86            | 4.23            | 2.5          | 5.76                  |
| Qwen3-Coder | Q4_K_XL        | 260              | 15                | 2.38            | 5.06            | 3.14         | 6.75                  |
| Qwen3-Coder | BF_16          | 874              | 33                | 1.01            | 1.72            | 1.26         | 0.67                  |

## Другие модели
| Model                         | Quantization | RAM requirements | VRAM requirements | 1 CPU llama.cpp | 1 CPU llama.cpp + GPU ускорением |
| ----------------------------- | ------------ | ---------------- | ----------------- | --------------- | -------------------------------- |
| Qwen3-235B-A22B-Thinking-2507 | Q8_0         | 225              | 13.9              | 4.57            | 6.06                             |
| DeepSeek-R1-0528 671B         | Q8_0         | 648              | 25                | 2.33            | 4.19                             |
| Kimi-K2-1026B                 | Q4_K_M       | 571              | 15.4              | 4.63            | 6.46                             |
| GPT OSS 120B                  | Q4_K_M       | 59               | 5                 | 21.99           | 25.8                             |
| GLM-4.5-Air 355B              | Q4_K_M       | 69               | 10.9              | 10.46           | 16.1                             |

Команда с нулевой температурой для `llama.cpp` с GPU ускорением
```shell
export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact 
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

numactl --cpunodebind=0 --membind=0 ./build/bin/llama-server \
  -m /home/serverflow/Q4_K_S/Qwen3-Coder-480B-A35B-Instruct-Q4_K_S-00001-of-00006.gguf \
  --n-cpu-moe 999 \
  --n-gpu-layers 999 \
  --host 0.0.0.0 \
  --port 11435 \
  --no-mmap \
  --threads 64 \
  --ctx-size 16384 \
  --temp 0.0 \
  --n-predict 300 \
  --min-p 0.0 \
  --top-p 1 \
  --top-k 0 \
  --repeat-penalty 1.0
```

Команда с нулевой температурой для `llama.cpp` только на процессоре
```shell
export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact 
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

./build/bin/llama-server \
  --model /mnt/services/models/Qwen3-Coder-480B-A35B-Instruct-Q4_K_S-00001-of-00006.gguf \
  --host 0.0.0.0 \
  --port 11435 \
  --threads 64 \
  --ctx-size 16384 \
  --n-gpu-layers 0 \
  -ot ".ffn_.*_exps.[^0]=CPU" \
  --mmapped \
  --temp 0.0 \
  --n-predict 300 \
  --min-p 0.0 \
  --top-p 1 \
  --top-k 0 \
  --repeat-penalty 1.0
```

Так, как в ollama нет встроенного счетчика токенов в секунду под запрос, для нее использовался такой скрипт, который тоже фиксирует нулевую температуру
```shell
#!/bin/bash
PROMPT="Write a merge sort algorithm."
MODEL=${1:-Q5_K_M:latest}
MAX_TOKENS=300

TOKEN_COUNT=0
START=$(date +%s.%N)

# Stream and count tokens
curl -s -N -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"stream\": true,
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$PROMPT\"}
    ],
    \"options\": {
      \"temperature\": 0.0,
      \"min_p\": 0.0,
      \"top_p\": 1,
      \"top_k\": 0,
      \"repeat_penalty\": 1.0
    }
  }" | while read -r line; do
    echo $line
    echo $TOKEN_COUNT
    content=$(echo "$line" | jq -r '.message.content // empty' 2>/dev/null)
    if [[ -n "$content" ]]; then
      ((TOKEN_COUNT++))
    fi
    [[ "$TOKEN_COUNT" -ge "$MAX_TOKENS" ]] && break
  done

END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc -l)
TPS=$(echo "scale=2; $MAX_TOKENS / $ELAPSED" | bc -l)

echo
echo "-----------------------------------------"
echo "Elapsed time: $ELAPSED seconds"
echo "Estimated tokens: $MAX_TOKENS"
echo "Average tokens/sec: $TPS"
```