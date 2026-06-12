# 느린 shell / 명령 지연 해결 (conda on Lustre)

다른 환경(서버/노드)에도 적용할 수 있는 진단·수정 메모.

## 증상
`git`, `ls` 등 **모든 명령이 7~8초씩 지연** (python뿐 아니라 전부).

## 원인
`~/.bashrc`의 conda init 블록이 **매 shell 시작마다** 아래 줄로 conda(파이썬 스크립트)를
실행하는데, 그 conda가 **네트워크 파일시스템(Lustre, 예: `/data1`) 위**에 설치돼 있어
cold 읽기로 수 초씩 stall한다.

```bash
__conda_setup="$('/data1/.../miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
```

이 환경은 명령마다 shell을 `.bashrc`부터 새로 띄우므로 모든 명령이 이 비용을 낸다.

## 진단 (한 줄씩)
```bash
time bash -c   true     # 프로필 X → 0.00s (정상)
time bash -lic true     # 프로필 O → 7~8s 면 .bashrc가 범인
df -T <느린경로>         # FSTYPE가 lustre/nfs면 네트워크 FS 확인

# (선택) python 자체도 느린지: site 로딩이 범인인지 확인
time python3 -S -c pass # site 스킵 → 빠르면 .pth/site-packages 읽기가 원인
time python3 -c pass    # 전체 startup (cold일 때 수 초면 env가 네트워크 FS)
```

## 수정 — 정적 `conda.sh`로 교체 (python 훅 제거, 기능 동일)
적용 전 백업:
```bash
cp ~/.bashrc ~/.bashrc.preconda.bak
```

`~/.bashrc`의 conda 블록을 아래로 교체 (경로는 환경에 맞게):
```bash
# >>> conda initialize >>>
if [ -f "/data1/.../miniconda3/etc/profile.d/conda.sh" ]; then
    . "/data1/.../miniconda3/etc/profile.d/conda.sh"
else
    export PATH="/data1/.../miniconda3/bin:$PATH"
fi
# <<< conda initialize <<<
```
- `conda shell.bash hook`(python 실행) 줄을 삭제하고 정적 `conda.sh` source만 남긴다.
- `conda activate`는 그대로 동작한다.
- 결과: shell 시작 **7~8s → 0.02s**.

## 한계 / 근본 해결
이 수정은 **shell 시작 비용**만 없앤다. `python train.py`처럼 무거운 import(torch 등
수천 파일)는 conda env가 네트워크 FS에 있는 한 첫 실행이 여전히 느리다.
근본 해결은 **miniconda/env를 로컬 디스크(ext4 등)로 이전**하는 것.
- 데이터셋·체크포인트(대용량)는 네트워크 FS에 두고, **코드 + conda env만 로컬**로 두는 게 정석.
- conda-pack으로 통째 이전 가능:
  ```bash
  conda install -n base conda-pack -y
  conda pack -n <env> -o /tmp/<env>.tar.gz
  mkdir -p ~/envs/<env> && tar xzf /tmp/<env>.tar.gz -C ~/envs/<env>
  source ~/envs/<env>/bin/activate && conda-unpack
  ```

## 이 환경에서 확인된 수치 (참고)
- `/data1` = Lustre (`*@tcp:/GSDATA3`), conda env(`/data1/.../envs/reco`, 8.0G)가 그 위에 설치됨.
- `bash -lic true`: 7.5~8.4s → 수정 후 0.02s.
- `conda shell.bash hook` 단독: warm 1.19s / cold 수 초.
- `python3 -S -c pass`: 0.012s (빠름) vs `python3 -c pass` cold: 20s+ hang.
- strace상 stdlib 한 파일 읽다 7.4s stall 관측 (Lustre cold read).
