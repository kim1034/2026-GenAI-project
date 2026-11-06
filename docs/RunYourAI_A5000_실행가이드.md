# RunYourAI RTX A5000 실행 가이드

이 문서는 GPU 담당자가 저장소를 처음 받은 뒤 실험을 시작하는 절차다. 대상은 **NVIDIA RTX A5000 1장, VRAM 24 GB**의 RunYourAI GPU Cloud 머신이다. 현재 대기 중이므로 실제 템플릿, 운영체제·드라이버·CUDA·디스크·가격·SSH 주소는 대여 시 화면에서 확인하고 `reports/progress.md`에 기록한다. 가격이나 설치 명령을 고정값으로 추정하지 않는다.

## 1. 머신 대여와 SSH 발급

1. [RunYourAI](https://www.runyour.ai/) 계정에 로그인한다. GPU Cloud에서 **RTX A5000, GPU 1개, VRAM 24 GB**인 머신을 확인한다. 대기 중인 항목은 대여 가능해질 때까지 기다린다. 공식 FAQ에 따르면 대기 신청만으로 즉시 대여·과금되지는 않지만, 실제 상태는 계정 화면에서 확인한다.
2. 시간제 On-Demand로 빌릴 경우 화면의 현재 요금과 예상 사용 시간을 확인하고 충분한 크레딧을 준비한다. 대여 가능한 머신을 선택한 뒤 **SSH 지원 템플릿**을 선택한다. 이 프로젝트에는 터미널, Python, CUDA가 필요하다.
3. 대여 후 `[마이페이지] 대시보드`에서 머신을 열고 `SSH 연결 정보`를 선택한다. 플랫폼이 제공하는 PEM 키를 내려받고 화면의 SSH 명령을 그대로 사용한다. 키와 접속 정보를 저장소에 넣거나 공유 채널에 붙여넣지 않는다. PEM 권한 오류가 나면 플랫폼의 [PEM 권한 안내](https://runyourai.gitbook.io/userguide/tutorial/ssh)를 따른다.
4. SSH 접속 후 아래 값을 기록한다. 값이 기대한 A5000/24GB와 다르면 실험을 시작하지 않고 머신 선택을 다시 확인한다.

```bash
nvidia-smi
python3 --version
df -h .
git --version
```

공식 안내: [머신 대여](https://runyourai.gitbook.io/userguide/eng-runyour-ai/gpu-cloud/machine-rental), [SSH 연결](https://runyourai.gitbook.io/userguide/tutorial/ssh), [FAQ](https://runyourai.gitbook.io/userguide/faq). 사이트 화면은 바뀔 수 있으므로 버튼 이름이 다르면 공식 안내와 현재 대시보드를 우선한다.

## 2. 코드 받기와 Python 환경

SSH로 접속한 Linux 셸에서 실행한다. 공개 저장소 URL은 다음과 같다.

```bash
git clone https://github.com/kim1034/2026-GenAI-project.git
cd 2026-GenAI-project
git rev-parse HEAD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

`python3 -m venv`가 실패하면 템플릿에 `venv` 패키지가 있는지 확인한다. Python은 이 저장소에서 로컬 검산에 사용한 3.11 계열을 우선한다. **CUDA PyTorch를 먼저** [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 Linux / Pip / Python / 머신 드라이버에 맞는 CUDA 빌드로 설치한다. `requirements-gpu.txt`는 `torch>=2.3,<3`을 요구한다. 설치된 torch가 이 범위에 있는지 확인한 뒤 나머지를 설치한다. 템플릿마다 드라이버가 다를 수 있어 CUDA wheel URL을 이 문서에 고정하지 않는다.

```bash
python -m pip install -r requirements-gpu.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
python -m scripts.check_env
python -m scripts.check_data
python -m pytest -q
```

`torch.cuda.is_available()`가 `True`이고 장치 이름이 RTX A5000이며, 데이터 검증과 29개 기존 테스트가 통과해야 한다. 설치된 **정확한** 패키지 버전은 `python -m pip freeze`로 보관한다. `scripts.check_env`는 추적 중인 `reports/env.json`을 갱신하므로 실험 설정 고정 전 검토·커밋한다. CUDA·패키지·모델 로딩이 확인되기 전에는 `run_experiment`를 시작하지 않는다.

## 3. 고정 모델과 첫 GPU 점검

`configs/base.yaml`의 모델은 `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`이다. `src/model.py`는 이를 float32/eager로 CUDA에 로드하고 28개 층·중간 차원 8960·SiLU를 검사한다. Hugging Face에서 처음 내려받을 때 네트워크와 모델 캐시 공간이 필요하다.

```bash
python -c 'import yaml; from src.model import load_model; c=yaml.safe_load(open("configs/base.yaml", encoding="utf-8")); m,t=load_model(c); print(m.config.model_type, len(m.model.layers), t.vocab_size)'
```

로딩 중 VRAM 부족 또는 결정론 연산 오류가 나면 출력과 환경 버전을 기록한다. 시험 조건을 임의로 float16이나 다른 attention 구현으로 바꾸지 않는다. 필요하면 사양 변경을 먼저 검토한다.

현재 예시 run YAML은 `configs/runs/dev_base_greedy.yaml`, `dev_base_A.yaml`, `dev_base_C_pilot.yaml`이다. 실행기는 YAML의 **분할 전체**를 처리한다. 소량 샘플만 돌리는 CLI 옵션은 아직 없으므로 긴 실행 전에 리허설 방법을 코드로 마련하거나 문제 수와 예상 시간을 판단한다.

```bash
python -m scripts.run_experiment configs/runs/dev_base_greedy.yaml
python -m scripts.calibrate_rho
python -m scripts.run_experiment configs/runs/dev_base_A.yaml
python -m scripts.build_bank configs/runs/dev_base_C_pilot.yaml
python -m scripts.run_experiment configs/runs/dev_base_C_pilot.yaml
```

순서의 이유: greedy는 레이블과 기준 출력, `calibrate_rho`는 상대 노이즈의 층별 크기, A는 노이즈 없는 표집 기준선, C는 Gaussian 파일럿이다. C의 노이즈 뱅크는 실행 전에 생성해야 한다. `calibrate_rho`와 greedy가 같은 `dev_base`를 각각 계산하므로 실제 GPU 시간을 함께 기록한다. 완성되지 않은 파일럿을 본 실험 결과로 취급하지 않는다.

각 실행 뒤 `outputs/manifests/*.json`의 `status`, `completed_problems`, `sample_count`, `elapsed_seconds`, `gpu_max_memory_bytes`, `hook_calls`, `git_head`와 `outputs/samples/*.jsonl`을 확인한다. 같은 설정을 재실행하면 완성 결과를 확인하고, 중단된 문제는 기존 JSON에서 이어가도록 구현돼 있다. 출력과 해시가 맞지 않으면 원인을 찾기 전까지 덮어쓰지 않는다.

## 4. 시간·메모리 예산과 본 실험으로 넘어가는 기준

`configs/base.yaml`은 총 120 GPU 시간과 A5000을 가정한다. `assumed_t_run_s`는 실측치가 아니다. 16응답·32응답의 문제당 시간을 각각 측정한 뒤 다음처럼 재계산한다.

```bash
python -m scripts.budget --rows16-seconds <실측_초> --rows32-seconds <실측_초>
```

24GB에서 32응답과 최대 512토큰을 처리할 수 있는지 확인한다. 리허설 출력이 반복 실행에서 동일한지, hook 호출 수·노이즈 크기·마지막 위치 로짓이 의도대로인지 검증한다. 충분한 근거 없이 설정의 dtype, 행 수, 노이즈 세기 또는 분할을 바꾸지 않는다. 개발 선택과 LOCK의 상세 게이트는 [실행 계획](실행계획.md)과 [인계 문서 v1.3](../GUE_Gaussian_환각탐지_인계문서_ClaudeCode용_v1.3.md)을 따른다.

**test는 아직 봉인돼 있다.** `LOCK.md`에 설정 해시를 기록하고 그 커밋에 `lock-v1` 태그를 붙여야 실행기가 test 로드를 허용한다. LOCK 전에 test 모델 출력을 보거나 trial 실행을 하지 않는다.

## 5. 작업 결과 보존과 머신 반납

`bank/`, `outputs/`는 `.gitignore`에 포함돼 Git clone/push로 전달되지 않는다. `calib/rho.json`, 갱신된 보고서·설정·LOCK과 설치 버전도 별도 보존 대상을 확인한다. 대여 머신이 중지·반환되면 컨테이너의 작업 데이터와 환경이 사라질 수 있으므로, 세션 중간에도 저장소의 영구 볼륨 또는 팀이 정한 백업 위치에 복사하고 파일 해시를 확인한다. SSH PEM·토큰·개인 경로는 커밋하지 않는다. 백업을 확인한 뒤 대시보드에서 머신을 반납해 과금을 끝낸다. [RunYourAI FAQ](https://runyourai.gitbook.io/userguide/faq)는 반환 후 미백업 데이터가 복구되지 않는다고 안내한다.
