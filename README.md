# GUE · Gaussian 환각 탐지 실험

GSM8K 문제에서 고정된 `Qwen/Qwen2.5-1.5B-Instruct`의 추론 중 MLP 활성에 노이즈를 주입하고, 생성 답의 엔트로피가 greedy 오답을 얼마나 잘 구분하는지 비교한다. 모델 가중치는 학습하지 않는다.

## 처음 보는 팀원 또는 에이전트

1. [인계 가이드](docs/인계_및_코드지도.md): 연구 목적, 파일별 역할, 구현 상태, 실험 규칙과 남은 작업.
2. [RunYourAI GPU 실행 가이드](docs/RunYourAI_A5000_실행가이드.md): A5000 대여·접속, 환경 준비, 첫 실행, 결과 보존.
3. [실행 계획](docs/실행계획.md): 단계별 게이트와 LOCK 이후 시험 절차.
4. [인계 문서 v1.3](GUE_Gaussian_환각탐지_인계문서_ClaudeCode용_v1.3.md): 실험의 상세 사양. 충돌할 때 이 문서의 v1.3 데이터 출처와 Part 2를 우선한다.

## 현재 상태

- 원본 GSM8K train 7,473행과 test 1,319행의 출처·SHA-256·정답 파싱을 검증했다.
- 개발 300문제를 `dev_base` 50, `dev_select` 200, `dev_anal` 50으로 고정했다.
- CPU 검산 29개가 통과했다. Qwen/CUDA의 실제 forward, GPU 메모리, 실행 시간은 검증 전이다.
- 목표 GPU는 RunYourAI의 NVIDIA RTX A5000 24 GB 한 장이다. 장비 대기 중이다.
- `LOCK.md`는 아직 없다. test 모델 실행은 금지되며 코드도 이를 검사한다.

## 로컬 사전 점검

저장소 루트에서 Python 3.11 환경으로 실행한다.

```bash
python -m scripts.check_data
python -m pytest -q
```

GPU 명령과 설치 절차는 [GPU 실행 가이드](docs/RunYourAI_A5000_실행가이드.md)를 따른다. `bank/`와 `outputs/`는 Git에서 제외되므로 별도 백업이 필요하다.
