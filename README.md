# GUE · Gaussian 환각 탐지 실험

GSM8K 문제에서 고정된 `Qwen/Qwen2.5-1.5B-Instruct`의 추론 중 MLP 활성에 노이즈를 주입하고, 생성 답의 엔트로피가 greedy 오답을 얼마나 잘 구분하는지 비교한다. 모델 가중치는 학습하지 않는다.

## 처음 보는 팀원 또는 에이전트

1. [인계 가이드](docs/인계_및_코드지도.md): 연구 목적, 파일별 역할, 구현 상태, 실험 규칙과 남은 작업.
2. [RunYourAI GPU 실행 가이드](docs/RunYourAI_A5000_실행가이드.md): A5000 대여·접속, 환경 준비, 첫 실행, 결과 보존.
3. [실행 계획](docs/실행계획.md): 단계별 게이트와 LOCK 이후 시험 절차.
4. [인계 문서 v1.3](GUE_Gaussian_환각탐지_인계문서_ClaudeCode용_v1.3.md): 실험의 상세 사양. 충돌할 때 이 문서의 v1.3 데이터 출처와 Part 2를 우선한다.

## 현재 상태 — 2026-11-16

**승인된 V1.4 축소 실험 완료. A·B 본시험과 원논문 기준선 재현은 미완료.**

- [팀 공유용 웹 보고서](https://kim1034.github.io/2026-GenAI-project/results/) · [결과 Markdown](docs/results/report.md) · [공유 자료 목록](docs/results/README.md)
- Qwen2.5-1.5B-Instruct / GSM8K test 1,319문항에서 Greedy·C(Gaussian)·D(GUE)·E(주변분포 독립 대조군) 총 64,631응답을 완료했다.
- 주 비교 D−C AUROC 차이 −0.001735, 95% CI [−0.012643, 0.009019]. 사전 ±0.02 기준 내 실용적 동등성 확인이며 이 모델·데이터·설정에 한정된다.
- A(T=0.5)는 개발 실행만 완료했고 B(Uniform)는 본시험을 하지 않았다. Greedy는 A가 아니며 E는 Uniform이 아니다.
- [A·B 보완 계획](docs/results/plan.md)은 30시간 목표 초안으로 미등록·미실행이다. [수행 과정](docs/results/process.md)과 [확정 LOCK 사본](docs/results/evidence/LOCK.md)을 공개한다.
- 실행 코드와 전체 원시 자료는 로컬 검증 백업에 보존한다. 공개 main 코드가 실제 LOCK 코드와 같다고 주장하지 않는다. [보존 범위·한계](docs/results/README.md)를 확인한다.

초기 인계 가이드와 아래 사전 점검은 준비 단계 문서다. 최신 상태와 해석은 위 보고서를 기준으로 읽는다.

## 로컬 사전 점검

저장소 루트에서 Python 3.11 환경으로 실행한다.

```bash
python -m scripts.check_data
python -m pytest -q
```

GPU 명령과 설치 절차는 [GPU 실행 가이드](docs/RunYourAI_A5000_실행가이드.md)를 따른다. `bank/`와 `outputs/`는 Git에서 제외되므로 별도 백업이 필요하다.
