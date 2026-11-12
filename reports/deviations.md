# 실행 변경 기록

## D-101 — LOCK 전 분할 ID 파일의 줄바꿈 형식 정규화

Windows에서 생성한 `.ids`의 CRLF 바이트를 기준으로 SHA256SUMS가 작성됐으나, Git이 Linux 복제본을 LF로 변환해 기록된 바이트 해시와 달라졌다. 기존 Windows 원본 4개와 기존 체크섬 목록은 `outputs/reproducibility/split_eol_original/`에 원래 바이트와 해시 이름으로 보존했다.

각 파일의 디코딩된 ID 목록을 순서까지 대조해 동일함을 확인했다. 50/200/50/1319개, 분할 내부 중복 없음, 분할 간 중복 없음과 train/test ID 접두사를 검증한 뒤 LF 바이트로 기록하고 해시를 재계산했다. old/new/이전 기록 해시 대응표는 `reports/split_eol_normalization.json`에 있다. 새로운 체크섬은 Linux 원본 파일의 실제 해시와 일치한다.

분할을 다시 추출하지 않았고 raw JSONL, 생성 응답, 기존 manifest와 역사 기록은 변경하지 않았다. 이후 결정과 LOCK은 canonical LF 해시를 사용한다. `prepare_data.py`의 ID·체크섬 출력도 LF를 명시하도록 수정했다.

## D-102 — 별도 GPU 리허설을 선택 자료의 CPU 파이프라인 점검으로 대체

2026-11-12 사용자 요청과 Astra 검토에 따라 dev_anal C/D/E 각 50문제·16응답의 추가 생성 150문제 실행을 생략한다. 이미 완료한 dev_anal greedy·기울기 진단은 그대로 보존한다. 선택이 완전히 끝난 dev_select 200문제의 C/D/E 각 16응답과 기존 greedy를 최종 분석 코드에 넣어 점수·bootstrap·표·그림·출처 검증을 CPU에서 끝까지 실행한다.

선택 자료를 재사용하므로 scope는 `development_pipeline_check`, `selection_data_reused=true`, `independent_evaluation=false`로 기록한다. 이 점검 결과로 세기나 조건을 다시 선택하지 않으며 독립 성능 검증이나 확증 결론을 주장하지 않는다. LOCK G3는 이 기술 점검으로 대체한다. 시험 1,319문제, noisy 조건 16응답, 공통 세기, 통계 키와 봉인 규칙은 유지한다.

진행 중인 파일럿의 base·analysis·protocol 스냅샷과 src/runner 바이트는 변경하지 않는다. 운영 보충 기록 `configs/development_check_policy.json`을 출처 해시 및 LOCK에 추가한다. 원래 리허설 대기 프로세스는 종료했으며, 기존 리허설 실행 코드도 해당 정책이 있으면 GPU 추론을 거부한다. 이는 새 가설이나 별도 후속 실험이 아니다.
