# 실행 변경 기록

## D-101 — LOCK 전 분할 ID 파일의 줄바꿈 형식 정규화

Windows에서 생성한 `.ids`의 CRLF 바이트를 기준으로 SHA256SUMS가 작성됐으나, Git이 Linux 복제본을 LF로 변환해 기록된 바이트 해시와 달라졌다. 기존 Windows 원본 4개와 기존 체크섬 목록은 `outputs/reproducibility/split_eol_original/`에 원래 바이트와 해시 이름으로 보존했다.

각 파일의 디코딩된 ID 목록을 순서까지 대조해 동일함을 확인했다. 50/200/50/1319개, 분할 내부 중복 없음, 분할 간 중복 없음과 train/test ID 접두사를 검증한 뒤 LF 바이트로 기록하고 해시를 재계산했다. old/new/이전 기록 해시 대응표는 `reports/split_eol_normalization.json`에 있다. 새로운 체크섬은 Linux 원본 파일의 실제 해시와 일치한다.

분할을 다시 추출하지 않았고 raw JSONL, 생성 응답, 기존 manifest와 역사 기록은 변경하지 않았다. 이후 결정과 LOCK은 canonical LF 해시를 사용한다. `prepare_data.py`의 ID·체크섬 출력도 LF를 명시하도록 수정했다.
