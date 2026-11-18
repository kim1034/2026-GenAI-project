# 팀 공유용 실험 자료

기준일: 2026-11-16. **승인된 V1.4 축소 범위 완료 / A·B 본시험 미실행.**

- [브라우저 보고서](https://kim1034.github.io/2026-GenAI-project/results/) · [HTML 원본](index.html)
- [결과 상세](report.md) · [수행 과정](process.md) · [A·B 보완 계획 초안](plan.md) · [후속 기록 양식](record-template.md)
- [지표 CSV](data/metrics.csv) · [비교 CSV](data/comparisons.csv) · [K 곡선 CSV](data/k_curve.csv) · [공유용 요약 JSON](data/summary.json)
- [확정 LOCK](evidence/LOCK.md) · [출처·해시 목록](evidence/provenance.json)

HTML은 저장소를 ZIP으로 내려받아 `docs/results/index.html`을 열어도 읽을 수 있다. 그림·자료 링크를 사용하려면 하위 폴더를 함께 보관한다. 인쇄 메뉴에서 PDF 저장도 가능하다.

공개 자료는 원본 집계·그림과 공유용 파생 문서다. 전체 원시 응답·뱅크·로그·코드 bundle 약 3.37GB는 로컬 검증 백업으로 보존하며 이 저장소에는 올리지 않았다. 외부 모델 가중치 캐시는 백업에서 제외됐다. 현재 main 코드와 실제 실행한 봉인 코드의 일치를 주장하지 않는다. 실제 실행 LOCK 커밋은 `1d66a23e344f273eeb8a5f5cd0d7a18b9135d14e`이며 로컬 bundle에 보존되어 있다.
