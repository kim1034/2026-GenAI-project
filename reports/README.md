# 실험 자료 안내

최종 정리일: 2026-11-16 (한국시간)

**승인된 V1.4 축소 실험·분석·백업 검증은 완료됐다. 원래 목표인 기존 Uniform 재현·비교는 미완료다. 사용자는 GPU 재대여를 위한 후속 계획과 과정·결과 기록 정리를 요청했다. 새 계획은 검토용 초안이며 아직 등록하거나 실행하지 않았다. 실제 GPU 반납·재대여 여부는 확인하지 않았다.**

## 먼저 읽을 자료

| 자료 | 내용 |
| --- | --- |
| [실험결과_v1.4.md](실험결과_v1.4.md) | 최종 조건, 결과 표·그림, 통계 해석과 한계 |
| [다음 GPU A·B 보완 계획](다음_GPU_AB_보완계획.md) | 현재 초안: 새 평가 없이 기존 test의 A·B만 추가, 30시간 목표 |
| [실험 기록 양식](실험기록_양식.md) | 실행 원장, 장애·재개, 결과·검수·백업 기록 기준 |
| [Astra 최종 해석](../outputs/reproducibility/astra_final_interpretation_20261116.json) | 사전 기준에 따른 실용적 동등성 판단과 적용 범위 |
| [프로젝트 완료 검증](../outputs/reproducibility/project_completion_verified.json) | 본시험·분석·백업 완료 근거 |
| [최종 백업 감사](../outputs/reproducibility/final_backup_audit/final_supplemental_evidence_verified.json) | 8,382개 파일·3,370,723,176바이트, 누락·해시 불일치 0건 |
| [외부 GPU 작업 진단](../outputs/reproducibility/gpu_external_workload_diagnostic_20261116.json) | 반납 전 확인된 프로젝트 밖 채굴 프로세스와 조사 한계 |

핵심 결과는 GUE(D)−Gaussian(C)의 AUROC 차이 −0.001735, 95% 신뢰구간 [−0.012643, 0.009019]이다. 신뢰구간 전체가 사전에 정한 ±0.02 안에 들어가므로 **이번 모델·데이터·고정 설정에서 실용적 동등성**을 확인했다. GUE의 우월성이나 보편적 동등성을 뜻하지 않는다.

## 확정 조건과 원본 분석

- [LOCK.md](../outputs/remote_backups/continuous/latest/LOCK.md): 최종 시험 범위와 봉인 기록.
- [final_plan.json](../outputs/remote_backups/continuous/latest/configs/generated/budget120/final/final_plan.json): 공통 α와 조건별 실행 설정 경로.
- [base.yaml](../outputs/remote_backups/continuous/latest/configs/generated/budget120/pilot/base.yaml): 모델·프롬프트·교란·생성 공통 설정. 실제 사용 범위는 V1.4 LOCK과 함께 읽는다.
- [analysis.json](../outputs/reproducibility/final_analysis_backup/outputs/analysis/budget120/test/analysis.json): 반올림하지 않은 최종 통계와 입력 출처.
- [metrics.csv](../outputs/reproducibility/final_analysis_backup/outputs/analysis/budget120/test/metrics.csv) · [comparisons.csv](../outputs/reproducibility/final_analysis_backup/outputs/analysis/budget120/test/comparisons.csv): 결과 표 원본.
- [figure2.png](../outputs/reproducibility/final_analysis_backup/outputs/analysis/budget120/test/figure2.png): 응답 수별 AUROC와 조건 간 차이 그림.

## 백업 위치와 보존 방법

주 백업은 `outputs/remote_backups/continuous/latest/`에 있다. 별도 검증 스냅샷과 보충 파일은 `outputs/reproducibility/`에 있다. [파일별 원격 해시·로컬 경로 대응표](../outputs/reproducibility/final_backup_audit/source_mapping_full_sha256.json)를 보존해야 모든 검증 파일의 위치를 찾을 수 있다.

**`outputs/remote_backups/`와 `outputs/reproducibility/`를 함께 보존한다.** 결과 Markdown만 옮기면 원시 응답·노이즈 뱅크·코드 이력까지 보존되는 것은 아니다. 외부 모델 캐시의 가중치 파일 자체는 백업 범위 밖이며, 고정 revision과 관련 해시 기록을 보존했다. SSH 개인키나 비밀번호를 자료 묶음에 포함하지 않는다.

## 진행 기록을 읽을 때

[progress.md](progress.md)와 [gpu_session_20261112.md](gpu_session_20261112.md)의 과거 행·본문은 당시 상태를 기록한 이력이다. 현재 완료 여부는 이 안내와 최종 검증 기록을 기준으로 읽는다. 초기 v1.3 계획 전체의 가설을 모두 수행한 것은 아니며, 제외 항목과 인간 2인 서명 미충족은 최종 결과 보고서에 명시했다.

## 후속 연구 및 반납 상태

- 후속 전체안을 검토했지만 등록·실행하지 않았다. 사용자가 30시간 목표로 범위를 줄이도록 요청해 [A·B만 보완하는 초안](다음_GPU_AB_보완계획.md)으로 수정했다. 기존 공식 test의 A·B는 여전히 미실행이며 별도 후속 설계 사용은 **0/1회**다. [이전 전체안](다음_GPU_실험계획.md)은 초안 이력으로 보존한다.
- GPU에서 프로젝트 밖 SRBMiner·onezerominer 등의 프로세스가 관찰됐다. 실제 설치 주체와 원인은 확정하지 못했으며, 임의 종료나 원격 파일 변경은 하지 않았다.
- 봉인 코드·주요 산출물의 해시 일치는 별도로 확인했다. 이 검증만으로 과거 외부 작업의 모든 영향이나 시스템 전체의 무결성을 증명하지는 않는다.
- 사용자가 GPU를 직접 반납한다. 반납 준비는 완료됐지만 대시보드에서 실제 반납됐다고 기록하지 않는다.
