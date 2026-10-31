# 국회의원 발언 추적

국회의원 이름 + 정책 주제를 넣으면 관련 발언을 모아서 입장을 정리해주는 도구.

국회 회의록 PDF를 OCR로 읽어서 발언 단위로 파싱하고, 임베딩 검색으로 주제 관련 발언을 찾은 뒤 LLM으로 입장을 요약한다.

## 기능

- **발언 검색 + 일관성 분석**: 의원 + 주제로 검색 → 시점별(초반/중반/후반) 입장 변화, 모순 발언, 일관성 점수
- **입장 분류**: 검색된 발언마다 찬성/반대/중립/판단유보 + 확신도를 구조화된 출력으로 분류 (질문·진행 멘트가 "입장"으로 둔갑하는 것을 방지)
- **주요 키워드**: 의원 선택 시 그 의원 발언에서 실제로 히트가 있는 주제만 순위로 보여줌 (없는 주제를 검색해서 무관한 발언이 끌려오는 것을 예방)
- **의원 프로필**: 의원이 주로 다루는 주제 top N + 주제별 입장 요약 (SSE로 순위 먼저, 요약은 끝나는 대로 하나씩) + 정당·지역구·선수·위원장 역임 등 실제 인물 정보
- **RAG 챗봇**: 자유 질문에 회의록 발언만 근거로 답변. 자기소개/잡담 질문은 검색 없이 바로 응대하고, 포괄적 질문은 프로필 기능을 재사용하며, 후속 질문은 직전 대화를 참고해 검색 문장을 다시 씀
- **웹 화면**: FastAPI + 정적 HTML — 의원/주제를 코드 수정 없이 그때그때 선택해서 조회, 발언 원문 대조, Chart.js로 입장 변화 시각화
- **시각화(CLI)**: `analyze.py chart` — 시계열 유사도/점수 게이지/구간별 발언 수 차트를 이미지로 출력

## 설치

```powershell
py -m pip install -r requirements.txt
```

Tesseract OCR 엔진 별도 설치 필요 (한국어 언어팩 포함):

```powershell
winget install --id tesseract-ocr.tesseract -e
```

`.env` 파일에 API 키 설정:

```
ASSEMBLY_API_KEY=열린국회정보_API_키
OPENAI_API_KEY=OpenAI_API_키
```

## 사용법

```powershell
# 1. 의원 명단 수집 (최초 1회)
py collect.py members --key YOUR_API_KEY

# 2. 회의록 수집 (PDF 다운로드 → OCR → 파싱 → DB 저장)
py collect.py meetings --type 본회의 --dae 21 --years 2020 2021 2022 2023 2024
py collect.py meetings --type 위원회 --dae 21 --years 2020 2021 2022 2023 2024

# 3. 의원별 청크 임베딩
py embed.py --member 주호영

# 4. 검색 + 분석 (CLI)
py analyze.py search  --member 주호영 --topic 연금개혁
py analyze.py profile --member 주호영 --topics 3
py analyze.py chart   --member 주호영 --topic 연금개혁 --save

# 5. 웹 화면 (선택)
py -m uvicorn api:app --reload --port 8000
# 브라우저에서 http://127.0.0.1:8000
```

### 웹 화면 (FastAPI)

CLI 대신 브라우저에서 의원/주제를 골라가며 조회하고 싶으면 로컬 서버를 띄운다. `analyze.py`의 함수를 그대로 불러써서 만들었고, 빌드 도구 없이 정적 HTML 한 장(`static/index.html`)이 API와 같은 포트에서 서빙된다. 아직 임베딩 안 된 의원을 프로필 조회하면 서버가 자동으로 임베딩부터 실행한다(SSE로 진행률 표시).

| 엔드포인트 | 설명 |
|---|---|
| `GET /api/members` | 전체 349명 의원 목록, 정당별로 그룹핑 + 임베딩 여부(`embedded`) 표시 |
| `GET /api/member_bio?member=` | 의원 실제 정보 (정당·지역구·선수, 위원장 역임 여부) — 임베딩 여부와 무관하게 항상 조회 가능 |
| `GET /api/topics?member=` | 그 의원의 주요 키워드 (고정 12개 주제 중 실제 히트 있는 것만 순위순) |
| `GET /api/search?member=&topic=&top=` | 주제 검색 + 일관성 분석 (`top`: 10/30/50/100, 기본 30) |
| `POST /api/chat` | RAG 챗봇 — `{member, question, history, top}`을 받아 자유 질문에 답변 |
| `GET /api/profile?member=&topics=` | 의원 프로필 (주제별 입장 요약, 한 번에 반환) |
| `GET /api/profile/stream?member=&topics=` | 위와 같지만 SSE로 순위 → 요약 순서대로 흘려보냄 |
| `GET /api/speech/{speech_id}` | 발언 원문 조회 (검색 결과 발췌문과 원문 대조용) |

## 파이프라인

```
국회 OpenAPI → PDF 다운로드 → 페이지 이미지 렌더링 → Tesseract OCR → 발언 파싱 → speeches / chunks 테이블   [collect.py]
                                                                            ↓
                                            청크 임베딩(text-embedding-3-small) → SQLite embeddings 테이블   [embed.py]
                                                                            ↓
                                         주제 쿼리 → 코사인 유사도 검색 → speech_id 역참조 → 원문             [analyze.py]
                                                                            ↓
                                                  GPT 일관성 분석 / 프로필 요약 → 점수 · 모순 · 차트           [analyze.py]
```

## 파일 구조

| 파일 | 역할 |
|---|---|
| `collect.py` | 회의록 API 수집 + PDF 다운로드 + DB 저장, 의원 명단 수집, party/district 조인 (서브커맨드: `meetings`, `members`, `join`) |
| `parse.py` | PDF → 이미지 렌더링 → Tesseract OCR → 발언 파싱 → 청크 분할 |
| `embed.py` | 의원별 청크를 OpenAI 임베딩으로 변환해 DB에 저장 (위원장·소위원장 등 진행 role 발언은 제외) |
| `analyze.py` | 검색 + 입장 분류 + 일관성 분석 + 의원 프로필 + 챗봇 + 차트 (서브커맨드: `search`, `profile`, `chart`) |
| `api.py` | 위 함수들을 노출하는 FastAPI 서버 (엔드포인트 8개, 위 표 참고) |
| `static/index.html` | 웹 화면 (정적 HTML + fetch/EventSource + Chart.js, `api.py`가 같이 서빙) |

벡터 검색은 ChromaDB가 아니라 numpy 브루트포스 코사인 유사도를 쓴다 (의원당 청크가 최대 수천 개 수준이라 성능 문제 없음, 자세한 이유는 `CLAUDE.md` 참고).

## 주의

- PDF는 디지털 네이티브 문서지만 텍스트 레이어를 직접 못 쓰는 2단 컬럼 문제가 있어 이미지 OCR로 처리한다. 자세한 배경과 미해결 이슈(한자 이름 미인식 등)는 `CLAUDE.md` 참고.
- `.env`, `output/`, `members.db`/`members.csv`는 git에 커밋하지 않는다 (`.gitignore` 참고).
