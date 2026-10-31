# FastAPI 서버 — analyze.py의 함수를 그대로 불러써서 엔드포인트로 노출.
# 박사님 화면가이드 리뷰 권장: 정적 페이지 대신 로컬 서버로 가서
# 의원 추가/주제 추가를 코드 수정 없이 다 볼 수 있게 함.
#
# 실행: py -m uvicorn api:app --reload --port 8000
#
# 엔드포인트 5개:
#   GET /api/members                     임베딩된 의원 목록
#   GET /api/member_bio?member=          의원 실제 정보(정당·지역구·선수·위원장 역임)
#   GET /api/topics?member=              그 의원의 주요 키워드(고정 12개 주제 중 히트 있는 것만, 순위순)
#   GET /api/search?member=&topic=&top=  주제 검색 + 일관성 분석
#   POST /api/chat                       챗봇 — 자유 질문에 발언 근거로 답변 (직전 대화 history 포함)
#   GET /api/profile?member=&topics=     의원 프로필 (주제별 입장 요약)
#   GET /api/speech/{speech_id}          발언 원문 조회 (인용문 원문 대조용)

import json, os, sqlite3, time
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import analyze

app = FastAPI(title='국회발언추적 API')

# analyze.py의 ValueError 문구는 "py embed.py --member ... 실행하세요"처럼
# 개발자용 CLI 안내라 화면에 그대로 보여주면 안 됨 — 사용자용 문구로 대체해서
# 임베딩 안 된 의원을 만나는 모든 엔드포인트가 같은 메시지를 쓰게 함.
_NOT_EMBEDDED_MSG = '아직 검색 준비가 안 된 의원입니다 (임베딩 전).'

# 챗봇 질문 하나에 GPT 호출이 최대 3번까지 이어진다(_classify_question_type →
# _rewrite_followup_query → classify_stances → 답변 생성. 2026-09-22에
# meta/offtopic/broad 세 번의 분류 호출을 하나로 합쳐서 줄임). 아직 인증·요청
# 상한이 없는 상태라 누군가 짧은 시간에 반복 호출하면 비용이 그대로 쌓이는
# 문제가 있어서, 정식 인증을 붙이기 전 단기 조치로 IP당 시간당 요청 수를
# 제한한다.
_CHAT_RATE_LIMIT = 20      # IP당 시간당 허용 요청 수
_CHAT_RATE_WINDOW = 3600   # 초
_chat_request_log = defaultdict(list)   # ip -> [timestamp, ...]


def _check_chat_rate_limit(ip):
    now = time.time()
    log = _chat_request_log[ip]
    while log and now - log[0] > _CHAT_RATE_WINDOW:
        log.pop(0)
    if len(log) >= _CHAT_RATE_LIMIT:
        return False
    log.append(now)
    return True

_client = None


def get_client():
    """OpenAI 클라이언트는 요청마다 새로 안 만들고 프로세스에서 하나만 재사용."""
    global _client
    if _client is None:
        _client = analyze.get_client()
    return _client


@app.get('/api/members')
def list_members():
    """전체 의원 명단(정당별)을 발언 수와 함께 반환한다. 임베딩이 끝난
    의원만 보여주던 걸 바꿈 — 지금은 21대 회의록 일부만 수집한 상태라
    다들 발언 수가 적을 수밖에 없는데, 5명만 보이면 나머지 344명은
    존재조차 확인할 길이 없었음. 대신 `embedded`로 검색 가능 여부를
    같이 내려서, 프런트가 아직 임베딩 안 된 의원을 구분해서 보여준다.

    처음엔 의원마다 EXISTS 서브쿼리를 돌렸는데(349명 × chunks 70만 행 스캔),
    chunks(name)에 인덱스가 없어서 로딩에 2분 넘게 걸렸다. 인덱스를 추가해도
    349번 반복 자체가 남아서 여전히 6~7초 걸림 — 임베딩된 이름 집합을 딱 한 번만
    조회해서(embeddings 테이블은 2만여 건뿐이라 가벼움) 파이썬에서 매칭하는
    쪽으로 바꿔서 서브쿼리 반복을 없앤다."""
    con = sqlite3.connect(analyze.DB_PATH)
    rows = con.execute('''
        SELECT ms.name, ms.party, COUNT(*) AS speech_count
        FROM member_speeches ms
        GROUP BY ms.name
        ORDER BY ms.party, speech_count DESC
    ''').fetchall()
    # JOIN으로 쓰면(chunks c JOIN embeddings e) SQLite가 70만 행짜리 chunks를
    # 기준으로 스캔해버려서(embeddings가 2만여 행뿐인데도) 3.8초 걸렸다.
    # IN 서브쿼리로 바꾸면 작은 embeddings 쪽에서 시작해 chunks를 PK로 찾아가서
    # 0.4초로 줄어든다.
    embedded_names = {
        r[0] for r in con.execute('''
            SELECT DISTINCT name FROM chunks WHERE id IN (SELECT chunk_id FROM embeddings)
        ''').fetchall()
    }
    con.close()
    return [
        {'name': r[0], 'party': r[1], 'speech_count': r[2], 'embedded': r[0] in embedded_names}
        for r in rows
    ]


@app.get('/api/member_bio')
def api_member_bio(member: str):
    """프로필 화면 상단에 보여줄 실제 인물 정보(정당·지역구·선수, 위원장
    역임 여부) — GPT 요약과는 별개로 DB에 있는 사실 정보를 그대로 노출.
    임베딩 여부와 무관하게 항상 조회 가능(그냥 members 테이블 조회라 빠름)."""
    bio = analyze.get_member_bio(member)
    if bio is None:
        raise HTTPException(status_code=404, detail='의원 정보를 찾을 수 없음')
    return bio


@app.get('/api/topics')
def api_topics(member: str):
    """의원 선택 직후 보여줄 "주요 키워드" — 12개 고정 주제를 임베딩 검색만
    돌려서(GPT 호출 없음, 빠름) 실제 히트가 있는 것만 순위로 반환한다.
    프런트가 이 목록을 검색창 위에 칩으로 보여주면, 그 의원 발언에 없는
    주제를 쳐서 무관한 발언(질문/진행 멘트 포함)까지 끌려오는 문제를
    사용자가 미리 피할 수 있다."""
    client = get_client()
    try:
        results = analyze.rank_topics(member, client)
    except ValueError:
        raise HTTPException(status_code=404, detail=_NOT_EMBEDDED_MSG)

    ranked = sorted(results.items(), key=lambda kv: -kv[1][0])
    return [{'topic': t, 'query': analyze.TOPICS[t], 'total_hits': total} for t, (total, _) in ranked]


@app.get('/api/search')
def api_search(member: str, topic: str, top: int = Query(10, gt=0, le=100)):
    client = get_client()
    try:
        hits, total_hits = analyze.search(member, topic, top, client)
    except ValueError:
        raise HTTPException(status_code=404, detail=_NOT_EMBEDDED_MSG)

    hits = analyze._label_period(hits)

    # 세로축에 유사도 대신 찬성/반대/중립을 그리려면(박사님 화면가이드 리뷰 —
    # "제목은 일관성 분석인데 축은 유사도라 입장이 바뀌었는지 안 보인다") 발언마다
    # 입장 분류가 있어야 해서 classify_stances를 먼저 돌리고, 그 결과를 analyze()에도
    # 그대로 넘겨서 같은 분류를 두 번 GPT 호출하지 않는다.
    stances = analyze.classify_stances(member, topic, hits, client) if hits else []
    for h, s in zip(hits, stances):
        h['stance'] = s['stance']
        h['stance_confidence'] = s['confidence']

    analysis = analyze.analyze(member, topic, hits, client, stances=stances) if hits else None

    return {
        'member': member,
        'topic': topic,
        'total_hits': total_hits,
        'hits': hits,
        'analysis': analysis,
    }


class ChatTurn(BaseModel):
    role: str   # 'user' 또는 'assistant'
    content: str


class ChatRequest(BaseModel):
    member: str
    question: str
    history: list[ChatTurn] = []
    top: int = 30


@app.post('/api/chat')
def api_chat(req: ChatRequest, request: Request):
    """자유 질문 챗봇. /api/search와 동일하게 classify_stances로 먼저 걸러낸
    뒤 답변을 생성한다 (정확도 우선 — 비용은 따지지 않기로 함).
    history를 받아 "근거들 전부 보여줘" 같은 후속 질문도 대화 맥락으로 이해하고
    검색하게 한다 (GET 쿼리로는 history 배열을 넘기기 번거로워 POST로 바꿈).

    IP당 시간당 요청 수를 제한한다 — 챗봇 하나가 GPT 호출을 여러 번 체이닝하는
    엔드포인트라 다른 GET 엔드포인트보다 비용 노출이 크다 (2026-09-22)."""
    client_ip = request.client.host if request.client else 'unknown'
    if not _check_chat_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f'요청이 너무 많습니다 (시간당 최대 {_CHAT_RATE_LIMIT}회). 잠시 후 다시 시도해주세요.',
        )

    client = get_client()
    try:
        answer, hits, search_query = analyze.chat_answer(
            req.member, req.question, client,
            history=[t.model_dump() for t in req.history], top_k=req.top,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail=_NOT_EMBEDDED_MSG)

    return {
        'member': req.member,
        'question': req.question,
        'search_query': search_query,
        'answer': answer,
        'hits': [
            {
                'speech_id': h['speech_id'], 'conf_dt': h['conf_dt'], 'conf_type': h['conf_type'],
                'comm_name': h['comm_name'], 'similarity': h['similarity'], 'period': h.get('period', ''),
            }
            for h in hits
        ],
    }


@app.get('/api/profile')
def api_profile(member: str, topics: int = Query(3, gt=0, le=12)):
    client = get_client()
    try:
        return analyze.build_profile(member, topics, client)
    except ValueError:
        raise HTTPException(status_code=404, detail=_NOT_EMBEDDED_MSG)


@app.get('/api/profile/stream')
def api_profile_stream(member: str, topics: int = Query(3, gt=0, le=12)):
    """/api/profile과 같은 계산을 하되, 주제 하나가 끝날 때마다 바로 흘려보낸다.
    프로필 하나에 GPT 호출이 여러 번 겹쳐서(주제당 2번) 수십 초씩 걸리는데,
    끝날 때까지 아무 화면도 안 보이는 것보다 하나씩 채워지는 게 낫다는
    박사님 화면가이드 리뷰 반영. GET 쿼리라 브라우저 EventSource로 바로 받는다."""
    client = get_client()

    def gen():
        try:
            had_data = False
            for event in analyze.build_profile_stream(member, topics, client):
                had_data = True
                yield f'data: {json.dumps(event, ensure_ascii=False)}\n\n'
            if not had_data:
                yield f'data: {json.dumps({"type": "empty"}, ensure_ascii=False)}\n\n'
        except ValueError:
            yield f'data: {json.dumps({"type": "error", "detail": _NOT_EMBEDDED_MSG}, ensure_ascii=False)}\n\n'
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(gen(), media_type='text/event-stream')


@app.get('/api/speech/{speech_id}')
def api_speech(speech_id: str):
    """검색 결과에 나온 발췌문이 진짜 원문과 맞는지 확인하러 가는 길.
    박사님 리뷰: "근거가 인용문 한 줄이라 요약이 맞는지 확인할 길이 없다" 지적에
    대한 답 — 프런트에서 발췌문을 누르면 이 엔드포인트로 전체 발언을 보여준다.

    2026-10-27, 박사님 피드백: 발언 하나를 통째로 보여줘도 "무슨 질문에 대한
    답인지, 바로 다음에 뭐라고 답했는지" 같은 대화 흐름(맥락)은 여전히 안 보임
    — 완전한 회의록 전체를 읽는 건 불가능하다고 보고 포기했지만, 그 타협치고는
    너무 약하다고 판단해서 바로 앞/뒤 발언 1개씩을 같이 내려준다. speeches.id가
    "{conf_id}:{seq}" 형식이고 같은 회의 안에서 seq가 1씩 증가하므로, 그냥
    seq±1을 조회하면 됨 (member_speeches 뷰는 party가 없는 발언자(위원장 진행,
    장관 답변 등)를 걸러내서 앞뒤 발언엔 못 씀 — speeches 원본 테이블을 쓴다)."""
    con = sqlite3.connect(analyze.DB_PATH)
    row = con.execute('''
        SELECT id, conf_id, seq, conf_type, conf_dt, comm_name, name, role, party, content
        FROM speeches WHERE id = ?
    ''', (speech_id,)).fetchone()

    if not row:
        con.close()
        raise HTTPException(status_code=404, detail='해당 발언 없음')

    keys = ['id', 'conf_id', 'seq', 'conf_type', 'conf_dt', 'comm_name', 'name', 'role', 'party', 'content']
    data = dict(zip(keys, row))

    def _neighbor(offset):
        nid = f"{data['conf_id']}:{data['seq'] + offset}"
        r = con.execute('SELECT id, name, role, content FROM speeches WHERE id = ?', (nid,)).fetchone()
        return dict(zip(['id', 'name', 'role', 'content'], r)) if r else None

    data['prev'] = _neighbor(-1)
    data['next'] = _neighbor(1)
    con.close()
    return data


_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if os.path.isdir(_STATIC_DIR):
    app.mount('/', StaticFiles(directory=_STATIC_DIR, html=True), name='static')
