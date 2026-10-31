# 발언 검색 + 일관성 분석 + 의원 프로필 + 시각화
#
# 서브커맨드 3개:
#   py analyze.py search  --member 박찬대 --topic 탄핵 [--csv]
#   py analyze.py profile --member 송석준 [--topics 3]
#   py analyze.py chart   --member 윤상현 --topic 탄핵 [--save]
#
# search : 주제 관련 발언 검색 + LLM 일관성 분석 (초반/중반/후반 입장 변화, 모순, 점수)
# profile: 의원이 주로 다루는 주제 top N + 주제별 입장 요약 (시계열/점수화 없이)
# chart  : search 결과를 시계열 유사도 + 일관성 점수 게이지 + 구간별 발언 수 차트로 출력
#
# 의존: pip install openai numpy matplotlib
#
# chromadb 대신 numpy 브루트포스 코사인 유사도를 씀 (embed.py 참고 —
# chromadb 1.x가 이 환경에서 HNSW 인덱스가 깨지는 버그가 있어서 뺌).

import argparse, functools, hashlib, json, os, sqlite3, sys
from collections import Counter
from datetime import datetime
import numpy as np

def _find_root():
    """코드는 정민규3/ 안에 있는데 .env·output/은 그 상위 폴더에 있는 상태라,
    둘 중 실제로 .env나 output이 있는 쪽을 프로젝트 루트로 쓴다."""
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, '.env')) or os.path.exists(os.path.join(here, 'output')):
        return here
    parent = os.path.dirname(here)
    if os.path.exists(os.path.join(parent, '.env')) or os.path.exists(os.path.join(parent, 'output')):
        return parent
    return here


HERE = _find_root()


def load_env():
    env = os.path.join(HERE, '.env')
    if os.path.exists(env):
        for line in open(env, encoding='utf-8'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()

DB_PATH = os.path.join(HERE, 'output', 'speeches.db')

# 유사도 이 미만인 청크는 검색 결과에서 제외. 0.3~0.4 구간에 주제와 무관한
# 발언(예: 연금개혁 검색에 일반 예산 질문이 섞임)이 자주 걸려서 0.4로 테스트 중.
SIM_THRESHOLD = 0.35


def get_client():
    """API 서버(api.py)에서도 그대로 재사용할 수 있도록 sys.exit 대신 예외를 던진다.
    CLI에서는 각 cmd_* 함수가 이 예외를 잡아서 지금까지와 같은 방식으로 종료한다."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY 없음 — .env에 추가하세요.')
    from openai import OpenAI
    return OpenAI(api_key=api_key)


# ═══════════════════════════════════════════════════════════════════════════
# GPT 호출 캐시 (2026-10-02) — classify_stances/analyze/stance_summary는 같은
# (의원, 주제) 조합이 검색·프로필·챗봇 여러 경로에서 반복 호출된다. 매번 새로
# GPT를 부르면 비용·응답 지연이 그대로 쌓이는데, temperature=0이라 입력(의원·
# 주제·검색된 발언 내용)이 같으면 결과도 항상 같아야 하므로 캐싱 대상으로
# 적합하다. 프로세스 재시작해도 유지되게 SQLite에 저장(speeches.db에 테이블만
# 추가, 원본 데이터와 분리). 내부 로직을 고쳐서 같은 입력에도 다른 결과를 내야
# 할 땐 _CACHE_VERSION을 올리면 예전 캐시가 자동으로 무시된다.
# ═══════════════════════════════════════════════════════════════════════════

_CACHE_VERSION = 1


def _init_cache_table(con):
    con.execute('''
        CREATE TABLE IF NOT EXISTS gpt_cache (
            key TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')


def _cache_key(*parts):
    raw = f'v{_CACHE_VERSION}|' + '|'.join(str(p) for p in parts)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _cache_get(key):
    con = sqlite3.connect(DB_PATH)
    _init_cache_table(con)
    row = con.execute('SELECT payload FROM gpt_cache WHERE key=?', (key,)).fetchone()
    con.close()
    return json.loads(row[0]) if row else None


def _cache_set(key, value):
    con = sqlite3.connect(DB_PATH)
    _init_cache_table(con)
    con.execute(
        'INSERT OR REPLACE INTO gpt_cache (key, payload, created_at) VALUES (?, ?, ?)',
        (key, json.dumps(value, ensure_ascii=False), datetime.now().isoformat()),
    )
    con.commit()
    con.close()


def _hits_signature(hits):
    """캐시 키 재료. classify_stances/analyze 프롬프트에 실제로 들어가는 것과
    똑같이 (날짜, 발췌문 600자)만 본다 — 검색 결과 하나라도 바뀌면(재임베딩,
    SIM_THRESHOLD 변경 등) 자연히 다른 키가 돼서 캐시가 틀리게 재사용되는 일이
    없다."""
    parts = []
    for h in hits:
        text = (h.get('excerpt') or h.get('content') or '')[:600]
        parts.append(f"{h.get('conf_dt', '')}:{text}")
    return '||'.join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# 검색 (search.py의 핵심 — profile/chart도 이 함수를 씀)
# ═══════════════════════════════════════════════════════════════════════════

def unpack(blob):
    return np.frombuffer(blob, dtype=np.float32)


def query_embed(text, client):
    resp = client.embeddings.create(
        model='text-embedding-3-small',
        input=[text],
    )
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def _balanced_top_sids(speech_best, speech_meta, top_k):
    """유사도 top_k만 그대로 뽑으면 그 주제에서 제일 무난하고 반복적인 발언들로
    쏠린다 (박사님 리뷰 지적). 초반/중반/후반 시기별로 나눠서 뽑아 특정 시기로
    쏠리는 걸 줄인다."""
    top_k = max(0, top_k)
    sids = list(speech_best.keys())
    if len(sids) <= top_k:
        return sids

    dates = [speech_meta[s]['conf_dt'] for s in sids if speech_meta[s]['conf_dt']]
    if not dates:
        return sorted(sids, key=lambda s: speech_best[s], reverse=True)[:top_k]

    dates_sorted = sorted(dates)
    fmt = '%Y-%m-%d'
    d0 = datetime.strptime(dates_sorted[0][:10], fmt)
    d1 = datetime.strptime(dates_sorted[-1][:10], fmt)
    span = (d1 - d0).days or 1

    buckets = {'초반': [], '중반': [], '후반': []}
    for s in sids:
        conf_dt = speech_meta[s]['conf_dt']
        try:
            d = datetime.strptime(conf_dt[:10], fmt)
            ratio = (d - d0).days / span
        except Exception:
            ratio = 0.0
        label = '초반' if ratio < 0.33 else ('중반' if ratio < 0.66 else '후반')
        buckets[label].append(s)

    for bucket in buckets.values():
        bucket.sort(key=lambda s: speech_best[s], reverse=True)

    quota, remainder = divmod(top_k, 3)
    quotas = {'초반': quota, '중반': quota, '후반': quota}
    for label in list(quotas)[:remainder]:
        quotas[label] += 1

    picked, leftover = [], []
    for label, bucket in buckets.items():
        picked.extend(bucket[:quotas[label]])
        leftover.extend(bucket[quotas[label]:])

    if len(picked) < top_k:
        leftover.sort(key=lambda s: speech_best[s], reverse=True)
        picked.extend(leftover[:top_k - len(picked)])

    return picked[:top_k]


def search(member, topic, top_k, client):
    """해당 의원의 임베딩 청크 전체를 불러와 브루트포스 코사인 유사도로 검색
    → speech_id로 원문 발언 조회"""
    con = sqlite3.connect(DB_PATH)
    rows = con.execute('''
        SELECT c.id, c.speech_id, c.conf_type, c.conf_dt,
               COALESCE(c.comm_name, ''), COALESCE(ms.party, ''),
               c.chunk_seq, e.embedding
        FROM chunks c
        JOIN embeddings e ON e.chunk_id = c.id
        JOIN member_speeches ms ON c.speech_id = ms.id
        WHERE c.name = ?
    ''', (member,)).fetchall()

    if not rows:
        con.close()
        raise ValueError(f'{member} 임베딩 없음 — py embed.py --member {member} 먼저 실행하세요.')

    q_emb = query_embed(topic, client)
    q_norm = q_emb / np.linalg.norm(q_emb)

    mat = np.stack([unpack(r[7]) for r in rows])          # (N, 1536)
    mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    sims = mat_norm @ q_norm                              # (N,) 코사인 유사도

    # 청크별 유사도 → speech_id별 최고 유사도 청크만 보존
    speech_best      = {}   # speech_id → max similarity
    speech_meta      = {}   # speech_id → (conf_type, conf_dt, comm_name, party)
    speech_best_seq  = {}   # speech_id → 가장 유사도 높은 청크의 chunk_seq
    filtered = 0
    for r, sim in zip(rows, sims):
        sim = round(float(sim), 4)
        if sim < SIM_THRESHOLD:
            filtered += 1
            continue
        sid = r[1]
        if sid not in speech_best or sim > speech_best[sid]:
            speech_best[sid] = sim
            speech_meta[sid] = {'conf_type': r[2], 'conf_dt': r[3], 'comm_name': r[4], 'party': r[5]}
            speech_best_seq[sid] = r[6]

    if filtered:
        print(f'  (유사도 {SIM_THRESHOLD} 미만 청크 {filtered}개 제외)')

    if not speech_best:
        con.close()
        return [], 0

    top_sids = _balanced_top_sids(speech_best, speech_meta, top_k)

    # speeches.db에서 원문 조회 + 매칭된 청크 기준 발췌문(excerpt) 재구성
    placeholders = ','.join('?' * len(top_sids))
    speech_rows = con.execute(f'''
        SELECT id, conf_type, conf_dt,
               COALESCE(comm_name, ''), name, content
        FROM member_speeches
        WHERE id IN ({placeholders})
        ORDER BY conf_dt
    ''', top_sids).fetchall()

    hits = []
    for r in speech_rows:
        sid = r[0]
        best_seq = speech_best_seq.get(sid)
        excerpt = r[5]
        if best_seq:
            # 청크는 이제 안 겹치므로, 매칭된 청크 앞뒤로 하나씩 가져와 그대로 이어붙이면
            # 원문의 해당 구간이 그대로 복원됨 (겹침은 저장 시가 아니라 조회 시에만 만듦)
            neighbor_rows = con.execute('''
                SELECT content FROM chunks
                WHERE speech_id=? AND chunk_seq BETWEEN ? AND ?
                ORDER BY chunk_seq
            ''', (sid, best_seq - 1, best_seq + 1)).fetchall()
            if neighbor_rows:
                excerpt = ''.join(nr[0] for nr in neighbor_rows)
        hits.append({
            'speech_id': sid,
            'conf_dt':   r[2],
            'conf_type': r[1],
            'comm_name': r[3],
            'name':      r[4],
            'content':   r[5],       # 발언 원문 전체 (CSV 내보내기 등에 사용)
            'excerpt':   excerpt,    # 매칭된 청크 중심으로 재구성한 발췌문 (분석 프롬프트에 사용)
            'similarity': speech_best[sid],
            'party':     speech_meta[sid].get('party', ''),
        })
    con.close()

    print(f'  청크 검색 → {len(speech_best)}개 발언 히트, 상위 {len(hits)}건 원문 반환')
    return hits, len(speech_best)


def _label_period(hits):
    """발언 목록에 시점 레이블 부착 (초반/중반/후반)"""
    if not hits:
        return hits
    dates = [h['conf_dt'] for h in hits if h['conf_dt']]
    if not dates:
        return hits
    dates_sorted = sorted(dates)
    start, end = dates_sorted[0], dates_sorted[-1]
    fmt = '%Y-%m-%d'
    try:
        d0 = datetime.strptime(start[:10], fmt)
        d1 = datetime.strptime(end[:10], fmt)
        span = (d1 - d0).days or 1
        for h in hits:
            d = datetime.strptime(h['conf_dt'][:10], fmt)
            ratio = (d - d0).days / span
            h['period'] = '초반' if ratio < 0.33 else ('중반' if ratio < 0.66 else '후반')
    except Exception:
        for h in hits:
            h['period'] = ''
    return hits


# 감점 항목 고정 배점표. 예전엔 GPT가 서술과 점수·감점 내역을 같은 자유 텍스트
# 안에서 동시에 쓰다 보니 산수가 어긋나는 경우가 있어서(_recompute_score로 정규식
# 파싱해 재계산하는 땜질을 했었는데, GPT가 형식을 조금만 벗어나도 그 안전망 자체가
# 조용히 무력화됨 — 박사님 지적으로 재현, 2026-09-22), 점수 계산을 GPT 손에서
# 아예 떼어냈다. GPT는 각 항목이 "해당하는지(true/false)"만 구조화 출력으로
# 판단하고, 배점은 이 표에 고정된 값을 코드가 직접 적용한다 — GPT가 점수를 스스로
# 정하지 않으니 산수가 어긋날 수가 없다.
_DEDUCTION_CATEGORIES = [
    ('reversal', '모순 발언에 실제로 적힌 두 발언 사이에서 입장이 정반대로 바뀜', 40),
    ('weakened', '강하게 주장하다 슬그머니 입장이 약화됨', 15),
    ('evasive', '입장을 밝혀야 할 상황에서 회피하거나 모호하게 답함', 10),
]

_DEDUCTION_SCHEMA = {
    'type': 'object',
    'properties': {
        'deductions': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'category': {'type': 'string', 'enum': [cid for cid, _, _ in _DEDUCTION_CATEGORIES]},
                    'applies':  {'type': 'boolean'},
                    'reason':   {'type': 'string'},
                },
                'required': ['category', 'applies', 'reason'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['deductions'],
    'additionalProperties': False,
}


def _judge_deductions(narrative_text, stance_hits_count, client):
    """analyze()가 방금 쓴 서술(특히 "모순 발언" 항목)을 보고, 감점 항목별로 실제
    해당하는지를 구조화된 출력으로 판정한다. "발언 3건 미만" 항목은 개수만 세면
    되는 기계적 조건이라 GPT에게 묻지 않고 코드가 바로 적용한다 (예전엔 GPT가
    종종 빼먹어서 발언 1건뿐인데 100/100을 준 사례가 있었음)."""
    categories_desc = '\n'.join(f'- {cid}: {desc}' for cid, desc, _ in _DEDUCTION_CATEGORIES)
    prompt = f"""다음은 한 국회의원의 발언 일관성 분석 결과입니다.

{narrative_text}

위 분석 내용을 보고, 아래 감점 항목 각각이 실제로 해당하는지 판단하세요
(모든 항목에 대해 판단할 것 — 해당 안 되면 applies:false, reason엔 왜 해당
안 되는지 짧게 적으세요):

{categories_desc}

**중요**: "reversal"(정반대로 바뀜)은 위 분석의 "모순 발언"란에 실제로 구체적인
두 발언이 적혀 있을 때만 applies:true로 하세요. "모순 발언"이 "없음"이라고
써 있으면 반드시 applies:false입니다."""

    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0,
        response_format={
            'type': 'json_schema',
            'json_schema': {'name': 'deduction_judgement', 'schema': _DEDUCTION_SCHEMA, 'strict': True},
        },
    )
    data = json.loads(resp.choices[0].message.content)
    by_cat = {d['category']: d for d in data['deductions']}

    lines = []
    total = 0
    for cid, desc, pts in _DEDUCTION_CATEGORIES:
        d = by_cat.get(cid)
        if d and d['applies']:
            total += pts
            lines.append(f'- {desc} (-{pts}점): {d["reason"]}')

    if stance_hits_count is not None and stance_hits_count < 3:
        total += 10
        lines.append(
            f'- 근거 발언 3건 미만으로 분석 불충분 (-10점): 근거 발언이 총 '
            f'{stance_hits_count}건뿐이라 판단 근거가 부족함'
        )

    score = max(0, 100 - total)
    deduction_text = '\n'.join(lines) if lines else '감점 없음'
    return score, deduction_text


def analyze(member, topic, hits, client, stances=None):
    """LLM으로 입장 일관성 분석 (시점 레이블 + 감점 기준 명시).

    찬성/반대로 실제 입장이 드러난 발언만 근거로 삼는다 — 검색된 발언을 그냥
    다 넣고 GPT한테 "입장"을 뽑아내라고 시키면 질문·진행 멘트까지 입장으로
    둔갑하는 문제(박사님 리뷰 5번)가 여기서도 났음. stance_summary와 같은
    방식으로 classify_stances 결과로 걸러낸다.
    stances를 안 주면(CLI에서 바로 부를 때) 여기서 직접 분류하고, api.py처럼
    호출부가 이미 분류해뒀으면 그 결과를 받아써서 GPT 호출을 중복하지 않는다.

    2026-10-02: 최종 서술 + 점수까지 전부 캐싱 (hits 내용이 같으면 stances를
    api.py가 넘겨주는지 여부와 무관하게 결과가 같아야 하므로, 캐시 키는
    member/topic/hits만 본다)."""
    cache_key = _cache_key('analyze', member, topic, _hits_signature(hits))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    hits = _label_period(hits)
    if stances is None:
        stances = classify_stances(member, topic, hits, client)

    stance_hits = [(h, s) for h, s in zip(hits, stances) if s['relevant'] and s['stance'] in ('찬성', '반대')]
    excluded = len(hits) - len(stance_hits)

    if not stance_hits:
        no_evidence = _no_evidence_clause(member, f'"{topic}"', len(hits))
        result = (
            f'**전반적 입장**: {no_evidence}\n\n'
            f'**시점별 입장 변화**: 판단 불가\n\n'
            f'**근거 발언 상세**: 없음\n\n'
            f'**모순 발언**: 판단 불가\n\n'
            f'**일관성 점수**: 산정 불가\n'
            f'**감점 내역**: 근거 발언 없음'
        )
        _cache_set(cache_key, result)
        return result

    speech_block = ''
    for h, s in stance_hits:
        period = f"[{h.get('period', '')}] " if h.get('period') else ''
        text = h.get('excerpt') or h['content']
        speech_block += (
            f"{period}{h['conf_dt']} ({h['conf_type']} {h['comm_name']}) [{s['stance']}, 확신도 {s['confidence']:.2f}]\n"
            f"{text[:1500]}{'...' if len(text) > 1500 else ''}\n\n"
        )

    prompt = f"""다음은 국회의원 {member}의 "{topic}" 관련 발언 중, 찬성/반대 입장이 실제로 드러난 것만
추린 {len(stance_hits)}건입니다 (시간순, 초반/중반/후반 레이블 포함. 전체 {len(hits)}건 중
질문·진행 멘트 등 {excluded}건은 제외).

{speech_block}

## 분석 기준
일관성이란 "정치적 유불리나 시점에 상관없이 같은 입장을 고수하는 것"이다.
아래 발언 전체를 시간순으로 놓고, 같은 세부 쟁점에 대해 입장이 바뀐 지점이 있는지
확인하라. 초반/중반/후반 레이블은 아래 "시점별 입장 변화"를 서술할 때 쓰는 구간
표시일 뿐, 모순 탐지 범위를 "초반 vs 후반"으로 한정하는 게 아니다 — 같은 구간
안의 두 발언 사이에서 입장이 바뀌었을 수도 있고, 중반이 낀 세 지점 사이에서
바뀌었을 수도 있으니 특정 구간 조합으로만 좁혀 보지 마라.

**모순으로 인정하는 기준 (중요)**: "{topic}"이라는 큰 주제가 같다는 이유만으로
모순이라고 하지 마라. 두 발언이 **같은 세부 쟁점**(같은 법안 조항, 같은 정책
수단, 같은 질문에 대한 답)에 대해 정반대 입장을 밝힌 경우에만 모순으로 인정한다.
예를 들어 "재정신청권을 수사기관에 주는 것에 반대"와 "수사기관은 신속히 수사해야
한다"는 서로 다른 세부 쟁점(권한 부여 여부 vs 속도)이라 모순이 아니다. 두 발언이
정확히 같은 세부 쟁점을 다루는지 먼저 확인하고, 확신이 없으면 "없음"으로 처리하라.

## 출력 형식 (반드시 이 형식 그대로 — 일관성 점수·감점 내역은 여기서 쓰지 마세요, 따로 계산합니다)

**전반적 입장**: (3~5문장. "찬성/반대한다"는 결론만 말하지 말고, 어떤 조건·맥락에서
그런 입장을 보였는지, 어느 정도 강도로 표현했는지(단정적/유보적 등)까지 구체적으로 서술)

**시점별 입장 변화**:
- 초반: (2~3문장 요약 + 날짜와 함께 근거 발언 최소 1개 인용)
- 중반: (위와 동일. 발언 없으면 "발언 없음")
- 후반: (위와 동일)

**근거 발언 상세**: (분석에 쓴 발언을 날짜순으로 전부 나열. 각 항목마다
"YYYY-MM-DD [찬성/반대]: "인용문" — 이 발언이 왜 그 입장을 보여주는지 한 줄 설명" 형식)

**모순 발언**: (위 "모순으로 인정하는 기준"을 통과한 경우에만 작성. 두 발언 각각
날짜·맥락과 함께 인용하고, 두 발언이 다루는 세부 쟁점이 동일함을 먼저 밝힌 뒤 왜
모순인지 설명. 세부 쟁점이 다르면 "없음"이라고 쓰고 이유는 적지 마라)

발언 내용만 근거로 분석하고, 추측은 하지 마세요."""

    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': _SYSTEM_PERSONA},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0,
    )
    narrative = resp.choices[0].message.content
    score, deduction_text = _judge_deductions(narrative, len(stance_hits), client)
    result = f'{narrative}\n\n**일관성 점수**: {score}/100\n**감점 내역**:\n{deduction_text}'
    _cache_set(cache_key, result)
    return result


def _rewrite_followup_query(question, history, client):
    """"근거들 전부 보여줘", "그거 더 자세히" 같은 후속 질문은 그 문장 그대로
    임베딩 검색을 돌리면 전혀 엉뚱한 게 걸린다 (챗봇이 대화 맥락을 기억 못 해서
    생긴 문제). 이전 대화를 참고해서 검색 가능한 독립 문장으로 다시 쓰는 단계를
    앞에 붙인다. history가 없으면 원래 질문 그대로 반환."""
    if not history:
        return question

    hist_text = '\n'.join(
        f"{'사용자' if h['role'] == 'user' else '챗봇'}: {h['content']}" for h in history[-6:]
    )
    prompt = f"""다음은 이전 대화입니다.

{hist_text}

새 질문: "{question}"

새 질문이 "그거 더 자세히", "근거 다 보여줘"처럼 이전 대화 맥락에 기대는 후속 질문이면,
대화 내용을 참고해서 검색에 바로 쓸 수 있는 완전한 독립 문장으로 다시 쓰세요.
이미 그 자체로 독립적인 질문이면 고치지 말고 그대로 반환하세요.
설명 없이 검색에 쓸 문장 하나만 출력하세요."""

    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip().strip('"')


# 인트로/오프토픽 안내에 공통으로 쓰는 질문 예시 3종 — 유형별로 하나씩 골라서
# "이렇게 물어보면 잘 찾는다"는 감을 준다 (그냥 예시 하나만 있던 걸 확장, 2026-09-18).
_EXAMPLE_QUESTIONS = [
    '"최저임금 인상에 대해 어떻게 생각해?" — 특정 정책에 대한 입장',
    '"예전이랑 지금이랑 입장이 달라졌어?" — 시점별 입장 변화',
    '"그 근거가 된 발언 원문 보여줘" — 근거 발언 확인',
]
_EXAMPLE_BLOCK = '\n'.join(f'- {q}' for q in _EXAMPLE_QUESTIONS)


_BOT_INTRO_TMPL = (
    '저는 국회의원들의 실제 회의록 발언을 검색해서 답해드리는 챗봇이에요. '
    '지금 {member} 의원을 선택하고 계시니, {member} 의원이 특정 주제에 대해 '
    '실제로 어떻게 말했는지, 시점에 따라 입장이 바뀌었는지 자유롭게 물어보세요.\n\n'
    '예를 들어 이렇게 물어보시면 좋아요:\n' + _EXAMPLE_BLOCK + '\n\n'
    '발언에 없는 내용은 지어내지 않고, 관련 근거를 못 찾으면 못 찾았다고 말씀드려요.'
)


_OFFTOPIC_REPLY_TMPL = (
    '그건 제가 답하기 어려운 질문이에요 — 저는 {member} 의원이 실제 회의록에서 '
    '한 발언만 근거로 답하는 챗봇이라, 그 범위를 벗어나는 건 다루지 않아요.\n\n'
    '예를 들어 이렇게 물어보시면 좋아요:\n' + _EXAMPLE_BLOCK
)


_QUESTION_TYPE_SCHEMA = {
    'type': 'object',
    'properties': {
        'type': {'type': 'string', 'enum': ['meta', 'offtopic', 'broad', 'specific']},
    },
    'required': ['type'],
    'additionalProperties': False,
}


@functools.lru_cache(maxsize=256)
def _classify_question_type(question, client):
    """질문을 meta/offtopic/broad/specific 네 유형 중 정확히 하나로 분류.

    원래는 _is_meta_question / _is_offtopic_question / _is_broad_question 세
    개를 독립된 이진 분류로 따로 호출했는데, 라이브 테스트에서 "오늘 날씨
    어때?"가 meta에도 true, offtopic에도 true로 동시에 걸리는 사례가 나왔음
    (2026-09-22) — 서로 다른 프롬프트라 겹치지 않는다는 보장이 없었던 것.
    체이닝 순서상 먼저 검사하는 meta가 이겨서 "오늘 날씨 어때?"에 자기소개
    문구가 나가는 오답이 발생했다. 하나의 Structured Outputs 호출에서
    상호배타적인 enum 값 중 하나만 고르게 하면 이런 충돌이 구조적으로
    불가능해진다 — 겹치는 프롬프트 두 개를 각각 손보는 것보다 근본적인 수정.
    호출 3번을 1번으로 줄이는 효과도 있어서 챗봇 비용·응답 지연에도 도움됨.

    temperature=0이라 같은 질문엔 항상 같은 답이 나오므로 lru_cache로 재사용한다."""
    prompt = f"""다음 질문을 아래 네 유형 중 정확히 하나로 분류하세요.

- meta: 챗봇 자기 자신(정체성·기능·사용법)에 대해 묻거나 챗봇에게 건네는 인사말
  (예: "너 뭐 하는 애야?", "안녕", "여기서 뭘 물어볼 수 있어?")
- offtopic: 국회의원의 발언·입장과도, 챗봇 자신과도 무관한 일반 잡담이나 질문
  (예: "오늘 날씨 어때?", "점심 뭐 먹지", 정치와 무관한 잡학 지식)
- broad: 특정 하나의 구체적 정책 이슈가 아니라 여러 주제를 통틀어 "관심 있는
  주제/테마가 뭔지", "전반적인 입장이 뭔지" 묻는 포괄적인 질문
- specific: 국회의원의 특정 정책·발언·입장에 대해 구체적으로 묻는 질문
  (표현이 막연해도 국회의원 발언과 조금이라도 관련 있으면 이쪽. 후속 질문·
  근거 요청도 포함)

질문: "{question}\""""
    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0,
        response_format={
            'type': 'json_schema',
            'json_schema': {'name': 'question_type', 'schema': _QUESTION_TYPE_SCHEMA, 'strict': True},
        },
    )
    return json.loads(resp.choices[0].message.content)['type']


def _answer_broad_profile(member, client, top_n=3):
    """"관심 테마 전체"를 묻는 포괄적 질문에 대한 답. 검색 한 번으로는 답이
    안 나오니, 프로필 기능(rank_topics + stance_summary)을 그대로 재사용해서
    실제 히트 있는 주제 상위 top_n개를 찾고 주제별로 입장을 정리해 합친다."""
    results = rank_topics(member, client)
    if not results:
        return f'{member} 의원은 준비된 12개 주제 중 우리 자료에서 검색되는 발언이 없습니다.', []

    ranked = sorted(results.items(), key=lambda kv: -kv[1][0])[:top_n]
    sections = []
    all_hits = []
    for topic, (total_hits, hits) in ranked:
        summary = stance_summary(member, topic, hits, client)
        sections.append(f'**{topic}** ({total_hits}건 히트)\n{summary}')
        all_hits.extend(hits)

    header = f'{member} 의원 발언 중 우리 자료에서 히트가 많은 주제 상위 {len(ranked)}개를 기준으로 정리했습니다.\n\n'
    return header + '\n\n'.join(sections), all_hits


def chat_answer(member, question, client, history=None, top_k=30):
    """챗봇용 자유 질문 답변. 처음엔 비용 절감 위해 "관련성 판단 + 답변"을
    프롬프트 하나로 합쳤었는데(관련 없는 후보를 GPT가 알아서 무시하도록만 지시),
    그렇게 하니 "의원이 남(예: 차관)의 입장을 되묻는 질문"까지 근거로 섞이는 경우가
    있었다 — 검색/분석 경로에서 이미 한 번 잡았던 "질문이 입장으로 둔갑" 문제와
    같은 종류. 비용보다 정확도 우선이라는 지시에 따라 search/analyze와 동일하게
    classify_stances로 먼저 걸러내는 2단계 방식으로 되돌린다.

    history(직전 대화 몇 턴)를 받으면 _rewrite_followup_query로 검색용 질문을
    먼저 다시 쓴다 — 안 그러면 "근거들 전부 보여줘" 같은 후속 질문이 그 문장
    그대로 검색에 들어가서 아무 것도 못 찾는 문제가 있었다.

    질문은 먼저 _classify_question_type으로 meta/offtopic/broad/specific 중
    하나로 분류한다 (2026-09-22, 원래는 세 번의 독립된 이진 분류였는데 서로
    겹칠 수 있어서 하나의 상호배타적 분류로 통합함 — 아래 함수 docstring 참고).

    "너 뭐 하는 애야?" 같은 meta는 검색 없이 바로 소개 문구로 답한다 (안 그러면
    그 문장 그대로 검색해서 아무 발언도 안 걸리니 "근거를 찾지 못했습니다"라는
    엉뚱한 답이 나갔었다).

    "오늘 날씨 어때" 같은 offtopic은 검색 없이 정중하게 범위 밖이라고 안내한다 —
    챗봇이 최대한 정확하게 답하려면, 애초에 답할 수 없는 질문에 억지로 검색을
    돌려서 무관한 근거로 답을 지어내는 것보다, 다룰 수 있는 범위를 명확히
    긋고 그 범위 안에서만 답하는 게 낫다는 판단.

    "관심 테마와 입장을 분석해서 보여줘"처럼 특정 이슈 하나가 아니라 여러
    주제를 통틀어 묻는 broad 질문은 _answer_broad_profile(프로필 기능 재사용)로
    답한다 — 그 문장 그대로 단일 검색을 돌리면 정책 키워드가 없어서 검색이
    거의 안 걸리고 "근거를 찾지 못했습니다"로 잘못 답했던 실제 사례가 있었다."""
    qtype = _classify_question_type(question, client)

    if qtype == 'meta':
        return _BOT_INTRO_TMPL.format(member=member), [], question

    if qtype == 'offtopic':
        return _OFFTOPIC_REPLY_TMPL.format(member=member), [], question

    if qtype == 'broad':
        answer, hits = _answer_broad_profile(member, client)
        return answer, hits, question

    search_query = _rewrite_followup_query(question, history, client)
    hits, total_hits = search(member, search_query, top_k, client)
    if not hits:
        return f'"{question}"과 관련된 {member} 의원의 발언을 찾지 못했습니다.', [], search_query

    hits = _label_period(hits)
    stances = classify_stances(member, search_query, hits, client)
    stance_hits = [(h, s) for h, s in zip(hits, stances) if s['relevant'] and s['stance'] in ('찬성', '반대')]

    if not stance_hits:
        return _no_evidence_clause(member, f'"{question}"', len(hits)), hits, search_query

    speech_block = ''
    for h, s in stance_hits:
        period = f"[{h.get('period', '')}] " if h.get('period') else ''
        text = h.get('excerpt') or h['content']
        speech_block += (
            f"{period}{h['conf_dt']} ({h['conf_type']} {h['comm_name']}) [{s['stance']}, 확신도 {s['confidence']:.2f}]\n"
            f"{text[:1500]}{'...' if len(text) > 1500 else ''}\n\n"
        )

    hist_block = ''
    if history:
        hist_text = '\n'.join(
            f"{'사용자' if h['role'] == 'user' else '챗봇'}: {h['content']}" for h in history[-6:]
        )
        hist_block = f"## 이전 대화 (참고용, 답변 톤/맥락 유지에만 쓰고 근거로는 쓰지 마세요)\n{hist_text}\n\n"

    prompt = f"""{hist_block}다음은 국회의원 {member}의 발언 중, "{question}"이라는 질문에 대해 실제로
찬성/반대 입장이 드러난 것만 추린 {len(stance_hits)}건입니다 (검색된 {len(hits)}건 중
질문·진행 멘트·무관한 내용 {len(hits) - len(stance_hits)}건은 이미 제외됨).

{speech_block}

## 지침
- 이 발언들만 근거로 질문에 직접 답하세요.
- 추측하지 말고 발언에 실제로 나온 내용만 사용하세요.
- 시점에 따라 입장이 달라졌다면(초반 vs 후반) 그것도 언급하세요.

## 출력 형식 (반드시 이 형식 그대로)
**답변**: (질문에 대한 직접적인 답. 2~5문장)
**근거**: (사용한 발언을 날짜순으로. "YYYY-MM-DD [찬성/반대]: "인용문"" 형식)"""

    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': _SYSTEM_PERSONA},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content, [h for h, s in stance_hits], search_query


def export_csv(member, topic, hits, out_dir):
    """유사도 높은 순으로 정렬된 발언 원문을 CSV로 저장"""
    import csv
    hits_sorted = sorted(hits, key=lambda h: h['similarity'], reverse=True)
    safe_topic = topic.replace('/', '_').replace('\\', '_')
    fname = f'{member}_{safe_topic}_발언.csv'
    fpath = os.path.join(out_dir, fname)
    with open(fpath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['순위', '유사도', '시점', '날짜', '회의종류', '위원회', '발언내용'])
        for i, h in enumerate(hits_sorted, 1):
            writer.writerow([
                i,
                h['similarity'],
                h.get('period', ''),
                h['conf_dt'][:10],
                h['conf_type'],
                h['comm_name'],
                h['content'],
            ])
    print(f'CSV 저장됨: {fpath}')
    return fpath


def cmd_search(args):
    try:
        client = get_client()
        print(f'검색 중: {args.member} + "{args.topic}" ...')
        hits, total_hits = search(args.member, args.topic, args.top, client)
    except (RuntimeError, ValueError) as e:
        sys.exit(f'[오류] {e}')
    print(f'검색 결과: {len(hits)}건 (전체 히트 {total_hits}건 중 시기별 균형 추출)\n')

    hits = _label_period(hits)

    for i, h in enumerate(hits, 1):
        print(f'[{i:2}] {h["conf_dt"]} ({h["conf_type"]}) 유사도={h["similarity"]}')
        print(f'     {h["content"][:80]}...')
    print()

    if args.csv:
        out_dir = os.path.join(HERE, 'output')
        os.makedirs(out_dir, exist_ok=True)
        export_csv(args.member, args.topic, hits, out_dir)
        return

    print('=' * 60)
    print(f'[일관성 분석] {args.member} × {args.topic}')
    print('=' * 60)
    analysis = analyze(args.member, args.topic, hits, client)
    print(analysis)


# ═══════════════════════════════════════════════════════════════════════════
# profile: 의원이 주로 다루는 주제 + 주제별 입장 요약
# ═══════════════════════════════════════════════════════════════════════════

# 주제별 검색 쿼리 문장(임베딩 검색용, 단어 하나보다 문장이 잘 잡힘)
#
# 예전엔 "주로 다루는 주제"를 키워드 LIKE 빈도로 먼저 뽑고, 그 top N에 대해서만
# 임베딩 검색을 했었다. 근데 OCR 띄어쓰기가 자주 깨져서("증빙 을", "향 유를" 처럼
# 단어 중간에 공백이 낌) LIKE 매칭이 실제보다 적게 잡히고, 그러다 보니 LIKE 순위와
# 임베딩 검색 히트 순위가 서로 어긋났다 (박사님 리뷰: 이철규 1위가 뒤바뀌고 5명 중
# 4명이 순위가 어긋남). 그래서 지금은 12개 주제 전부 임베딩 검색까지 돌려서 나온
# 히트 수 하나로만 순위를 매긴다 — 기준을 하나로 통일.
TOPICS = {
    '연금개혁':        '국민연금 소득대체율 보험료율 연금개혁',
    '부동산/주택':      '부동산 가격 안정 주택 공급 전세사기 대책',
    '검찰개혁/수사권':   '검찰 수사권 조정 검찰개혁 공수처',
    '노동/최저임금':     '최저임금 인상 노동시간 근로자 권리',
    '저출산/육아':      '저출산 대책 육아휴직 양육비 지원',
    '탄핵':            '대통령 탄핵 소추 헌법재판소',
    '반도체/산업':      '반도체 산업 경쟁력 지원 수출',
    '교육':            '교육과정 개편 사교육비 입시제도',
    '의료/의대증원':    '의대 정원 확대 의료개혁 필수의료',
    '언론/방송법':      '방송법 개정 언론 자유 공정성',
    '기후/에너지':      '탄소중립 기후위기 대응 재생에너지 원전',
    '해병대원 특검':    '해병대원 순직 사건 채상병 특검',
}


# 검색 분석/프로필/챗봇 세 곳(analyze/stance_summary/chat_answer)의 답변 생성
# 호출에 공통으로 얹는 system 메시지. 예전엔 이 내용이 _PERSONA_STYLE_HINT라는
# 한 줄짜리 힌트로 각 프롬프트 끝에 반복해서 붙는 형태였는데, 정체성·톤·확신도
# 처리 규칙이 한 군데에 없어서 answer 세 곳이 미묘하게 다른 어투를 낼 수 있었음
# (2026-09-19, system 역할로 승격).
_SYSTEM_PERSONA = """당신은 '국회발언추적' — 국회의원의 실제 회의록 발언만 근거로 답하는 분석 도구입니다.

## 정체성과 범위
- 검색된 발언 원문만 근거로 삼습니다. 근거에 없는 내용은 절대 추측하거나 지어내지 않습니다.
- 근거가 부족하거나 없으면 "근거가 없다"고 정직하게 말합니다. 없는 걸 있는 것처럼 꾸미지 않습니다.
- 스스로를 언급할 때는 1인칭("제가 찾은 근거로는" 등)을 쓰고, 분석 대상 의원은 항상 3인칭으로 서술합니다.

## 확신도 처리
- 근거 발언 옆에 확신도(예: "확신도 0.42")가 표시돼 있으면, 0.6 미만인 근거는
  "~라고 밝혔다"처럼 단정하지 말고 "~로 보인다", "~라는 취지로 말한 것으로 해석된다"처럼
  한 단계 낮춰서 표현하세요. 0.6 이상은 평소대로 분명하게 서술해도 됩니다.
- 확신도 표시가 없는 발언에는 이 규칙을 적용하지 않습니다.

## 표현 방식
의원이 실제로 쓴 인상적인 표현이나 어투가 근거 발언에 있으면, 밋밋하게 패러프레이즈만
하지 말고 그 표현을 그대로 살려서 문장에 자연스럽게 녹이세요 (예: "탄력근로제를
'굉장히 잔인한 법'이라고까지 표현하며 반대했다"). 단, 근거 발언에 없는 표현이나 어투는
절대 지어내지 마세요 — 실제로 한 말만 그대로 살려 쓸 것."""


def _no_evidence_clause(member, topic_desc, hits_count):
    """analyze()/stance_summary()/chat_answer() 세 곳이 각자 손으로 쓰던 거절 문구를
    하나로 통일 (2026-09-19). 세 함수의 출력 형식(마크다운 필드 구성)은 서로 달라서
    전체 반환값을 통일할 순 없지만, 핵심 문장의 어투는 하나로 맞춘다."""
    return (
        f'검색된 발언 {hits_count}건이 전부 질문·진행 멘트이거나 주제와 무관한 내용이라, '
        f'{member} 의원의 {topic_desc}에 대한 입장을 판단할 근거가 없습니다.'
    )


_STANCE_SCHEMA = {
    'type': 'object',
    'properties': {
        'classifications': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'index':      {'type': 'integer'},
                    'relevant':   {'type': 'boolean'},
                    'stance':     {'type': 'string', 'enum': ['찬성', '반대', '중립', '판단유보']},
                    'confidence': {'type': 'number'},
                    'quote':      {'type': 'string'},
                },
                'required': ['index', 'relevant', 'stance', 'confidence', 'quote'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['classifications'],
    'additionalProperties': False,
}


def classify_stances(member, topic, hits, client):
    """발언마다 "{topic}"에 대한 실제 찬반 입장 표명인지, 질문·진행 멘트라 입장이
    없는지 구조화된 출력(Structured Outputs)으로 분류.

    "관리하는 주택?", "감치명령을 받는 경우가 통계적으로 어느 정도 됩니까?" 같은
    질문이 그대로 "~를 강조하고 있다"는 입장으로 둔갑했던 문제(박사님 리뷰) 때문에
    추가한 단계. stance_summary가 이 결과로 찬성/반대만 걸러서 요약 근거로 쓴다.

    relevant 필드는 나중에 추가함 — 검색(임베딩 유사도)이 주제랑 무관한 발언을
    걸러오는 경우(예: "해병대원 특검" 검색에 "병원협회에서 반대지요"처럼 전혀 다른
    안건 얘기가 걸림)에, "반대"라는 단어만 보고 그 주제에 대한 반대로 착각하는
    문제가 있었음. relevant=false면 stance 값과 무관하게 근거에서 제외한다.

    2026-10-02, evaluate.py로 50건 정답 라벨 대조 후 발견: relevant를 "입장이
    드러나는지"와 섞어서 판단하는 경향이 있었음 — 질문 형태로 던진 발언이 실제로는
    그 주제의 구체적 사안(예: "방통위가 채널A 재승인 과정에서 동아일보 기자 파견
    금지시켰지요?")을 다루는데도, 찬반이 안 드러난다는 이유로 relevant까지
    false로 떨어뜨려 recall이 깎이는 사례가 있었다. relevant 판단 기준에서
    "찬반 여부와 무관하다"는 걸 명시해서 분리함.

    2026-10-02: 같은 (의원, 주제, 발언 묶음)이 검색/프로필/챗봇에서 반복
    호출되므로 캐싱함 (temperature=0이라 입력이 같으면 결과도 항상 같음)."""
    cache_key = _cache_key('classify_stances', member, topic, _hits_signature(hits))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    block = ''
    for i, h in enumerate(hits):
        text = h.get('excerpt') or h['content']
        block += f"[{i}] {h['conf_dt'][:10]}: {text[:600]}\n\n"

    prompt = f"""다음은 국회의원 {member}의 "{topic}" 검색 결과로 나온 발언 {len(hits)}건입니다
(임베딩 유사도 검색이라 실제로는 다른 안건 얘기인데 단어가 겹쳐서 잘못 걸린 경우가 있습니다).

각 발언(인덱스 0~{len(hits) - 1})에 대해 먼저 이 발언이 실제로 "{topic}"에 관한 내용인지부터
판단하고, 그다음 찬성/반대 입장을 밝히는지 판단하세요.

{block}

- relevant: 이 발언이 다루는 소재(법안, 정책, 결정, 사건 등)가 "{topic}"에 속하면
  true — 찬반 입장이 드러나는지는 relevant 판단과 무관하다. 질문 형태라도 그 질문이
  "{topic}"에 속하는 구체적 사안을 다루고 있으면 true로 하고, 입장이 없다는 건
  stance를 "판단유보"로 표시하는 걸로 반영하라 (relevant까지 false로 하지 말 것).
  반대로 다른 안건/사안 얘기인데 단어만 비슷해서 검색된 것으로 보이면(예: "반대지요"
  같은 단어만 일치) false
- stance: relevant가 false면 무조건 "판단유보". relevant가 true일 때만:
  "{topic}"에 찬성이면 찬성, 반대면 반대, 입장은 드러나지만 찬반이 불분명하면 중립,
  질문·진행 멘트·사실 확인 등 입장 자체가 없으면 판단유보
- confidence: 0~1 사이 확신도
- quote: 판단 근거가 된 원문 그대로의 짧은 인용

추측하지 말고 발언에 실제로 쓰인 표현만 근거로 삼으세요."""

    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0,
        response_format={
            'type': 'json_schema',
            'json_schema': {'name': 'stance_classification', 'schema': _STANCE_SCHEMA, 'strict': True},
        },
    )
    data = json.loads(resp.choices[0].message.content)
    by_index = {c['index']: c for c in data['classifications']}
    default = {'relevant': False, 'stance': '판단유보', 'confidence': 0.0, 'quote': ''}
    result = [by_index.get(i, default) for i in range(len(hits))]
    _cache_set(cache_key, result)
    return result


def stance_summary(member, topic, hits, client):
    """검색된 발언 근거로 입장 요약 (시계열/점수화 없이 순수 입장 정리).
    질문·진행 멘트는 classify_stances로 먼저 걸러내고 실제 찬반 표명만 근거로 쓴다.

    2026-10-02: 최종 서술까지 포함해서 캐싱(classify_stances는 이미 캐싱되지만,
    이 함수의 GPT 호출 — 서술 생성 — 은 별도라서 한 번 더 감싼다)."""
    cache_key = _cache_key('stance_summary', member, topic, _hits_signature(hits))
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    stances = classify_stances(member, topic, hits, client)
    dist = Counter(s['stance'] for s in stances)
    stance_hits = [(h, s) for h, s in zip(hits, stances) if s['relevant'] and s['stance'] in ('찬성', '반대')]

    if not stance_hits:
        no_evidence = _no_evidence_clause(member, f'"{topic}"', len(hits))
        result = (
            f'**입장 요약**: {no_evidence}\n'
            f'**근거 발언 상세**: 없음\n'
            f"**비고**: 입장 분류 결과 — 찬성 {dist['찬성']}건, 반대 {dist['반대']}건, "
            f"중립 {dist['중립']}건, 판단유보 {dist['판단유보']}건"
        )
        _cache_set(cache_key, result)
        return result

    block = ''
    for h, s in stance_hits:
        text = h.get('excerpt') or h['content']
        block += f"- {h['conf_dt'][:10]} [{s['stance']}, 확신도 {s['confidence']:.2f}] ({h['conf_type']} {h['comm_name']}): {text[:600]}\n\n"

    prompt = f"""다음은 국회의원 {member}의 "{topic}" 관련 발언 중, 찬성/반대 입장이 실제로 드러난 것만 추린 {len(stance_hits)}건입니다
(전체 {len(hits)}건 중 질문·진행 멘트 등 {len(hits) - len(stance_hits)}건은 제외).

{block}

이 발언들만 근거로 {member} 의원이 "{topic}"에 대해 어떤 입장/의견을 가지고 있는지 정리하세요.
추측하지 말고 발언에 실제로 나온 내용만 사용하세요.

## 출력 형식
**입장 요약**: (3~5문장. "찬성/반대한다"는 결론만 말하지 말고, 어떤 조건·맥락에서
그런 입장을 보였는지, 시간에 따라 강도나 표현이 달라졌는지까지 구체적으로 서술)
**근거 발언 상세**: (위 발언을 날짜순으로 전부 나열. 각 항목마다
"YYYY-MM-DD [찬성/반대]: "인용문" — 이 발언이 왜 그 입장을 보여주는지 한 줄 설명" 형식)
**비고**: (발언들 사이 뉘앙스 차이, 조건부 표현, 확신도 차이 등 세부 사항. 없으면 "없음")"""

    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': _SYSTEM_PERSONA},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0,
    )
    result = resp.choices[0].message.content
    _cache_set(cache_key, result)
    return result


def rank_topics(member, client, top_k=10):
    """12개 고정 주제를 임베딩 검색만 돌려서(GPT 호출 없음, 빠름) 히트 수로
    순위를 매기는 부분만 떼어낸 함수. build_profile_stream과 가벼운
    /api/topics("주요 키워드") 둘 다 여기서 재사용한다."""
    results = {}   # topic -> (total_hits, hits)
    for topic, query in TOPICS.items():
        hits, total = search(member, query, top_k, client)
        if total:
            results[topic] = (total, hits)
    return results


def _is_embedded(member):
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        'SELECT 1 FROM chunks c JOIN embeddings e ON e.chunk_id = c.id WHERE c.name = ? LIMIT 1',
        (member,)
    ).fetchone()
    con.close()
    return row is not None


def get_member_bio(member):
    """프로필 화면에 보여줄 실제 인물 정보. GPT 요약만 보여주면 이 사람이
    누군지(정당·지역구·선수) 감이 없어서 members 테이블(열린국회정보 명단)
    기본 정보를 가져오고, 위원장/소위원장류 role은 회의록에서 실제로 어느
    위원회를 진행했는지 보여주는 데 쓴다 (member_speeches.role — comm_name과
    달리 그 회의에서 이 사람이 맡은 직책이 그대로 적혀 있음).
    members 테이블은 대수마다 한 행씩이라(정당 바뀌면 행이 늘어남) 가장 최근
    대수 것을 기본 정보로 쓴다."""
    con = sqlite3.connect(DB_PATH)
    bio_row = con.execute('''
        SELECT party, district, elect_type, reele, units, dae
        FROM members WHERE name = ?
        ORDER BY dae DESC LIMIT 1
    ''', (member,)).fetchone()

    role_rows = con.execute('''
        SELECT DISTINCT role FROM member_speeches
        WHERE name = ? AND role LIKE '%위원장%'
    ''', (member,)).fetchall()
    con.close()

    if not bio_row:
        return None

    return {
        'party': bio_row[0],
        'district': bio_row[1],
        'elect_type': bio_row[2],
        'reele': bio_row[3],
        'units': bio_row[4],
        'dae': bio_row[5],
        'chair_roles': sorted(r[0] for r in role_rows),
    }


def build_profile_stream(member, n_topics, client):
    """프로필을 주제별로 끝나는 대로 하나씩 yield (SSE 스트리밍용, api.py가 씀).

    아직 임베딩이 안 된 의원이면 프로필을 보기 전에 여기서 먼저 임베딩부터
    돌린다 — "검색 준비 전"이라고만 뜨고 CLI로 따로 embed.py를 돌려야 했던
    걸, 프로필 보기 버튼 하나로 되게 함. 임베딩 진행 상황도 'embedding'
    이벤트로 흘려보낸다 (의원에 따라 청크가 수천 개라 시간이 좀 걸림).

    12개 주제 임베딩 검색(빠름)은 먼저 다 끝내서 순위를 정하고, 그다음 상위
    n_topics개에 대한 입장 요약(느림 — 주제당 GPT 호출 2번, classify_stances +
    stance_summary)을 하나씩 끝내는 대로 내보낸다. 박사님 화면가이드 리뷰:
    "오래 걸리는 응답은 흘려보내기" 반영."""
    if not _is_embedded(member):
        import embed
        for progress in embed.embed_member_stream(member, client, DB_PATH):
            yield {'type': 'embedding', **progress}

    results = rank_topics(member, client)
    if not results:
        return

    ranked = sorted(results.items(), key=lambda kv: -kv[1][0])[:n_topics]
    yield {'type': 'ranked', 'topics': [{'topic': t, 'total_hits': total} for t, (total, _) in ranked]}

    for topic, (total, hits) in ranked:
        summary = stance_summary(member, topic, hits, client)
        yield {'type': 'topic', 'topic': topic, 'total_hits': total, 'summary': summary, 'hits': hits}


def build_profile(member, n_topics, client):
    """의원 프로필을 순수 데이터로 계산 (출력은 안 함, 스트리밍 없이 한 번에 반환).
    CLI(cmd_profile)와 API 서버(api.py)의 비-스트리밍 엔드포인트가 공용으로 쓴다."""
    topics_out = []
    got_ranked = False
    for event in build_profile_stream(member, n_topics, client):
        if event['type'] == 'embedding':
            continue
        if event['type'] == 'ranked':
            got_ranked = True
        else:
            topics_out.append({
                'topic': event['topic'],
                'total_hits': event['total_hits'],
                'summary': event['summary'],
                'hits': event['hits'],
            })

    if not got_ranked:
        return {'member': member, 'topics': []}
    return {'member': member, 'topics': topics_out}


def cmd_profile(args):
    try:
        client = get_client()
        data = build_profile(args.member, args.topics, client)
    except (RuntimeError, ValueError) as e:
        sys.exit(f'[오류] {e}')

    print(f'\n{"="*70}')
    print(f'  {args.member} 의원 프로필')
    print(f'{"="*70}')

    if not data['topics']:
        print('  12개 주제 중 매칭되는 발언 없음')
        return

    print(f'\n[주로 발언하는 주제 top {len(data["topics"])}]')
    for t in data['topics']:
        print(f'  - {t["topic"]}: {t["total_hits"]}건 히트')

    for t in data['topics']:
        print(f'\n--- [{t["topic"]}] ({t["total_hits"]}건 히트) ---')
        print(t['summary'])


# ═══════════════════════════════════════════════════════════════════════════
# chart: search + analyze 결과를 시각화
# ═══════════════════════════════════════════════════════════════════════════

def draw_chart(member, topic, hits, analysis_text, save=False):
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if not hits:
        print('[시각화] 발언 없음')
        return

    dates = []
    sims  = []
    for h in hits:
        try:
            dates.append(datetime.strptime(h['conf_dt'][:10], '%Y-%m-%d'))
            sims.append(h['similarity'])
        except Exception:
            pass

    if not dates:
        print('[시각화] 날짜 파싱 실패')
        return

    score = None
    for line in analysis_text.split('\n'):
        if '일관성 점수' in line and '/' in line:
            try:
                score = int(line.split(':')[1].strip().split('/')[0].strip())
            except Exception:
                pass

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])   # 상단: 시계열
    ax2 = fig.add_subplot(gs[1, 0])   # 하단좌: 점수 게이지
    ax3 = fig.add_subplot(gs[1, 1])   # 하단우: 구간별 발언 수

    fig.suptitle(f'{member} × {topic}  발언 일관성 분석', fontsize=14, fontweight='bold')

    ax1.scatter(dates, sims, color='steelblue', s=60, zorder=3)
    ax1.plot(dates, sims, color='steelblue', alpha=0.4, linewidth=1)
    ax1.axhline(SIM_THRESHOLD, color='tomato', linestyle='--', linewidth=0.8, label=f'임계값 {SIM_THRESHOLD}')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('유사도')
    ax1.set_title('시간별 관련 발언 유사도')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

    if score is not None:
        color = '#2ecc71' if score >= 70 else ('#f39c12' if score >= 40 else '#e74c3c')
        ax2.barh([0], [score], color=color, height=0.4)
        ax2.barh([0], [100], color='#eeeeee', height=0.4, zorder=0)
        ax2.set_xlim(0, 100)
        ax2.set_yticks([])
        ax2.set_xlabel('점수')
        ax2.set_title(f'일관성 점수: {score}/100')
        ax2.text(score / 2, 0, f'{score}점', ha='center', va='center',
                 fontweight='bold', color='white' if score > 20 else 'black', fontsize=13)
    else:
        ax2.text(0.5, 0.5, '점수 파싱 실패', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('일관성 점수')

    period_counts = {'초반': 0, '중반': 0, '후반': 0}
    for h in hits:
        p = h.get('period', '')
        if p in period_counts:
            period_counts[p] += 1

    bars = ax3.bar(period_counts.keys(), period_counts.values(),
                   color=['#3498db', '#9b59b6', '#e67e22'])
    ax3.set_ylabel('발언 수')
    ax3.set_title('시점별 관련 발언 수')
    for bar, val in zip(bars, period_counts.values()):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 str(val), ha='center', va='bottom', fontsize=10)

    if save:
        fname = f'{member}_{topic}_분석.png'
        fpath = os.path.join(HERE, 'output', fname)
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'저장됨: {fpath}')
    else:
        plt.show()


def cmd_chart(args):
    try:
        client = get_client()
        print(f'검색 중: {args.member} × {args.topic}...')
        hits, _ = search(args.member, args.topic, args.top, client)
    except (RuntimeError, ValueError) as e:
        sys.exit(f'[오류] {e}')
    print(f'검색 결과: {len(hits)}건')

    if not hits:
        sys.exit('관련 발언 없음')

    hits = _label_period(hits)
    analysis = analyze(args.member, args.topic, hits, client)
    print('\n' + '=' * 60)
    print(analysis)
    print('=' * 60)

    draw_chart(args.member, args.topic, hits, analysis, save=args.save)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='발언 검색 · 일관성 분석 · 의원 프로필 · 시각화')
    sub = p.add_subparsers(dest='cmd', required=True)

    ps = sub.add_parser('search', help='주제 검색 + 일관성 분석')
    ps.add_argument('--member', required=True, help='의원 이름')
    ps.add_argument('--topic',  required=True, help='검색 주제')
    ps.add_argument('--top',    type=int, default=50, help='검색 결과 수 (기본 50)')
    ps.add_argument('--csv',    action='store_true', help='결과를 CSV로 저장 (유사도 높은 순)')
    ps.set_defaults(func=cmd_search)

    pp = sub.add_parser('profile', help='의원 프로필: 주제 집중도 + 입장 요약')
    pp.add_argument('--member', required=True)
    pp.add_argument('--topics', type=int, default=3, help='상위 몇 개 주제 (기본 3)')
    pp.set_defaults(func=cmd_profile)

    pc = sub.add_parser('chart', help='발언 일관성 시각화')
    pc.add_argument('--member', required=True)
    pc.add_argument('--topic',  required=True)
    pc.add_argument('--top',    type=int, default=50)
    pc.add_argument('--save',   action='store_true', help='화면 대신 PNG 저장')
    pc.set_defaults(func=cmd_chart)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
