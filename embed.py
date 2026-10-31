# 발언 임베딩 파이프라인
# chunks 테이블의 청크 단위로 임베딩 → speeches.db의 embeddings 테이블에 저장
# 검색 시 speech_id로 원문 발언 역참조 가능 (analyze.py가 씀)
#
# 사용:
#   py embed.py --member 주호영
#   py embed.py --member 주호영 --reset  # 기존 데이터 초기화 후 재실행
#
# 의존: pip install openai numpy
#
# chromadb 대신 numpy 브루트포스 코사인 유사도를 씀 — chromadb 1.x가 이 환경(Windows,
# Python 3.14)에서 큰 배치를 넣으면 HNSW 인덱스가 깨지는 알려진 버그가 있고, 구버전은
# 네이티브 확장이 빌드가 안 돼서 설치 불가. 의원당 청크가 최대 수천 개 수준이라
# 브루트포스로도 검색이 0.1초 안팎이라 성능 문제 없음.

import argparse, os, sqlite3, sys, time
import numpy as np


def _find_root():
    """코드는 정민규3/ 안에 있는데 .env·output/은 그 상위 폴더에 있는 상태라,
    둘 중 실제로 .env나 output이 있는 쪽을 프로젝트 루트로 쓴다 (analyze.py와 동일—
    전엔 여기만 안 고쳐서 api.py에서 임베딩을 걸 때 엉뚱한 경로를 보고 있었음)."""
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


def init_embeddings_table(con):
    con.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id  TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
        )
    ''')
    con.commit()


def pack(vec):
    """float 리스트 -> BLOB (float32)"""
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob):
    """BLOB -> float32 numpy 배열"""
    return np.frombuffer(blob, dtype=np.float32)


def get_chunks(member, db_path):
    """의원 청크 조회 (chunks 테이블 × member_speeches JOIN).

    role이 '위원'/'의원'(실제 자기 의견)이 아니라 '위원장'/'소위원장'/'~위원장대리'
    등 회의 진행 role인 청크는 임베딩 대상에서 뺀다 — 확인해보니 상위 의원
    발언의 75~79%가 이런 진행 멘트였음(예: 이철규 79%). 임베딩 비용·저장공간
    낭비이기도 하고, 검색에 섞이면 "의견"으로 오인되는 노이즈이기도 함.
    이미 임베딩된 5명(박대출/백혜련/송석준/안호영/이철규)은 그대로 두고,
    앞으로 새로 임베딩하는 의원부터 적용한다."""
    con = sqlite3.connect(db_path)
    rows = con.execute('''
        SELECT c.id, c.speech_id, c.conf_type, c.conf_dt,
               COALESCE(c.comm_name, ''), c.name,
               COALESCE(ms.party, ''), COALESCE(ms.district, ''),
               c.chunk_seq, c.total_chunks, c.content
        FROM chunks c
        JOIN member_speeches ms ON c.speech_id = ms.id
        WHERE c.name = ?
          AND c.content IS NOT NULL
          AND length(c.content) >= 10
          AND ms.role IN ('위원', '의원')
        ORDER BY c.conf_dt, c.seq, c.chunk_seq
    ''', (member,)).fetchall()
    con.close()
    return rows


def embed_batch(texts, client):
    """OpenAI text-embedding-3-small 배치 임베딩"""
    resp = client.embeddings.create(
        model='text-embedding-3-small',
        input=texts,
    )
    return [d.embedding for d in resp.data]


def embed_member_stream(member, client, db_path=None, batch_size=50):
    """의원 청크를 임베딩하면서 진행 상황을 배치마다 {'done', 'total'} 로 yield.
    build_profile_stream(analyze.py)이 "검색 준비 전" 의원의 프로필을 볼 때
    바로 이 자리에서 임베딩부터 시작하도록 붙이는 용도 — main()과 로직은
    같지만 print/sys.exit 대신 예외를 던지고 결과를 yield해서 API(SSE)에서도
    쓸 수 있게 함. 이미 임베딩된 청크는 건너뛴다."""
    db_path = db_path or DB_PATH
    chunks = get_chunks(member, db_path)
    if not chunks:
        raise ValueError(f'{member} 청크 없음 — 의원 이름 확인 또는 join 실행 여부 확인 (py collect.py join)')

    con = sqlite3.connect(db_path)
    init_embeddings_table(con)

    existing = {row[0] for row in con.execute(
        f"SELECT chunk_id FROM embeddings WHERE chunk_id IN ({','.join('?'*len(chunks))})",
        [r[0] for r in chunks]
    )}
    new_chunks = [r for r in chunks if r[0] not in existing]
    total = len(new_chunks)
    yield {'done': 0, 'total': total}

    for i in range(0, total, batch_size):
        batch = new_chunks[i:i + batch_size]
        ids   = [r[0] for r in batch]
        texts = [r[10] for r in batch]

        vectors = embed_batch(texts, client)
        con.executemany(
            'INSERT OR REPLACE INTO embeddings (chunk_id, embedding) VALUES (?, ?)',
            [(cid, pack(vec)) for cid, vec in zip(ids, vectors)]
        )
        con.commit()

        yield {'done': min(i + batch_size, total), 'total': total}
        time.sleep(0.3)

    con.close()


def main():
    p = argparse.ArgumentParser(description='발언 청크 임베딩 파이프라인')
    p.add_argument('--member', required=True, help='의원 이름 (예: 주호영)')
    p.add_argument('--db',     default=DB_PATH)
    p.add_argument('--reset',  action='store_true', help='기존 임베딩 삭제 후 재실행')
    args = p.parse_args()

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        sys.exit('[오류] OPENAI_API_KEY 없음 — .env에 추가하세요.')

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # ── 청크 조회 ────────────────────────────────────────────────────────────
    chunks = get_chunks(args.member, args.db)
    print(f'{args.member}: 청크 {len(chunks)}개 조회됨')

    con = sqlite3.connect(args.db)
    init_embeddings_table(con)

    if args.reset:
        # get_chunks()가 반환한 "지금 기준으로 유효한" 청크 id만 지우면, 필터
        # 조건(예: role IN ('위원','의원'))이 나중에 바뀌어서 이번엔 제외된
        # 청크의 예전 임베딩이 안 지워지고 그대로 남는 버그가 있었다 (2026-09-22,
        # 위원장 role 필터를 기존 8명에게 소급 적용하다가 발견 — 재임베딩 후에도
        # DB에 예전 임베딩이 뒤섞여 남아서 따로 정리해야 했음). 그래서 chunks
        # 결과와 무관하게, 이 의원 이름의 청크에 걸린 임베딩을 전부 지운다 —
        # chunks가 0개(필터링 후 아무것도 안 남는 경우, 예: 진행 발언만 있던
        # 의원)여도 예전 임베딩은 제대로 청소되게.
        deleted = con.execute(
            'DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE name=?)',
            (args.member,)
        ).rowcount
        con.commit()
        print(f'기존 임베딩 삭제 ({deleted}개)')

    if not chunks:
        con.close()
        sys.exit('청크 없음 — 의원 이름 확인 또는 join 실행 여부 확인 (py collect.py join)')

    # 이미 임베딩된 ID 확인 (중복 방지)
    existing = {row[0] for row in con.execute(
        f"SELECT chunk_id FROM embeddings WHERE chunk_id IN ({','.join('?'*len(chunks))})",
        [r[0] for r in chunks]
    )}
    new_chunks = [r for r in chunks if r[0] not in existing]
    print(f'신규 임베딩 대상: {len(new_chunks)}개 (기존 {len(existing)}개 건너뜀)')

    if not new_chunks:
        print('모두 이미 임베딩됨')
        con.close()
        return

    # ── 배치 임베딩 & 저장 ───────────────────────────────────────────────────
    BATCH = 50
    total = len(new_chunks)

    for i in range(0, total, BATCH):
        batch = new_chunks[i:i+BATCH]

        ids   = [r[0] for r in batch]   # chunk id (예: 054658:1:c1)
        texts = [r[10] for r in batch]  # chunk content

        vectors = embed_batch(texts, client)
        con.executemany(
            'INSERT OR REPLACE INTO embeddings (chunk_id, embedding) VALUES (?, ?)',
            [(cid, pack(vec)) for cid, vec in zip(ids, vectors)]
        )
        con.commit()

        done = min(i + BATCH, total)
        print(f'  [{done:4}/{total}] 완료')
        time.sleep(0.3)

    n = con.execute(
        f"SELECT COUNT(*) FROM embeddings WHERE chunk_id IN "
        f"(SELECT id FROM chunks WHERE name=?)", (args.member,)
    ).fetchone()[0]
    con.close()

    print(f'\n임베딩 완료 → {args.db} (embeddings 테이블)')
    print(f'총 {n}개 청크 저장됨')


if __name__ == '__main__':
    main()
