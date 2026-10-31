# 국회 회의록 + 의원 정보 수집
#
# 서브커맨드 3개:
#   py collect.py meetings --type 본회의 --dae 21 --years 2020 2021 2022 2023 2024
#   py collect.py members  --key YOUR_API_KEY
#   py collect.py join
#
# meetings: API → PDF 임시 다운로드 → OCR(Tesseract) → 파싱 → 청크 → speeches.db 저장
#           (마지막에 members.db가 있으면 join까지 자동 실행)
# members : 역대(20·21대) + 현역(22대) 의원 정보 수집 → members.db + members.csv
# join    : members.db를 speeches.db에 조인 → party/district 채움 + member_speeches 뷰 생성
#
# 의존: pip install pymupdf pytesseract numpy pillow requests
# 시스템 의존: Tesseract OCR (winget install tesseract-ocr.tesseract)

import argparse, csv, os, sqlite3, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse import extract_text, parse_speeches, split_text, validate_speeches

OUT_DIR   = os.path.join(HERE, 'output')
DB_PATH   = os.path.join(OUT_DIR, 'speeches.db')
CACHE_DIR = os.path.join(OUT_DIR, 'ocr_cache')


def load_env():
    env = os.path.join(HERE, '.env')
    if os.path.exists(env):
        for line in open(env, encoding='utf-8'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()


# ═══════════════════════════════════════════════════════════════════════════
# meetings: 회의록 API 목록 + PDF 다운로드/OCR/파싱/저장
# ═══════════════════════════════════════════════════════════════════════════

APIS = {
    "본회의": {
        "url":          "https://open.assembly.go.kr/portal/openapi/nzbyfwhwaoanttzje",
        "key":          "nzbyfwhwaoanttzje",
        "no_year_loop": False,
        "params":       lambda dae, year: {"DAE_NUM": str(dae), "CONF_DATE": str(year)},
        "id_field":     "CONFER_NUM",
        "mapper": lambda r: {
            "conf_id":   r.get("CONF_ID") or r.get("CONFER_NUM"),
            "title":     r.get("TITLE"),
            "comm_name": None,
            "conf_dt":   r.get("CONF_DATE"),
            "pdf_url":   r.get("PDF_LINK_URL"),
        },
    },
    "위원회": {
        "url":          "https://open.assembly.go.kr/portal/openapi/ncwgseseafwbuheph",
        "key":          "ncwgseseafwbuheph",
        "no_year_loop": False,
        "params":       lambda dae, year: {"DAE_NUM": str(dae), "CONF_DATE": str(year)},
        "id_field":     "CONFER_NUM",
        "mapper": lambda r: {
            "conf_id":   r.get("CONF_ID") or r.get("CONFER_NUM"),
            "title":     r.get("TITLE"),
            "comm_name": r.get("COMM_NAME"),
            "conf_dt":   r.get("CONF_DATE"),
            "pdf_url":   r.get("PDF_LINK_URL"),
        },
    },
    "국정감사": {
        "url":          "https://open.assembly.go.kr/portal/openapi/VCONFAPIGCONFLIST",
        "key":          "VCONFAPIGCONFLIST",
        "no_year_loop": True,
        "params":       lambda dae, year: {"ERACO": f"제{dae}대"},
        "id_field":     "CONF_ID",
        "mapper": lambda r: {
            "conf_id":   r.get("CONF_ID"),
            "title":     r.get("CMIT_NM"),
            "comm_name": r.get("CMIT_NM"),
            "conf_dt":   r.get("CONF_DT"),
            "pdf_url":   r.get("DOWN_URL"),
        },
    },
}


def init_db(db_path):
    con = sqlite3.connect(db_path)
    con.executescript('''
        CREATE TABLE IF NOT EXISTS speeches (
            id         TEXT PRIMARY KEY,
            conf_id    TEXT,
            conf_type  TEXT,
            dae        TEXT,
            conf_dt    TEXT,
            comm_name  TEXT,
            seq        INTEGER,
            prev_seq   INTEGER,
            name       TEXT,
            role       TEXT,
            speaker    TEXT,
            content    TEXT,
            party      TEXT,
            district   TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_sp_name  ON speeches(name);
        CREATE INDEX IF NOT EXISTS ix_sp_conf  ON speeches(conf_id);
        CREATE INDEX IF NOT EXISTS ix_sp_type  ON speeches(conf_type);
        CREATE INDEX IF NOT EXISTS ix_sp_dt    ON speeches(conf_dt);

        CREATE TABLE IF NOT EXISTS chunks (
            id           TEXT PRIMARY KEY,
            speech_id    TEXT,
            conf_id      TEXT,
            conf_type    TEXT,
            conf_dt      TEXT,
            comm_name    TEXT,
            name         TEXT,
            speaker      TEXT,
            seq          INTEGER,
            chunk_seq    INTEGER,
            total_chunks INTEGER,
            content      TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_ch_speech ON chunks(speech_id);
        CREATE INDEX IF NOT EXISTS ix_ch_conf   ON chunks(conf_id);

        CREATE TABLE IF NOT EXISTS processed (
            conf_id      TEXT PRIMARY KEY,
            conf_type    TEXT,
            speech_count INTEGER,
            chunk_count  INTEGER,
            processed_at TEXT
        );
    ''')
    con.commit()
    return con


SPEECH_FIELDS = ['id', 'conf_id', 'conf_type', 'dae', 'conf_dt', 'comm_name',
                 'seq', 'prev_seq', 'name', 'role', 'speaker',
                 'content_len', 'content', 'party', 'district']

CHUNK_FIELDS  = ['id', 'speech_id', 'conf_id', 'conf_type', 'conf_dt', 'comm_name',
                 'name', 'speaker', 'seq', 'chunk_seq', 'total_chunks', 'content']


def open_csv_writers(db_path):
    """speeches.csv / chunks.csv 를 DB 파일 옆에 열기. 이미 있으면 append."""
    base = os.path.splitext(db_path)[0]
    writers = {}
    handles = {}
    for name, fields in [('speeches', SPEECH_FIELDS), ('chunks', CHUNK_FIELDS)]:
        path   = f"{base}_{name}.csv"
        is_new = not os.path.exists(path)
        fh     = open(path, 'a', newline='', encoding='utf-8-sig')
        w      = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        if is_new:
            w.writeheader()
        writers[name] = w
        handles[name] = fh
    return writers, handles


def close_csv_writers(handles):
    for fh in handles.values():
        fh.close()


def save_meeting(con, speeches_raw, meeting, conf_type, dae, csv_writers=None):
    conf_id   = meeting["conf_id"]
    conf_dt   = (meeting.get("conf_dt") or "").replace("/", "-")
    comm_name = meeting.get("comm_name") or ""

    chunk_count = 0

    for i, s in enumerate(speeches_raw, 1):
        speaker = (f"{s['name']} {s['role']}"
                   if s['role'] in ('의원', '위원')
                   else f"{s['role']} {s['name']}")

        speech_id = f"{conf_id}:{i}"
        prev_seq  = s['prev_seq'] if s['prev_seq'] else None

        con.execute(
            'INSERT OR REPLACE INTO speeches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (speech_id, conf_id, conf_type, str(dae),
             conf_dt, comm_name,
             i, prev_seq,
             s['name'], s['role'], speaker,
             s['text'], None, None)
        )

        if csv_writers:
            csv_writers['speeches'].writerow({
                'id': speech_id, 'conf_id': conf_id, 'conf_type': conf_type,
                'dae': str(dae), 'conf_dt': conf_dt, 'comm_name': comm_name,
                'seq': i, 'prev_seq': prev_seq or '',
                'name': s['name'], 'role': s['role'], 'speaker': speaker,
                'content_len': len(s['text']), 'content': s['text'],
                'party': '', 'district': '',
            })

        chunks = split_text(s['text'])
        total  = len(chunks)
        for ci, chunk_text in enumerate(chunks, 1):
            chunk_id = f"{speech_id}:c{ci}"
            con.execute(
                'INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (chunk_id, speech_id,
                 conf_id, conf_type, conf_dt, comm_name,
                 s['name'], speaker,
                 i, ci, total, chunk_text)
            )
            if csv_writers:
                csv_writers['chunks'].writerow({
                    'id': chunk_id, 'speech_id': speech_id,
                    'conf_id': conf_id, 'conf_type': conf_type,
                    'conf_dt': conf_dt, 'comm_name': comm_name,
                    'name': s['name'], 'speaker': speaker,
                    'seq': i, 'chunk_seq': ci, 'total_chunks': total,
                    'content': chunk_text,
                })
        chunk_count += total

    con.execute(
        'INSERT OR REPLACE INTO processed VALUES (?,?,?,?,datetime("now","localtime"))',
        (conf_id, conf_type, len(speeches_raw), chunk_count)
    )
    con.commit()
    return chunk_count


def fetch_meetings(api_key, conf_type, dae, year):
    cfg     = APIS[conf_type]
    params  = cfg["params"](dae, year)

    meetings = {}
    page = 1
    while True:
        try:
            res = requests.get(cfg["url"], params={
                "KEY": api_key, "Type": "json",
                "pIndex": page, "pSize": 100,
                **params,
            }, timeout=30).json()
        except Exception as e:
            print(f"  [API 오류] {e}")
            break

        data  = res.get(cfg["key"], [{}])
        head  = data[0].get("head", [{}]) if data else [{}]
        total = head[0].get("list_total_count", 0) if head else 0
        rows  = data[1].get("row", []) if len(data) > 1 else []

        for r in rows:
            uid = r.get(cfg["id_field"])
            if uid and uid not in meetings:
                meetings[uid] = cfg["mapper"](r)

        fetched = (page - 1) * 100 + len(rows)
        if fetched >= total or not rows:
            break
        page += 1
        time.sleep(0.2)

    return list(meetings.values())


def fetch_and_parse(meeting, conf_type, cache_dir=CACHE_DIR):
    """PDF 다운로드 + OCR + 파싱. DB 접근이 전혀 없어서 여러 스레드에서 동시에
    안전하게 호출 가능 (sqlite 커넥션은 스레드 간 공유하면 안 되므로 이 함수는
    그 부분을 아예 담당하지 않음 — 저장은 메인 스레드의 save_meeting()이 함)."""
    conf_id = meeting.get("conf_id")
    pdf_url = meeting.get("pdf_url")

    if not pdf_url:
        return "no_pdf", None

    # record.assembly.go.kr가 동시 다운로드가 많으면 연결을 자주 끊는 것으로
    # 보여서(레이트리밋 추정), 지수 백오프로 재시도한다.
    r = None
    last_err = None
    for attempt in range(4):
        try:
            r = requests.get(pdf_url, timeout=180)
            if r.status_code == 200 and b'%PDF' in r.content[:10]:
                break
            last_err = f"status={r.status_code}"
            r = None
        except Exception as e:
            last_err = e
            r = None
        time.sleep(2 ** attempt)  # 1, 2, 4, 8초
    if r is None:
        return f"err:{last_err}", None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name

        text     = extract_text(tmp_path, conf_id=conf_id, cache_dir=cache_dir)
        speeches = parse_speeches(text, conf_type)

        if not speeches:
            return "empty", None
        return "ok", speeches

    except Exception as e:
        return f"err:{e}", None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_meetings(conf_type, dae, years, api_key, db_path, cache_dir=CACHE_DIR,
                  limit=None, validate=False, force=False, monthly=None, workers=1):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con                   = init_db(db_path)
    csv_writers, handles  = open_csv_writers(db_path)
    cfg                   = APIS[conf_type]

    all_meetings = []
    if cfg["no_year_loop"]:
        print(f"[{dae}대 / {conf_type}] 목록 수집 중...")
        all_meetings = fetch_meetings(api_key, conf_type, dae, None)
    else:
        for year in years:
            print(f"[{dae}대 {year}년 / {conf_type}] 목록 수집 중...")
            all_meetings += fetch_meetings(api_key, conf_type, dae, year)

    if limit:
        all_meetings = all_meetings[:limit]

    if monthly:
        from collections import defaultdict
        by_month = defaultdict(list)
        for m in all_meetings:
            dt = (m.get('conf_dt') or '').replace('/', '-')[:7]
            by_month[dt].append(m)
        sampled = []
        for month in sorted(by_month):
            sampled.extend(by_month[month][:monthly])
        all_meetings = sampled
        print(f'월별 샘플링 ({monthly}개/월): {len(all_meetings)}개 회의 선택')

    total = len(all_meetings)

    if force:
        pending = all_meetings
        skip = 0
    else:
        processed_ids = {row[0] for row in con.execute('SELECT conf_id FROM processed')}
        pending = [m for m in all_meetings if m.get('conf_id') not in processed_ids]
        skip = total - len(pending)

    print(f"총 {total}건 (건너뜀 {skip}건 제외, 처리 대상 {len(pending)}건)\n")
    if not pending:
        con.close()
        close_csv_writers(handles)
        return

    ok = fail = 0
    total_speeches = total_chunks = 0

    if workers <= 1:
        for idx, meeting in enumerate(pending, 1):
            conf_dt  = (meeting.get("conf_dt") or "").replace("/", "-")[:10]
            label    = meeting.get("comm_name") or meeting.get("title") or ""
            progress = f"[{idx:4}/{len(pending)}] {conf_dt} {label}"

            result, speeches = fetch_and_parse(meeting, conf_type, cache_dir)
            if result == "ok":
                if validate:
                    validate_speeches(speeches, source=meeting.get('conf_id'))
                n_ch = save_meeting(con, speeches, meeting, conf_type, dae, csv_writers)
                n_sp = len(speeches)
                print(f"  OK  {progress}  ({n_sp}발언 / {n_ch}청크)")
                ok += 1
                total_speeches += n_sp
                total_chunks   += n_ch
            elif result in ("no_pdf", "empty"):
                print(f"  ..  {progress}  [{result}]")
            else:
                print(f"  XX  {progress}  [{result}]")
                fail += 1
            time.sleep(0.3)
    else:
        print(f"워커 {workers}개로 병렬 처리 (PDF 다운로드+OCR만 — DB 저장은 순서대로 메인 스레드에서)\n")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fetch_and_parse, m, conf_type, cache_dir): m for m in pending}
            for idx, fut in enumerate(as_completed(futures), 1):
                meeting  = futures[fut]
                conf_dt  = (meeting.get("conf_dt") or "").replace("/", "-")[:10]
                label    = meeting.get("comm_name") or meeting.get("title") or ""
                progress = f"[{idx:4}/{len(pending)}] {conf_dt} {label}"

                try:
                    result, speeches = fut.result()
                except Exception as e:
                    result, speeches = f"err:{e}", None

                if result == "ok":
                    if validate:
                        validate_speeches(speeches, source=meeting.get('conf_id'))
                    n_ch = save_meeting(con, speeches, meeting, conf_type, dae, csv_writers)
                    n_sp = len(speeches)
                    print(f"  OK  {progress}  ({n_sp}발언 / {n_ch}청크)")
                    ok += 1
                    total_speeches += n_sp
                    total_chunks   += n_ch
                elif result in ("no_pdf", "empty"):
                    print(f"  ..  {progress}  [{result}]")
                else:
                    print(f"  XX  {progress}  [{result}]")
                    fail += 1

    con.close()
    close_csv_writers(handles)

    base = os.path.splitext(db_path)[0]
    print(f"\n=== {conf_type} 완료 ===")
    print(f"성공 {ok} / 건너뜀 {skip} / 실패 {fail}")
    print(f"발언 {total_speeches}건 / 청크 {total_chunks}건")
    print(f"  DB  → {db_path}")
    print(f"  CSV → {base}_speeches.csv")
    print(f"  CSV → {base}_chunks.csv")


def cmd_meetings(args):
    api_key = args.key or os.getenv("ASSEMBLY_API_KEY")
    if not api_key:
        sys.exit("[오류] API 키 없음. .env에 ASSEMBLY_API_KEY를 추가하세요.")

    cache_dir = None if args.no_cache else args.cache_dir
    validate  = args.limit is not None

    targets = list(APIS.keys()) if args.all else [args.type]
    for t in targets:
        run_meetings(t, args.dae, args.years, api_key, args.out, cache_dir=cache_dir,
                     limit=args.limit, validate=validate, force=args.force, monthly=args.monthly,
                     workers=args.workers)
        print()

    members_path = os.path.join(HERE, 'members.db')
    if os.path.exists(members_path):
        print("=== members 조인 ===")
        join_members(args.out, members_path)
    else:
        print("[경고] members.db 없음 — 조인 건너뜀 (py collect.py members 먼저 실행)")


# ═══════════════════════════════════════════════════════════════════════════
# members: 역대+현역 국회의원 정보 수집
# ═══════════════════════════════════════════════════════════════════════════

HISTORICAL_URL = "https://open.assembly.go.kr/portal/openapi/npffdutiapkzbfyvr"
CURRENT_URL    = "https://open.assembly.go.kr/portal/openapi/nwvrqwxyaytdsfvhu"

MEMBER_COLUMNS = ["mona_cd", "name", "dae", "party", "district", "elect_type", "reele", "units", "source"]


def fetch_historical(api_key, dae, unit_cd):
    rows, page = [], 1
    while True:
        res = requests.get(HISTORICAL_URL, params={
            "KEY": api_key, "Type": "json",
            "pIndex": page, "pSize": 300,
            "UNIT_CD": unit_cd,
        }).json()
        data  = res.get("npffdutiapkzbfyvr", [{}])
        total = data[0].get("head", [{}])[0].get("list_total_count", 0)
        batch = data[1].get("row", []) if len(data) > 1 else []
        rows.extend(batch)
        if len(rows) >= total or not batch:
            break
        page += 1
    return [{"mona_cd": r.get("MONA_CD"), "name": r.get("HG_NM"), "dae": dae,
             "party": r.get("POLY_NM"), "district": r.get("ORIG_NM"),
             "elect_type": r.get("ELECT_GBN_NM"), "reele": r.get("REELE_GBN_NM"),
             "units": r.get("UNITS"), "source": "historical"} for r in rows]


def fetch_current(api_key):
    rows, page = [], 1
    while True:
        res = requests.get(CURRENT_URL, params={
            "KEY": api_key, "Type": "json",
            "pIndex": page, "pSize": 300,
        }).json()
        data  = res.get("nwvrqwxyaytdsfvhu", [{}])
        total = data[0].get("head", [{}])[0].get("list_total_count", 0)
        batch = data[1].get("row", []) if len(data) > 1 else []
        rows.extend(batch)
        if len(rows) >= total or not batch:
            break
        page += 1
    return [{"mona_cd": r.get("MONA_CD"), "name": r.get("HG_NM"), "dae": "22",
             "party": r.get("POLY_NM"), "district": r.get("ORIG_NM"),
             "elect_type": r.get("ELECT_GBN_NM"), "reele": r.get("REELE_GBN_NM"),
             "units": r.get("UNITS"), "source": "current"} for r in rows]


def save_members_db(members, db_path):
    con = sqlite3.connect(db_path)
    con.execute('DROP TABLE IF EXISTS members')
    con.execute('''
        CREATE TABLE members (
            mona_cd TEXT, name TEXT, dae TEXT,
            party TEXT, district TEXT, elect_type TEXT,
            reele TEXT, units TEXT, source TEXT
        )
    ''')
    con.execute('CREATE INDEX ix_name ON members(name)')
    con.execute('CREATE INDEX ix_dae  ON members(dae)')
    con.executemany(
        'INSERT INTO members VALUES (:mona_cd,:name,:dae,:party,:district,'
        ':elect_type,:reele,:units,:source)', members)
    con.commit()
    con.close()


def save_members_csv(members, csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=MEMBER_COLUMNS)
        w.writeheader()
        w.writerows(members)


def cmd_members(args):
    api_key = args.key or os.getenv("ASSEMBLY_API_KEY")
    if not api_key:
        sys.exit("[오류] API 키 없음. .env 파일에 ASSEMBLY_API_KEY=키 추가하거나 --key 인자로 전달하세요.")

    all_members = []
    for dae, unit_cd in [("20", "100020"), ("21", "100021")]:
        rows = fetch_historical(api_key, dae, unit_cd)
        print(f"{dae}대: {len(rows)}명")
        all_members.extend(rows)

    rows = fetch_current(api_key)
    print(f"22대 현역: {len(rows)}명")
    all_members.extend(rows)

    save_members_db(all_members, args.out)
    csv_path = args.out.replace('.db', '.csv')
    save_members_csv(all_members, csv_path)

    print(f"\n저장 완료")
    print(f"  DB : {args.out} (총 {len(all_members)}명)")
    print(f"  CSV: {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# join: members.db → speeches.db 조인 (party/district 채움 + member_speeches 뷰)
# ═══════════════════════════════════════════════════════════════════════════

# 대수별 임기 시작일 (국회의원 총선 다음날부터 임기 시작). 발언 날짜가 어느
# 대수 구간에 속하는지로 매칭할 dae를 판정한다 — dae를 '22'로 고정하면 지금처럼
# 수집된 발언이 전부 21대 기간(2020-06~2024-05)인 상황에서 명단을 못 찾는다.
_DAE_START = [
    ('22', '2024-05-30'),
    ('21', '2020-05-30'),
    ('20', '2016-05-30'),
]


def _dae_for_date(conf_dt):
    if not conf_dt:
        return None
    for dae, start in _DAE_START:
        if conf_dt >= start:
            return dae
    return '19'


def join_members(speeches_path, members_path):
    if not os.path.exists(speeches_path):
        print(f"[오류] speeches.db 없음: {speeches_path}")
        return
    if not os.path.exists(members_path):
        print(f"[오류] members.db 없음: {members_path}")
        return

    con = sqlite3.connect(speeches_path)
    con.create_function('dae_for_date', 1, _dae_for_date)

    already = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='members'"
    ).fetchone()

    if already:
        print("members 테이블 이미 존재 — 복사 건너뜀")
    else:
        con.execute(f"ATTACH DATABASE '{members_path}' AS mb")
        con.execute("CREATE TABLE members AS SELECT * FROM mb.members")
        con.execute("CREATE INDEX IF NOT EXISTS ix_mb_name ON members(name)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_mb_dae  ON members(dae)")
        con.execute("DETACH DATABASE mb")
        con.commit()
        count = con.execute("SELECT COUNT(*) FROM members").fetchone()[0]
        print(f"members 테이블 복사 완료: {count:,}명")

    cur = con.execute("""
        UPDATE speeches
        SET party    = (SELECT m.party    FROM members m
                        WHERE m.name = speeches.name
                        AND m.dae = dae_for_date(speeches.conf_dt)),
            district = (SELECT m.district FROM members m
                        WHERE m.name = speeches.name
                        AND m.dae = dae_for_date(speeches.conf_dt))
        WHERE EXISTS (
            SELECT 1 FROM members m
            WHERE m.name = speeches.name
            AND m.dae = dae_for_date(speeches.conf_dt)
        )
    """)
    updated = cur.rowcount
    con.commit()
    print(f"party/district 업데이트: {updated:,}건")

    total     = con.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
    matched   = con.execute("SELECT COUNT(*) FROM speeches WHERE party IS NOT NULL").fetchone()[0]
    unmatched = total - matched

    print()
    print(f"speeches 전체:  {total:>7,}건")
    if total:
        print(f"의원 매칭됨:    {matched:>7,}건  ({matched/total*100:.1f}%)")
        print(f"매칭 안 됨:     {unmatched:>7,}건  (장관·위원장·참고인 등)")
    print()

    print("=== 정당별 발언 수 ===")
    for row in con.execute("""
        SELECT party, COUNT(*) as cnt
        FROM speeches
        WHERE party IS NOT NULL
        GROUP BY party
        ORDER BY cnt DESC
    """):
        print(f"  {row[0]:20} {row[1]:>6,}건")
    print()

    con.execute("DROP VIEW IF EXISTS member_speeches")
    con.execute("""
        CREATE VIEW member_speeches AS
        SELECT s.*
        FROM speeches s
        WHERE s.party IS NOT NULL
    """)
    con.commit()

    view_count = con.execute("SELECT COUNT(*) FROM member_speeches").fetchone()[0]
    print(f"member_speeches 뷰 생성: {view_count:,}건")
    print("  → 이후 분석은 이 뷰를 사용하세요")
    print()
    print(f"  speeches.db → {speeches_path}")

    con.close()


def cmd_join(args):
    join_members(args.speeches, args.members)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import datetime
    cur_year = datetime.date.today().year

    p = argparse.ArgumentParser(description="국회 회의록 + 의원 정보 수집")
    sub = p.add_subparsers(dest='cmd', required=True)

    # ── meetings ──────────────────────────────────────────────────────────
    pm = sub.add_parser('meetings', help='회의록 수집 (기본 커맨드)')
    pm.add_argument("--type",  default="본회의", choices=list(APIS.keys()),
                    help="회의 종류 (기본: 본회의)")
    pm.add_argument("--dae",   type=int, default=22,
                    help="대수 (기본: 22, 21대는 --dae 21)")
    pm.add_argument("--years", type=int, nargs="+",
                    default=list(range(2024, cur_year + 1)),
                    help="수집 연도 (기본: 22대 시작~현재, 21대는 --years 2020 2021 2022 2023 2024)")
    pm.add_argument("--all",   action="store_true",
                    help="본회의 + 위원회 + 국정감사 전부 실행")
    pm.add_argument("--out",   default=DB_PATH,
                    help=f"DB 저장 경로 (기본: {DB_PATH})")
    pm.add_argument("--key",   default=None,
                    help="API 키 (.env 우선)")
    pm.add_argument("--limit", type=int, default=None,
                    help="처리할 최대 회의 수 (테스트용)")
    pm.add_argument("--force", action="store_true",
                    help="이미 처리된 회의도 재처리 (processed 테이블 무시)")
    pm.add_argument("--monthly", type=int, default=None,
                    help="월별 최대 처리 회의 수 (예: --monthly 2 → 각 월에서 2개씩)")
    pm.add_argument("--cache-dir", default=CACHE_DIR,
                    help=f"OCR 결과 캐시 디렉터리 (기본: {CACHE_DIR})")
    pm.add_argument("--no-cache",  action="store_true",
                    help="OCR 캐시 사용 안 함 (매번 재OCR)")
    pm.add_argument("--workers", type=int, default=1,
                    help="회의 단위 병렬 처리 워커 수 (기본: 1=순차. CPU 코어 수만큼 권장)")
    pm.set_defaults(func=cmd_meetings)

    # ── members ───────────────────────────────────────────────────────────
    pmem = sub.add_parser('members', help='역대+현역 국회의원 정보 수집')
    pmem.add_argument("--key", default=os.getenv("ASSEMBLY_API_KEY"), help="열린국회정보 API 키 (.env 또는 직접 입력)")
    pmem.add_argument("--out", default=os.path.join(HERE, "members.db"), help="출력 DB (기본: members.db)")
    pmem.set_defaults(func=cmd_members)

    # ── join ──────────────────────────────────────────────────────────────
    pj = sub.add_parser('join', help='members.db를 speeches.db에 조인')
    pj.add_argument("--speeches", default=DB_PATH)
    pj.add_argument("--members",  default=os.path.join(HERE, "members.db"))
    pj.set_defaults(func=cmd_join)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
