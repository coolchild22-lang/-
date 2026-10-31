# classify_stances 정확도 검증용 정답 라벨셋 구축 + 채점
#
# 서브커맨드 3개:
#   py evaluate.py sample --n 100   # 라벨링 대상 CSV + GPT 판정(숨김) CSV 생성
#   py evaluate.py score            # 사람이 채운 라벨링 CSV와 GPT 판정을 비교해 정확도 계산
#   py evaluate.py reclassify       # 같은 50건 라벨셋 유지, classify_stances 프롬프트
#                                    # 수정 후 GPT 판정만 새로 돌려서 전/후 비교할 때 씀
#
# 왜 만들었나 (2026-09-22, 박사님 피드백):
# 지금까지 검색·입장분류 정확도는 백혜련/이철규 등 몇 개 사례를 사람이 눈으로
# 확인한 수준이었지, precision/recall처럼 정량 지표로 측정한 적이 없었다.
# "몇 퍼센트 정확하다"고 말할 근거가 없다는 지적에 대한 답으로, classify_stances
# (relevant·stance 판정 — search/profile/chat_answer가 전부 이 함수에 의존하는
# 병목 지점)의 정확도부터 정답 라벨셋으로 검증한다.
#
# sample이 만드는 두 파일:
#   eval_label_이것부터_채우기.csv  — 사람이 relevant/stance를 채울 파일
#                                     (GPT 판정은 안 보임 — 앵커링 편향 방지)
#   eval_gpt_판정_숨김.csv          — GPT의 classify_stances 판정. 라벨링 다
#                                     끝내기 전엔 열어보지 말 것.
#
# 샘플링 방식: 유사도 구간별 층화추출 (0.35~0.45 구간에 무관한 발언이 자주
# 섞인다고 이미 확인됐으므로 그 구간 비중을 높게 잡음) + 이미 알려진 문제
# 사례(백혜련×해병대원특검, 이철규×연금개혁)는 회귀 테스트용으로 항상 포함.

import argparse, csv, os, random, sys
from collections import Counter

import analyze

HERE = analyze.HERE
LABEL_CSV  = os.path.join(HERE, 'eval_label_이것부터_채우기.csv')
ANSWER_CSV = os.path.join(HERE, 'eval_gpt_판정_숨김.csv')

MEMBERS = ['백혜련', '이원욱', '조응천', '안호영', '박대출', '이철규', '송석준']

# (하한, 상한, 목표 비중) — 하한 0.35는 search()의 SIM_THRESHOLD와 동일
SIM_BANDS = [
    (0.35, 0.45, 0.5),
    (0.45, 0.60, 0.3),
    (0.60, 1.01, 0.2),
]

# 이미 알려진 문제 사례 — 회귀 테스트용으로 항상 포함 (CLAUDE.md 기록 참고).
# 검색 자체는 top_k=30까지 반환하지만, 앵커 하나당 이 개수만큼만 강제 포함하고
# 나머지는 일반 후보처럼 유사도 구간 층화추출 대상에 넣는다 — 안 그러면 앵커
# 둘이서만 샘플 예산을 통째로 잡아먹는다 (2026-09-22, 첫 실행에서 100건 중
# 59건이 앵커로 쏠리는 버그를 실제로 겪고 수정함).
ANCHOR_CASES = [
    ('백혜련', '해병대원 특검'),
    ('이철규', '연금개혁'),
]
ANCHOR_MAX_PER_CASE = 5

RANDOM_SEED = 42


def _band_of(sim):
    for lo, hi, _ in SIM_BANDS:
        if lo <= sim < hi:
            return (lo, hi)
    return SIM_BANDS[-1][:2]


def cmd_sample(args):
    client = analyze.get_client()
    rng = random.Random(RANDOM_SEED)

    pool = {}   # (member, topic, speech_id) -> (member, topic, hit)

    def add(member, topic, hits):
        for h in hits:
            key = (member, topic, h['speech_id'])
            pool.setdefault(key, (member, topic, h))

    print('후보 수집 중 (임베딩 검색만 — GPT 호출 없음)...')

    anchors = []
    for member, topic in ANCHOR_CASES:
        query = analyze.TOPICS.get(topic)
        if not query:
            continue
        hits, total = analyze.search(member, query, 30, client)
        add(member, topic, hits)
        # 유사도 상위 ANCHOR_MAX_PER_CASE개만 앵커로 강제 포함 (나머지는 일반
        # 후보로 풀에 남아서 층화추출 대상이 될 수 있음)
        top_hits = sorted(hits, key=lambda h: -h['similarity'])[:ANCHOR_MAX_PER_CASE]
        anchors.extend((member, topic, h['speech_id']) for h in top_hits)

    for member in MEMBERS:
        results = analyze.rank_topics(member, client, top_k=30)
        for topic, (total, hits) in results.items():
            add(member, topic, hits)

    print(f'후보 풀 크기: {len(pool)}건')

    # 앵커는 항상 포함, 나머지는 유사도 구간별 층화추출로 채움
    anchor_keys = [k for k in anchors if k in pool]
    remaining = {k: v for k, v in pool.items() if k not in anchor_keys}

    n_target = max(0, args.n - len(anchor_keys))
    banded = {b[:2]: [] for b in SIM_BANDS}
    for key, (member, topic, h) in remaining.items():
        band = _band_of(h['similarity'])
        banded.setdefault(band, []).append(key)

    selected_keys = list(anchor_keys)
    quotas = {b[:2]: round(n_target * b[2]) for b in SIM_BANDS}
    for band, keys in banded.items():
        rng.shuffle(keys)
        selected_keys.extend(keys[:quotas.get(band, 0)])

    # 목표치에 못 미치면 남은 후보로 채움
    if len(selected_keys) < args.n:
        leftover = [k for k in remaining if k not in selected_keys]
        rng.shuffle(leftover)
        selected_keys.extend(leftover[:args.n - len(selected_keys)])

    selected_keys = selected_keys[:args.n]
    print(f'최종 선택: {len(selected_keys)}건 (앵커 {len(anchor_keys)}건 포함)')

    # 선택된 항목을 (member, topic)별로 묶어서 classify_stances 배치 호출
    # (실제 시스템이 쓰는 방식과 동일 — 발언 하나씩 개별 호출하지 않음)
    grouped = {}
    for key in selected_keys:
        member, topic, _ = key
        grouped.setdefault((member, topic), []).append(key)

    # 발언 개수 많은 그룹을 한 번에 분류시키면 GPT 응답이 길어져서 JSON이
    # 중간에 잘리는 경우가 실제로 있었다 (2026-09-22, 30건짜리 그룹에서 재현 —
    # classify_stances 자체의 잠재 버그라 프로덕션에서도 날 수 있어 별도로
    # 고쳐야 함). 여기서는 그룹을 CHUNK_SIZE 단위로 쪼개 호출해서 잘림 위험을
    # 줄이고, 그래도 실패하는 청크는 건너뛰고 계속 진행한다.
    CHUNK_SIZE = 15
    print(f'classify_stances 호출 중 ({len(grouped)}개 (의원,주제) 그룹, {CHUNK_SIZE}건씩 분할)...')
    answers = {}   # key -> classification dict
    n_failed = 0
    for (member, topic), keys in grouped.items():
        for i in range(0, len(keys), CHUNK_SIZE):
            chunk_keys = keys[i:i + CHUNK_SIZE]
            hits = [pool[k][2] for k in chunk_keys]
            try:
                stances = analyze.classify_stances(member, topic, hits, client)
            except Exception as e:
                print(f'  [경고] {member} x {topic} ({len(hits)}건) classify_stances 실패, 건너뜀: {e}')
                n_failed += len(chunk_keys)
                continue
            for k, s in zip(chunk_keys, stances):
                answers[k] = s

    if n_failed:
        selected_keys = [k for k in selected_keys if k in answers]
        print(f'{n_failed}건은 분류 실패로 라벨셋에서 제외 (최종 {len(selected_keys)}건)')

    rng.shuffle(selected_keys)   # 라벨링 파일에서 순서로 힌트 안 주도록 섞음

    with open(LABEL_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['id', 'member', 'topic', 'date', 'conf_type', 'comm_name',
                    'similarity', 'excerpt',
                    '사람_relevant(TRUE/FALSE)', '사람_stance(찬성/반대/중립/판단유보)', '비고'])
        for key in selected_keys:
            member, topic, speech_id = key
            h = pool[key][2]
            row_id = f'{member}|{topic}|{speech_id}'
            excerpt = (h.get('excerpt') or h['content'])[:600]
            w.writerow([row_id, member, topic, h['conf_dt'][:10], h['conf_type'],
                        h['comm_name'], h['similarity'], excerpt, '', '', ''])

    with open(ANSWER_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['id', 'member', 'topic', 'date', 'gpt_relevant', 'gpt_stance',
                    'gpt_confidence', 'gpt_quote'])
        for key in selected_keys:
            member, topic, speech_id = key
            h = pool[key][2]
            s = answers[key]
            row_id = f'{member}|{topic}|{speech_id}'
            w.writerow([row_id, member, topic, h['conf_dt'][:10],
                        s['relevant'], s['stance'], s['confidence'], s['quote']])

    print(f'\n라벨링할 파일: {LABEL_CSV}')
    print(f'(비교용, 라벨링 끝나기 전엔 열지 말 것): {ANSWER_CSV}')
    print('\nLABEL 파일의 "사람_relevant"/"사람_stance" 칸을 채운 뒤 `py evaluate.py score`를 실행하세요.')
    print('relevant가 FALSE면 stance는 자동으로 판단유보로 취급되니 "판단유보"라고 적어주세요.')


def cmd_reclassify(args):
    """classify_stances 프롬프트를 고친 뒤, 기존에 뽑아둔 50건 라벨셋은 그대로
    두고 GPT 판정(ANSWER_CSV)만 새로 돌려서 덮어쓴다. sample을 다시 돌리면
    표본 자체가 바뀌어서 "프롬프트 수정 전/후"를 같은 기준으로 비교할 수 없기
    때문에 따로 만듦 (2026-10-02, relevant 판정 기준 수정 후 효과 검증용).

    LABEL_CSV의 excerpt 칸이 이미 cmd_sample에서 (h.get('excerpt') or
    h['content'])[:600]로 잘라둔 것과 동일해서, DB를 다시 조회하지 않고 CSV
    행만으로 classify_stances에 넣을 hit을 재구성할 수 있다."""
    client = analyze.get_client()
    label_rows = _read_csv(LABEL_CSV)

    grouped = {}
    for r in label_rows:
        grouped.setdefault((r['member'], r['topic']), []).append(r)

    CHUNK_SIZE = 15
    print(f'classify_stances 재호출 중 ({len(grouped)}개 (의원,주제) 그룹)...')
    answers = {}
    n_failed = 0
    for (member, topic), rows in grouped.items():
        for i in range(0, len(rows), CHUNK_SIZE):
            chunk = rows[i:i + CHUNK_SIZE]
            hits = [{'conf_dt': r['date'], 'excerpt': r['excerpt'], 'content': r['excerpt']} for r in chunk]
            try:
                stances = analyze.classify_stances(member, topic, hits, client)
            except Exception as e:
                print(f'  [경고] {member} x {topic} ({len(hits)}건) 실패, 건너뜀: {e}')
                n_failed += len(chunk)
                continue
            for r, s in zip(chunk, stances):
                answers[r['id']] = s

    if n_failed:
        print(f'{n_failed}건 분류 실패 (기존 판정 유지됨)')

    old_answers = {r['id']: r for r in _read_csv(ANSWER_CSV)}
    with open(ANSWER_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['id', 'member', 'topic', 'date', 'gpt_relevant', 'gpt_stance', 'gpt_confidence', 'gpt_quote'])
        for r in label_rows:
            s = answers.get(r['id'])
            if s is None:
                old = old_answers.get(r['id'], {})
                w.writerow([r['id'], r['member'], r['topic'], r['date'],
                            old.get('gpt_relevant', ''), old.get('gpt_stance', ''),
                            old.get('gpt_confidence', ''), old.get('gpt_quote', '')])
            else:
                w.writerow([r['id'], r['member'], r['topic'], r['date'],
                            s['relevant'], s['stance'], s['confidence'], s['quote']])

    print(f'\n{ANSWER_CSV} 갱신 완료. `py evaluate.py score`로 비교하세요.')


def _read_csv(path):
    if not os.path.exists(path):
        sys.exit(f'[오류] {path} 없음 — 먼저 `py evaluate.py sample`을 실행하세요.')
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def cmd_score(args):
    label_rows = _read_csv(LABEL_CSV)
    answer_rows = {r['id']: r for r in _read_csv(ANSWER_CSV)}

    labeled = [r for r in label_rows if r['사람_relevant(TRUE/FALSE)'].strip()]
    print(f'라벨링 완료: {len(labeled)} / {len(label_rows)}건')
    if not labeled:
        sys.exit('아직 라벨링된 행이 없습니다.')

    tp = fp = fn = tn = 0
    stance_correct = 0
    stance_confusion = Counter()
    disagreements = []

    for r in labeled:
        row_id = r['id']
        a = answer_rows.get(row_id)
        if not a:
            continue

        human_relevant = r['사람_relevant(TRUE/FALSE)'].strip().upper() == 'TRUE'
        gpt_relevant = a['gpt_relevant'].strip().upper() == 'TRUE'

        if human_relevant and gpt_relevant:
            tp += 1
        elif not human_relevant and gpt_relevant:
            fp += 1
        elif human_relevant and not gpt_relevant:
            fn += 1
        else:
            tn += 1

        human_stance = r['사람_stance(찬성/반대/중립/판단유보)'].strip() or '판단유보'
        gpt_stance = a['gpt_stance'].strip()
        stance_confusion[(human_stance, gpt_stance)] += 1
        if human_stance == gpt_stance:
            stance_correct += 1
        else:
            disagreements.append((row_id, human_relevant, gpt_relevant, human_stance, gpt_stance))

    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    relevant_acc = (tp + tn) / n if n else float('nan')

    print(f'\n=== relevant 판정 ({n}건) ===')
    print(f'정확도: {relevant_acc:.1%}  정밀도(precision): {precision:.1%}  재현율(recall): {recall:.1%}')
    print(f'TP={tp} FP={fp} FN={fn} TN={tn}')

    print(f'\n=== stance 판정 ({len(labeled)}건, relevant=false는 판단유보로 취급) ===')
    print(f'정확도: {stance_correct / len(labeled):.1%}')
    print('혼동행렬 (사람 → GPT, 건수):')
    for (h, g), c in sorted(stance_confusion.items(), key=lambda kv: -kv[1]):
        mark = '' if h == g else '  <- 불일치'
        print(f'  {h:>5} -> {g:<5} : {c}{mark}')

    if disagreements:
        print(f'\n=== 불일치 사례 ({len(disagreements)}건) ===')
        for row_id, hr, gr, hs, gs in disagreements:
            print(f'  {row_id}\n    사람: relevant={hr}, stance={hs}  |  GPT: relevant={gr}, stance={gs}')


def main():
    p = argparse.ArgumentParser(description='classify_stances 정확도 검증용 라벨셋 구축/채점')
    sub = p.add_subparsers(dest='cmd', required=True)

    ps = sub.add_parser('sample', help='라벨링 대상 CSV + GPT 판정(숨김) CSV 생성')
    ps.add_argument('--n', type=int, default=100, help='샘플 수 (기본 100)')
    ps.set_defaults(func=cmd_sample)

    pc = sub.add_parser('score', help='라벨링 결과와 GPT 판정을 비교해 정확도 계산')
    pc.set_defaults(func=cmd_score)

    pr = sub.add_parser('reclassify', help='같은 50건 라벨셋 유지하고 GPT 판정만 새로 돌려서 갱신')
    pr.set_defaults(func=cmd_reclassify)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
