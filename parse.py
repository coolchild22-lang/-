# PDF → 페이지 이미지 → Tesseract OCR → 발언 파싱
#
# collect.py가 extract_text / parse_speeches / split_text / validate_speeches를 import해서 씀.
# 이 파일 자체도 CLI로 PDF 한 장만 테스트해볼 수 있음:
#   py parse.py --pdf sample.pdf --type 본회의
#
# 의존: pip install pymupdf pytesseract numpy pillow
# 시스템 의존: Tesseract OCR (winget install tesseract-ocr.tesseract), 한국어 언어팩(kor) 포함돼 있어야 함

import argparse, os, re, sys
from io import BytesIO

import numpy as np
from PIL import Image
import pymupdf as fitz
import pytesseract

# ═══════════════════════════════════════════════════════════════════════════
# OCR: PDF → 페이지 이미지 → 텍스트
# ═══════════════════════════════════════════════════════════════════════════

DPI                = 300
LINE_BINS          = 60             # gutter 감지용 가로 밴드 개수
GUTTER_ZONE        = (0.30, 0.70)   # 페이지 중앙 30~70% 구간에서만 gutter 탐색
GUTTER_EMPTY_RATIO = 0.85           # 이 비율 이상 밴드가 비어있어야 gutter로 인정

# OCR 캐시 무효화 정책 (2026-09-22): output/ocr_cache/{conf_id}.txt 캐시는 내용
# 검증 없이 "파일이 있으면 그대로 재사용"이라, detect_gutter 같은 파싱 로직을
# 나중에 고쳐도 예전 캐시가 새 로직 반영 없이 계속 재사용되는 문제가 있었다.
# 캐시 파일명에 버전을 박아서(v{PARSE_VERSION}), OCR 결과에 영향을 주는 로직을
# 고칠 때마다 이 숫자만 올리면 예전 버전 캐시는 자동으로 무시되고 새로 OCR된다.
# 옛 버전 파일은 안 쓰인 채로 디스크에 남는데(자동 삭제 안 함), 필요하면 나중에
# 정리 스크립트로 지우면 됨.
PARSE_VERSION = 2  # 1: 최초 버전, 2: 글자 적은 페이지가 2단으로 오탐되던 문제 수정

# Tesseract 실행 경로 (winget 기본 설치 위치). 다른 경로면 TESSERACT_CMD 환경변수로 지정.
_DEFAULT_TESSERACT = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
_tess_cmd = os.getenv('TESSERACT_CMD', _DEFAULT_TESSERACT)
if os.path.exists(_tess_cmd):
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd

# 페이지 머리말/꼬리말
# 형식1: "10  제420회－제3차(2024년12월31일)"
# 형식2: "제420회－제3차(2024년12월31일)  10"
PAGE_HEADER = re.compile(
    r'^\d+\s+.+\(\d{4}년\d+월\d+일\)$'
    r'|^.+\(\d{4}년\d+월\d+일\)\s*\d+$'
)


def pdf_to_page_images(pdf_path, dpi=DPI):
    """PDF 각 페이지를 그레이스케일 PIL Image 리스트로 렌더링."""
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        im = Image.open(BytesIO(pix.tobytes('png'))).convert('L')
        images.append(im)
    doc.close()
    return images


def detect_gutter(im, line_bins=LINE_BINS, zone=GUTTER_ZONE, empty_ratio=GUTTER_EMPTY_RATIO):
    """줄 단위 공백 비율 기반 2단 컬럼 gutter 감지.

    페이지를 line_bins개 가로 밴드로 나누고, 각 x열이 각 밴드에서 잉크가 있는지 체크.
    중앙 zone 구간에서 empty_ratio 이상 비어있는 최장 구간의 중심을 컬럼 경계로 반환.
    그런 구간이 없으면 1단으로 보고 None 반환.
    회의 종류·대수로 분기하지 않고 페이지마다 직접 감지 — 위원회도 2단인 경우가 있어서.
    """
    arr = np.array(im)
    h, w = arr.shape
    bw = arr < 200  # True = 잉크(어두운 픽셀)

    band_h = max(1, h // line_bins)
    band_has_ink = np.zeros((line_bins, w), dtype=bool)
    for i in range(line_bins):
        y0, y1 = i * band_h, min((i + 1) * band_h, h)
        if y0 >= y1:
            continue
        band_has_ink[i] = bw[y0:y1].any(axis=0)

    empty_ratio_per_x = 1 - band_has_ink.mean(axis=0)

    zone_lo, zone_hi = int(w * zone[0]), int(w * zone[1])
    is_empty_enough = empty_ratio_per_x[zone_lo:zone_hi] >= empty_ratio

    best_len, best_start, cur_start = 0, 0, None
    for i, v in enumerate(is_empty_enough):
        if v:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                length = i - cur_start
                if length > best_len:
                    best_len, best_start = length, cur_start
                cur_start = None
    if cur_start is not None:
        length = len(is_empty_enough) - cur_start
        if length > best_len:
            best_len, best_start = length, cur_start

    if best_len == 0:
        return None

    gutter_x = zone_lo + best_start + best_len // 2

    # 글자가 적은 페이지(12줄 이하 등)는 중앙의 큰 빈 공간이 우연히 gutter 조건을
    # 만족해서 1단인데 2단으로 오인될 수 있다. 좌/우 양쪽에 실제로 텍스트가 있는
    # 밴드가 일정 개수 이상일 때만 2단으로 인정한다.
    left_bands  = band_has_ink[:, :gutter_x].any(axis=1).sum()
    right_bands = band_has_ink[:, gutter_x:].any(axis=1).sum()
    min_bands = max(1, line_bins // 10)
    if left_bands < min_bands or right_bands < min_bands:
        return None

    return gutter_x


def _ocr_image(im):
    return pytesseract.image_to_string(im, lang='kor', config='--psm 6')


def ocr_page(im):
    """페이지 이미지 1장을 OCR. 2단이면 gutter로 좌/우 분리 후 좌→우 순서로 이어붙임."""
    w, h = im.size
    gutter_x = detect_gutter(im)
    if gutter_x is None:
        text = _ocr_image(im)
    else:
        left  = im.crop((0, 0, gutter_x, h))
        right = im.crop((gutter_x, 0, w, h))
        text = _ocr_image(left) + '\n' + _ocr_image(right)

    lines = [l for l in text.split('\n') if not PAGE_HEADER.match(l.strip())]
    return '\n'.join(lines)


def ocr_pdf_to_text(pdf_path, cache_path=None, dpi=DPI):
    """PDF 전체를 OCR해서 텍스트로 반환. cache_path 있으면 캐싱(있으면 재사용, 없으면 생성)."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding='utf-8') as f:
            return f.read()

    images = pdf_to_page_images(pdf_path, dpi=dpi)
    full_text = '\n'.join(ocr_page(im) for im in images)

    # ◯가 줄 중간에 나오면 앞에 줄바꿈 삽입 (같은 줄에 여러 발언자 붙어있는 경우)
    full_text = re.sub(r'([^\n])◯', r'\1\n◯', full_text)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

    return full_text


def extract_text(pdf_path, conf_id=None, cache_dir=None, dpi=None):
    """PDF 텍스트 추출. 페이지를 이미지로 렌더링 후 Tesseract로 OCR.
    1단/2단 컬럼 여부는 gutter 감지로 자동 판단 — 회의 종류·대수 분기 불필요.
    conf_id + cache_dir 를 주면 OCR 결과를 캐싱해 재파싱 시 재사용(재OCR 방지).
    """
    cache_path = None
    if cache_dir and conf_id:
        cache_path = os.path.join(cache_dir, f'{conf_id}.v{PARSE_VERSION}.txt')
    kwargs = {'cache_path': cache_path}
    if dpi:
        kwargs['dpi'] = dpi
    return ocr_pdf_to_text(pdf_path, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# 발언자 마커
# ═══════════════════════════════════════════════════════════════════════════

# 원래 마커는 ◯. Tesseract OCR이 ◯를 매번 다른 괄호류 문자(또는 그 조합)로
# 잘못 읽는다 — 지금까지 관측된 것만 '(', '()', '(〉)', '<', ')' 등 제각각.
# 개별 패턴을 계속 추가하는 대신 원칙으로 처리한다: 줄 맨 앞의 괄호류 문자
# 1~3개는 전부 ◯의 오인식 후보로 본다. 진짜 괄호 지문(예: "(웃음)")과는
# match_speaker의 역할/이름 패턴 매칭 단계에서 자연히 걸러진다 — "웃음)" 같은
# 잔여 텍스트는 역할 접미사나 "의원/위원" 패턴에 안 맞아서 매칭이 실패함.
# 따옴표 등 괄호류가 아닌 문자는 제외해서 줄바꿈으로 잘린 실제 발언 내용
# (예: "'국민의 힘은…")까지 마커로 오인하지 않도록 함.
_BRACKETS = r'[(){}\[\]<>〈〉《》「」『』]'

def _is_marker_line(stripped):
    if stripped.startswith('◯'):
        return True
    return bool(re.match(_BRACKETS + '{1,3}', stripped))


# 마커(◯ 또는 괄호류 1~3자) 소비용. 뒤에 추가로 잡음 문자가 더 붙는 경우
# (예: "(-유상범 의원")도 있어서 마커 뒤 최대 3글자까지 더 건너뛴다.
_MARK = rf'(?:◯|{_BRACKETS}{{1,3}})[^가-힣]{{0,3}}'

# ── 산회 선언 ─────────────────────────────────────────────────────────────────
ADJOURN = re.compile(r'산회를 선포합니다')

# ── 목차(의사일정 목록) 조각 줄 ──────────────────────────────────────────────
# 목차의 점선 리더 사이에 있던 환각 글자를 _strip_ocr_garbage가 지우고 나면
# "부의 의 건 12"처럼 페이지 번호만 덩그러니 남거나, 줄바꿈 때문에 여는 괄호만
# 남고 안 닫힌 채 잘린 조각("...(의안")이 생김. 실제 발언은 이렇게 끝나지 않음.
# "말줄임표(…)로 끝나는 진짜 발언"(예: "80년도에도……")과는 구분해야 하므로
# 어디까지나 "숫자로 끝남" / "여는 괄호로 끝남" 신호만 본다.
TOC_LEADER = re.compile(r'[.…\s]\d{1,3}$|\([^()]{0,4}$')

# ── 의안 목록 줄 ──────────────────────────────────────────────────────────────
# 예) "1. 소득세법 일부개정법률안(유동수 의원 대표발의)(의안번호 2200134)"
#     "1. 순직 해병 수사 방해 및 사건 은폐 등의 진상규명을 위한 특별검사의 임명 등에 관한 법률안 재의의 건"
# 뒤에 (의안번호 …)가 없는 안건 제목(재의의 건 등)도 있어서 "숫자. "로 시작하는 것만으로 판단.
# 실제 발언은 "숫자. "로 줄을 시작하는 경우가 사실상 없음.
AGENDA_ITEM = re.compile(r'^\d+\.\s+\S')

# ── 타입별 행정 섹션 stop ─────────────────────────────────────────────────────
ADMIN_STOP = {
    "본회의": re.compile(
        # "의안"은 "의안번호"(=안건 제목 조각. OCR 줄바꿈으로 흔히 잘려 나옴)와 겹치면
        # 안 됨 — 실수로 걸리면 그 지점에서 전체 파싱이 멈춰버려 이후 발언이 통째로 날아감.
        rf'^{_MARK}(출석|개의|산회|청가|국회 참석자|상임위원 명단|의안(?!번호)|청원|요청서|추천|보고서|임시의석|본회의장)'
    ),
    "위원회": re.compile(
        rf'^{_MARK}(출석|산회|개의|청가|정부측)'
    ),
    "국정감사": re.compile(
        rf'^{_MARK}(출석|산회|개의|청가)'
    ),
}

# ── 발언자 역할 접미사 ────────────────────────────────────────────────────────
_ROLE_SFX = (
    # 행정부
    r'총리|부총리|장관|부장관|차관|차관보|청장|부청장'
    r'|실장|국장|과장|사무관|주무관|기획관|정책관|조사관|연구관'
    r'|감사관|조정관|관리관|대변인'
    # 국회
    r'|의장|부의장|위원장|부위원장|소위원장|전문위원'
    r'|보좌관|비서관|수석|대리'
    # 공공기관·단체
    r'|원장|부원장|총장|부총장|처장|이사장|부이사장'
    r'|본부장|단장|센터장|팀장|부장'
    r'|행장|부행장|조합장'
    r'|회장|부회장|사장|부사장|대표|이사|감사'
    r'|상임위원'
    # 지방자치
    r'|교육감|지사|부지사|시장|부시장|군수|구청장'
    # 사법·검찰
    r'|검사장|검사'
    # 직무대행·후보자
    r'|직무대행|후보자'
    # 청문회·국정감사 출석자
    r'|증인|참고인|진술인'
    # 누락 직함
    r'|총재|차장|소장|관장|사령관'
)

# Pattern A: ◯역할 이름  (역할이 접미사로 끝남)
#   예) ◯의장 우원식 / ◯국무총리 한덕수 / ◯기획재정부제1차관 김범석
SPEAKER_A = re.compile(
    rf'^{_MARK}(\S*(?:{_ROLE_SFX}))\s+([가-힣]{{2,4}})(?:\s+(.*))?$'
)

# Pattern B: ◯이름 의원/위원  (이름 앞, 역할 뒤)
#   예) ◯조배숙 의원 / ◯박성훈 위원
SPEAKER_B = re.compile(
    rf'^{_MARK}([가-힣]{{2,4}})\s+(의원|위원)(?:\s+(.*))?$'
)

# Pattern C: ◯이름  (국정감사 전용 — 역할 표기 없음)
#   예) ◯강선영 / ◯곽규택
SPEAKER_C = re.compile(
    rf'^{_MARK}([가-힣]{{2,4}})(?:\s+(.+))?$'
)


def match_speaker(stripped, conf_type, warn=False):
    """발언자 감지. (role, name, first_content) 또는 None 반환.
    warn=True 이면 마커로 시작하는데 패턴 미감지 시 경고 출력.
    """
    if not _is_marker_line(stripped):
        return None

    m = SPEAKER_A.match(stripped)
    if m:
        return m.group(1).strip(), m.group(2), m.group(3) or ''

    m = SPEAKER_B.match(stripped)
    if m:
        return m.group(2), m.group(1), m.group(3) or ''

    if conf_type == '국정감사':
        m = SPEAKER_C.match(stripped)
        if m:
            return '의원', m.group(1), m.group(2) or ''

    if warn:
        print(f'  [WARN] 발언자 미감지: {stripped[:60]}')

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 발언 파싱
# ═══════════════════════════════════════════════════════════════════════════

def _strip_ocr_garbage(text):
    """목차(의사일정 목록)의 점선 리더("………")가 OCR되면서 같은 음절 몇 개가
    반복되는 환각 텍스트로 잘못 읽히는 경우가 있음
    (예: "부의 의 건 …아에아아아아아아사아사야사아어어…… 12").
    공백으로 나눈 토큰 중 한글만 8자 이상이면서 서로 다른 글자 비율이 35% 이하면
    노이즈로 보고 제거 — 정상적인 한국어 단어는 이 정도로 낮은 글자 다양성을 안 보임.
    """
    kept = []
    for token in text.split(' '):
        core = re.sub(r'[^가-힣]', '', token)
        if len(core) >= 8 and len(set(core)) / len(core) <= 0.35:
            continue
        kept.append(token)
    return ' '.join(kept)


def _strip_parens_line(line):
    """괄호 지문 제거. 예) (일동 기립) (◯홍길동 의원 의석에서 ― ...)
    반드시 한 줄 단위로만 호출해야 함 — 전체 발언을 합친 뒤에 적용하면
    OCR이 괄호를 못 닫고 자른 조각(예: "...(의안")과 한참 뒤에 나오는
    진짜 닫힌 괄호(예: "(박수)") 사이의 실제 발언 내용까지 통째로
    지워버리는 문제가 생김 (\\([^)]*\\)가 가장 가까운 ')'까지 욕심껏 매칭하므로).
    """
    return re.sub(r'\([^)]*\)', '', line)


def _clean(text):
    """공백 정리 + OCR 노이즈 제거. 괄호 지문 제거는 각 줄이 append될 때 이미 끝남."""
    text = _strip_ocr_garbage(text)
    return ' '.join(text.split())


def validate_speeches(speeches, source=''):
    """파싱 결과 자동 검증. 의심 케이스를 콘솔에 출력."""
    tag = f'[검증] {source} ' if source else '[검증] '
    issues = 0

    circle_in = [s for s in speeches if '◯' in s['text']]
    if circle_in:
        print(f'{tag}◯ 잔류 {len(circle_in)}건')
        for s in circle_in[:3]:
            print(f'       → {s["name"]}: {s["text"][:80]}')
        issues += len(circle_in)

    too_short = [s for s in speeches if len(s['text']) < 10]
    if too_short:
        print(f'{tag}10자 미만 {len(too_short)}건')
        for s in too_short[:3]:
            print(f'       → {s["name"]}: "{s["text"]}"')
        issues += len(too_short)

    too_long = [s for s in speeches if len(s['text']) > 5000]
    if too_long:
        print(f'{tag}5000자 초과 {len(too_long)}건 (경계 누락 의심)')
        for s in too_long[:3]:
            print(f'       → {s["name"]}: {len(s["text"])}자')
        issues += len(too_long)

    hanja = [s for s in speeches if re.search(r'[一-鿿]', s['name'])]
    if hanja:
        names = list({s['name'] for s in hanja})[:5]
        print(f'{tag}이름 한자 {len(hanja)}건: {names}')
        issues += len(hanja)

    if issues == 0:
        print(f'{tag}이상 없음 (총 {len(speeches)}건)')
    else:
        print(f'{tag}이슈 {issues}건 / 전체 {len(speeches)}건')

    return issues


def parse_speeches(full_text, conf_type):
    speeches = []
    role, name, lines = None, None, []
    stop_re = ADMIN_STOP[conf_type]
    warn_count = 0

    for line in full_text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue

        if stop_re.match(stripped):
            break

        result = match_speaker(stripped, conf_type, warn=True)
        if _is_marker_line(stripped) and result is None:
            warn_count += 1
        if result:
            if name is not None:
                speeches.append({
                    'role': role, 'name': name,
                    'text': _clean(' '.join(lines)),
                    'prev_seq': len(speeches) - 1 if speeches else None,
                })
            role, name, first = result
            lines = [_strip_parens_line(first)] if first else []
        else:
            if name is not None:
                if ADJOURN.search(stripped):
                    lines.append(_strip_parens_line(stripped))
                    speeches.append({
                        'role': role, 'name': name,
                        'text': _clean(' '.join(lines)),
                        'prev_seq': len(speeches) - 1 if speeches else None,
                    })
                    name = None
                    break
                if AGENDA_ITEM.match(stripped):
                    speeches.append({
                        'role': role, 'name': name,
                        'text': _clean(' '.join(lines)),
                        'prev_seq': len(speeches) - 1 if speeches else None,
                    })
                    role, name, lines = None, None, []
                    continue
                if TOC_LEADER.search(stripped):
                    continue  # 목차 점선 리더 줄 — 발언 내용 아님
                if _is_marker_line(stripped):
                    continue  # 발언자 미감지 마커 라인 — 안건 제목 등, 내용에 섞이지 않도록 건너뜀
                lines.append(_strip_parens_line(stripped))

    if name is not None and lines:
        speeches.append({
            'role': role, 'name': name,
            'text': _clean(' '.join(lines)),
            'prev_seq': len(speeches) - 1 if speeches else None,
        })

    if warn_count:
        print(f'  [WARN 합계] 미감지 마커 라인: {warn_count}건')

    return speeches


# ═══════════════════════════════════════════════════════════════════════════
# 청크 분할
# ═══════════════════════════════════════════════════════════════════════════

def split_text(text, chunk_size=3000):
    """Recursive boundary split: \\n\\n → \\n → 。/?/! → 공백 → 글자 수 강제 분할.
    청크끼리 겹치지 않음 — 겹치는 텍스트를 매번 다시 임베딩하는 건 낭비라서 뺐음.
    문맥이 필요하면 검색 시점에 chunk_seq로 인접 청크를 가져와 이어붙여서 씀
    (analyze.py의 search() 참고)."""
    if len(text) <= chunk_size:
        return [text]

    separators = ['\n\n', '\n', '。', '?', '!', '. ', ' ']
    chunks = []
    start  = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            best = end
            search_from = start + chunk_size // 2  # 너무 짧은 청크 방지
            for sep in separators:
                idx = text.rfind(sep, search_from, end)
                if idx != -1:
                    best = idx + len(sep)
                    break
            end = best

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# CLI: PDF 한 장만 떼어내서 파싱 확인 (디버깅용)
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='국회 회의록 PDF 발언 파싱 (샘플 확인용)')
    p.add_argument('--pdf',  required=True, help='PDF 파일 경로')
    p.add_argument('--type', default='본회의',
                   choices=['본회의', '위원회', '국정감사'],
                   help='회의 종류 (기본: 본회의)')
    args = p.parse_args()

    print(f"PDF 로드: {args.pdf}  [타입: {args.type}]")
    text     = extract_text(args.pdf)
    speeches = parse_speeches(text, args.type)

    print(f"\n파싱된 발언: {len(speeches)}건")
    for i, s in enumerate(speeches, 1):
        prev      = f"←{s['prev_seq']}" if s['prev_seq'] else "   "
        role_disp = s['role'] if len(s['role']) <= 10 else s['role'][:9] + '…'
        print(f"  [{i:3}] {prev:5} {role_disp:<12} {s['name']}  ({len(s['text'])}자)")


if __name__ == '__main__':
    main()
