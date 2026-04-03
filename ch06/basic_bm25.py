from rank_bm25 import BM25Okapi  # BM25 알고리즘을 구현한 라이브러리 (문서 검색용)
from typing import List          # 타입 힌트를 위한 모듈 (가독성 및 유지보수 향상)

# 1. 말뭉치(Corpus) 정의
# 각 문장을 공백 기준으로 split()하여 "토큰 리스트" 형태로 변환
# BM25는 문자열이 아니라 "토큰 리스트(List[List[str]])"를 입력으로 사용함
corpus: List[List[str]] = [
    "에이전트 J는 패기가 넘치는 신입 대원이다".split(),  # 문장 1 → ["에이전트", "J는", ...]
    "에이전트 K는 수년간의 MIB 경험과 멋진 뉴럴라이저를 갖고 있다".split(),  # 문장 2
    "두 명의 에이전트가 검은 정장을 입고 은하계를 구했다".split(),  # 문장 3
]

# 2. BM25 인덱스 생성
# corpus를 기반으로 각 단어의 중요도(IDF), 문서 길이 등을 계산하여
# 빠르게 검색할 수 있는 내부 인덱스를 생성
bm25 = BM25Okapi(corpus)

# 3. 검색할 쿼리 정의
# 사용자 입력 문장을 동일하게 토큰화(split) 처리
# (중요: corpus와 동일한 방식으로 전처리해야 정확한 검색 가능)
query = "신입 대원은 누구지?".split()

# 4. BM25 검색 수행
# get_top_n:
# - query: 검색할 단어 리스트
# - corpus: 원본 문서 리스트
# - n: 상위 몇 개 결과를 반환할지
# 내부적으로 BM25 점수를 계산하여 높은 순으로 정렬 후 반환
top_n = bm25.get_top_n(query, corpus, n=2)

# 5. 결과 출력
# 토큰 리스트를 다시 문자열로 join하여 사람이 읽기 쉽게 변환
print("쿼리:", " ".join(query))  # ["신입", "대원은", "누구지?"] → "신입 대원은 누구지?"

print("상위 일치 문장:")
for line in top_n:
    # 각 결과도 토큰 리스트이므로 join으로 다시 문장 형태로 출력
    print(" •", " ".join(line))