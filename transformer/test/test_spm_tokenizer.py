# import pytest
#
# def test_pos_arabia_normalize(advanced_mecab_fixture):
#     test_sentences = [
#         "351번째 문제와 삼천오백구번 문제는 해결했다.",
#         "삼팔구호기에서 예닐곱개의 현상 관측해 세개의 보고서를 작성했다.",
#         "약 구십여개의 에러가 발생했는데, 그 중 삼천팔백팔번째 문제만 남았다.",
#         "세네번째 문제는 어렵지만 푸는 게 不可能하지는 않다."
#     ]
#
#     print("\n########### arabia normalize test ###########")
#     for test_sentence in test_sentences:
#         _test_result = advanced_mecab_fixture.pos(test_sentence, normalize_arabia=True, normalize_chinese=False)
#         test_result = " ".join([word for word, pos in _test_result])
#         print(test_result)
#         assert len(test_result) > 0
#
# def test_pos_chinese_normalize(advanced_mecab_fixture):
#     test_sentences = [
#         "현상 관측 결과 이상 無",
#         "利得을 본 경기였다.",
#         "상태 체크 보고, 동작 不"
#     ]
#
#     print("\n########### chinese normalize test ###########")
#     for test_sentence in test_sentences:
#         _test_result = advanced_mecab_fixture.pos(test_sentence, normalize_arabia=False, normalize_chinese=True)
#         test_result = " ".join([word for word, pos in _test_result])
#         print(test_result)
#         assert len(test_result) > 0
#
# def test_pos_interjection_normalize(advanced_mecab_fixture):
#     test_sentences = [
#         "ㅋㅋㅋㅋㅋㅋㅋㅋㅋ이렇게 많이 웃는, 경우에도 작동합니까?ㅋㅋㅋㅋㅋㅋ",
#         "하하하하하하하핫 이게 잘 작동할까아아아아아아요?"
#     ]
#
#     print("\n########### interjection normalize test ###########")
#     for test_sentence in test_sentences:
#         _test_result = advanced_mecab_fixture.pos(test_sentence, normalize_arabia=False, normalize_chinese=False)
#         test_result = " ".join([word for word, pos in _test_result])
#         print(test_result)
#         assert len(test_result) > 0
#
# def test_extract_keywords(advanced_mecab_fixture):
#     test_sentences = [
#         "351번째 문제와 삼천오백구번 문제는 해결했다.",
#         "삼팔구호기에서 예닐곱개의 현상 관측해 세개의 보고서를 작성했다.",
#         "약 구십여개의 에러가 발생했는데, 그 중 삼천팔백팔번째 문제만 남았다.",
#         "세네번째 문제는 어렵지만 푸는 게 不可能하지는 않다."
#     ]
#
#     print("\n########### interjection normalize test ###########")
#     for test_sentence in test_sentences:
#         test_result = advanced_mecab_fixture.extract_keywords(test_sentence)
#         assert isinstance(test_result, list)