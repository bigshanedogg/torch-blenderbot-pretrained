import pytest

def test_tokenizer_kor(mecab_tokenizer):
    test_sentences = [
        "351번째 문제와 삼천오백구번 문제는 해결했다.",
        "삼팔구호기에서 예닐곱개의 현상 관측해 세개의 보고서를 작성했다.",
        "약 구십여개의 에러가 발생했는데, 그 중 삼천팔백팔번째 문제만 남았다.",
        "세네번째 문제는 어렵지만 푸는 게 不可能하지는 않다."
    ]

    print("\n########### test_tokenizer_kor ###########")
    for test_sentence in test_sentences:
        test_result = mecab_tokenizer.tokenize_kor(test_sentence, return_pos=True)
        assert len(test_result) > 0
        assert len(test_result[0]) == 2

def test_tokenizer_eng(mecab_tokenizer):
    test_sentences = [
        "Hugging Face is a technology company based in New York and Paris.",
        "Do you really think so? I don't. It will just make us fat and act silly.",
        "I guess you are right. But what shall we do? I don't feel like sitting at home."
    ]

    print("\n########### test_tokenizer_eng ###########")
    for test_sentence in test_sentences:
        test_result = mecab_tokenizer.tokenize_eng(test_sentence, return_pos=True)
        assert len(test_result) > 0
        assert len(test_result[0]) == 2

def test_pos_normalize_interjection(mecab_tokenizer):
    test_sentences = [
        "ㅋㅋㅋㅋㅋㅋㅋㅋㅋ이렇게 많이 웃는, 경우에도 작동합니까?ㅋㅋㅋㅋㅋㅋ",
        "하하하하하하하핫 이게 잘 작동할까아아아아아아요?"
    ]

    print("\n########### test_pos_normalize_interjection ###########")
    for test_sentence in test_sentences:
        _test_result = mecab_tokenizer.pos(test_sentence)
        test_result = " ".join([word for word, pos in _test_result])
        print(test_result)
        assert len(test_result) > 0

def test_pos_normalize_arabia(mecab_tokenizer_normalize_arabia):
    test_sentences = [
        "351번째 문제와 삼천오백구번 문제는 해결했다.",
        "삼팔구호기에서 예닐곱개의 현상 관측해 세개의 보고서를 작성했다.",
        "약 구십여개의 에러가 발생했는데, 그 중 삼천팔백팔번째 문제만 남았다.",
        "세네번째 문제는 어렵지만 푸는 게 不可能하지는 않다."
    ]

    print("\n########### test_pos_normalize_arabia ###########")
    for test_sentence in test_sentences:
        _test_result = mecab_tokenizer_normalize_arabia.pos(test_sentence)
        test_result = " ".join([word for word, pos in _test_result])
        print(test_result)
        assert len(test_result) > 0

def test_pos_normalize_chinese(mecab_tokenizer_normalize_chinese):
    test_sentences = [
        "현상 관측 결과 이상 無",
        "利得을 본 경기였다.",
        "상태 체크 보고, 동작 不"
    ]

    print("\n########### test_pos_normalize_chinese ###########")
    for test_sentence in test_sentences:
        _test_result = mecab_tokenizer_normalize_chinese.pos(test_sentence)
        test_result = " ".join([word for word, pos in _test_result])
        print(test_result)
        assert len(test_result) > 0

def test_extract_keywords(mecab_tokenizer):
    test_sentences = [
        "351번째 문제와 삼천오백구번 문제는 해결했다.",
        "삼팔구호기에서 예닐곱개의 현상 관측해 세개의 보고서를 작성했다.",
        "약 구십여개의 에러가 발생했는데, 그 중 삼천팔백팔번째 문제만 남았다.",
        "세네번째 문제는 어렵지만 푸는 게 不可能하지는 않다."
    ]

    print("\n########### test_extract_keywords ###########")
    for test_sentence in test_sentences:
        test_result = mecab_tokenizer.extract_keywords(test_sentence)
        assert isinstance(test_result, list)